"""
stream_processor.py — Feature Platform Stream Processor

Reads validated cycle events from S3 (validated_engine_cycles/),
then writes to two stores:

  Redis (online store)
    Key   : engine:{unit_id}:window
    Value : JSON list of last 24 cycle feature dicts (rolling window)
    Used by: model-serving for real-time RUL inference

  PostgreSQL (offline store)
    Table : engine_features
    Rows  : one row per cycle, all feature columns + RUL label
    Used by: model-training for batch training

Feature engineering applied (matching existing preprocessing script):
  - Drop low-correlation sensors for FD002:
      OperSet1, OperSet2, OperSet3, SensorMes1, SensorMes5, SensorMes10,
      SensorMes18, SensorMes19
  - Keep remaining 16 sensor/opset columns as features
  - RUL cap at 125 cycles (piecewise linear — standard C-MAPSS practice)

Environment variables:
  S3_RAW_BUCKET    phm-raw-data
  DATASET          FD002
  AWS_REGION       ap-southeast-1
  REDIS_HOST       from ElastiCache endpoint
  REDIS_PORT       6379
  DB_HOST          from RDS endpoint
  DB_PORT          5432
  DB_NAME          phmdb
  DB_USER          phmadmin
  DB_PASSWORD      from RDS secret
  WINDOW_SIZE      24  (rolling window length)
  RUL_CAP          125
"""

import os
import json
import logging
import boto3
import redis
import psycopg2
import psycopg2.extras
import pandas as pd
from datetime import datetime, timezone
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
S3_RAW_BUCKET = os.environ.get("S3_RAW_BUCKET", "phm-raw-data")
DATASET       = os.environ.get("DATASET",        "FD002")
AWS_REGION    = os.environ.get("AWS_REGION",     "ap-southeast-1")

REDIS_HOST    = os.environ.get("REDIS_HOST",  "localhost")
REDIS_PORT    = int(os.environ.get("REDIS_PORT", "6379"))
WINDOW_SIZE   = int(os.environ.get("WINDOW_SIZE", "24"))

DB_HOST       = os.environ.get("DB_HOST",     "localhost")
DB_PORT       = int(os.environ.get("DB_PORT", "5432"))
DB_NAME       = os.environ.get("DB_NAME",     "phmdb")
DB_USER       = os.environ.get("DB_USER",     "phmadmin")
DB_PASSWORD   = os.environ.get("DB_PASSWORD", "")

RUL_CAP       = int(os.environ.get("RUL_CAP", "125"))

VALID_PREFIX   = f"validated_engine_cycles/{DATASET}/"
CHECKPOINT_KEY = f"checkpoints/{DATASET}/feature_processed_keys.json"

# ---------------------------------------------------------------------------
# FD002 feature selection — drop low-correlation sensors (from existing script)
# Columns dropped: OperSet1, OperSet2, OperSet3, SensorMes1, SensorMes5,
#                  SensorMes10, SensorMes18, SensorMes19
# ---------------------------------------------------------------------------
COLS_TO_DROP = [
    "OperSet1", "OperSet2", "OperSet3",
    "SensorMes1", "SensorMes5", "SensorMes10",
    "SensorMes18", "SensorMes19",
]

FEATURE_COLS = (
    [f"OperSet{i}" for i in range(1, 4)
     if f"OperSet{i}" not in COLS_TO_DROP]
    + [f"SensorMes{j}" for j in range(1, 22)
       if f"SensorMes{j}" not in COLS_TO_DROP]
)

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
s3_client = boto3.client("s3", region_name=AWS_REGION)


def get_redis() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=5,
    )


def get_pg_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=10,
    )


# ---------------------------------------------------------------------------
# PostgreSQL schema bootstrap
# ---------------------------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS engine_features (
    id              SERIAL PRIMARY KEY,
    dataset         VARCHAR(10)   NOT NULL,
    unit_id         INTEGER       NOT NULL,
    cycle           INTEGER       NOT NULL,
    rul             INTEGER,
    rul_capped      INTEGER,
    features        JSONB         NOT NULL,
    processed_at    TIMESTAMPTZ   DEFAULT NOW(),
    source_key      TEXT,
    UNIQUE (dataset, unit_id, cycle)
);
CREATE INDEX IF NOT EXISTS idx_engine_features_unit
    ON engine_features (dataset, unit_id, cycle);
"""


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    log.info("PostgreSQL schema ready — table: engine_features")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def extract_features(event: dict) -> dict:
    """
    Extract and select features from a validated event.
    Drops low-correlation sensors identified for FD002.
    """
    sensors = event.get("sensors", {})
    return {col: sensors[col] for col in FEATURE_COLS if col in sensors}


def cap_rul(rul: int) -> int:
    """
    Piecewise linear RUL cap — standard C-MAPSS practice.
    Engines far from failure are treated as equally healthy.
    """
    return min(rul, RUL_CAP)


# ---------------------------------------------------------------------------
# Redis — rolling 24-cycle window
# ---------------------------------------------------------------------------
def update_redis_window(r: redis.Redis, unit_id: int, features: dict, cycle: int) -> None:
    """
    Maintain a rolling window of the last WINDOW_SIZE cycles per engine.
    Key pattern: engine:{unit_id}:window
    Value: JSON list, newest cycle appended, oldest dropped when > WINDOW_SIZE
    """
    key     = f"engine:{unit_id}:window"
    entry   = {"cycle": cycle, "features": features}

    # Load existing window
    raw     = r.get(key)
    window  = json.loads(raw) if raw else []

    # Append new cycle and trim to window size
    window.append(entry)
    if len(window) > WINDOW_SIZE:
        window = window[-WINDOW_SIZE:]

    r.set(key, json.dumps(window))

    # Also store latest state for quick inference lookup
    r.set(f"engine:{unit_id}:latest", json.dumps({
        "cycle":        cycle,
        "features":     features,
        "updated_at":   datetime.now(timezone.utc).isoformat(),
    }))

    log.debug(f"Redis updated — engine:{unit_id} window_len={len(window)}")


# ---------------------------------------------------------------------------
# PostgreSQL — offline feature store
# ---------------------------------------------------------------------------
def upsert_feature_row(
    conn,
    event: dict,
    features: dict,
    source_key: str,
) -> None:
    """
    Insert or update one feature row in engine_features.
    RUL is derived from the event row_index and total cycles — approximated
    here as None until the model-training job computes it from the full dataset.
    """
    unit_id = int(event["unit_id"])
    cycle   = int(event["cycle"])

    # RUL not available in live stream events — set to NULL
    # model-training computes RUL from max_cycle - current_cycle
    rul        = None
    rul_capped = None

    sql = """
        INSERT INTO engine_features
            (dataset, unit_id, cycle, rul, rul_capped, features, source_key)
        VALUES
            (%(dataset)s, %(unit_id)s, %(cycle)s, %(rul)s, %(rul_capped)s,
             %(features)s, %(source_key)s)
        ON CONFLICT (dataset, unit_id, cycle)
        DO UPDATE SET
            features     = EXCLUDED.features,
            source_key   = EXCLUDED.source_key,
            processed_at = NOW();
    """

    with conn.cursor() as cur:
        cur.execute(sql, {
            "dataset":    DATASET,
            "unit_id":    unit_id,
            "cycle":      cycle,
            "rul":        rul,
            "rul_capped": rul_capped,
            "features":   json.dumps(features),
            "source_key": source_key,
        })


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_processed_keys() -> set:
    try:
        obj  = s3_client.get_object(Bucket=S3_RAW_BUCKET, Key=CHECKPOINT_KEY)
        data = json.loads(obj["Body"].read())
        return set(data.get("processed", []))
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return set()
        raise


def save_processed_keys(keys: set) -> None:
    s3_client.put_object(
        Bucket=S3_RAW_BUCKET,
        Key=CHECKPOINT_KEY,
        Body=json.dumps({
            "processed":  list(keys),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }),
        ContentType="application/json",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info(f"Stream processor starting — dataset={DATASET} window={WINDOW_SIZE}")

    # Connect to stores
    r    = get_redis()
    conn = get_pg_conn()
    ensure_schema(conn)

    # List validated events not yet processed
    paginator = s3_client.get_paginator("list_objects_v2")
    all_keys  = [
        obj["Key"]
        for page in paginator.paginate(Bucket=S3_RAW_BUCKET, Prefix=VALID_PREFIX)
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".json") and "checkpoint" not in obj["Key"]
    ]

    if not all_keys:
        log.info("No validated events found. Nothing to process.")
        conn.close()
        return

    processed_keys = load_processed_keys()
    pending_keys   = [k for k in all_keys if k not in processed_keys]

    log.info(f"Total={len(all_keys)} Pending={len(pending_keys)}")

    if not pending_keys:
        log.info("All validated events already processed.")
        conn.close()
        return

    processed = 0
    errors    = 0

    for key in pending_keys:
        try:
            obj      = s3_client.get_object(Bucket=S3_RAW_BUCKET, Key=key)
            event    = json.loads(obj["Body"].read())

            unit_id  = int(event["unit_id"])
            cycle    = int(event["cycle"])
            features = extract_features(event)

            # Write to Redis online store
            update_redis_window(r, unit_id, features, cycle)

            # Write to PostgreSQL offline store
            upsert_feature_row(conn, event, features, key)

            processed_keys.add(key)
            processed += 1

            # Commit every 50 rows to avoid large transactions
            if processed % 50 == 0:
                conn.commit()
                log.info(f"Progress — processed {processed} events")

        except Exception as e:
            log.error(f"Failed to process {key}: {e}")
            errors += 1

    conn.commit()
    conn.close()

    save_processed_keys(processed_keys)

    log.info(
        f"Stream processor done — "
        f"processed={processed} errors={errors} "
        f"feature_cols={len(FEATURE_COLS)}"
    )


if __name__ == "__main__":
    main()