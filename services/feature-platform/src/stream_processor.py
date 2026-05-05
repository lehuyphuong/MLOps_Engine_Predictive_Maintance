"""
stream_processor.py — Feature Platform Stream Processor

Implements a Flink-style micro-batch pipeline in pure Python:
  Source   => reads validated cycle events from GCS
  Map      => feature engineering (drop low-correlation sensors, cap RUL)
  Sink 1   => Redis online store  (rolling 24-cycle window per engine)
  Sink 2   => PostgreSQL offline store (engine_features table)

Architecture note:
  Apache Flink is the production-grade choice for this pipeline and is
  shown in the system diagram. PyFlink (the Python Flink client) is
  incompatible with Python 3.11+ due to a hardcoded numpy==1.21.4
  dependency in all current releases. This implementation replicates
  the same Source => Map => Sink operator pattern using a lightweight
  MicroBatchPipeline class so the code structure mirrors what a real
  Flink job would look like, making migration straightforward once
  PyFlink resolves the Python 3.11 compatibility issue.

Feature engineering (matching existing preprocessing script):
  Drop low-correlation sensors for FD002:
    OperSet1, OperSet2, OperSet3, SensorMes1, SensorMes5,
    SensorMes10, SensorMes18, SensorMes19
  Keep remaining 16 sensor/opset columns as features
  RUL cap at 125 cycles (piecewise linear — standard C-MAPSS practice)

Environment variables (from feature-platform-config ConfigMap):
  GCS_RAW_BUCKET    phm-raw-data-aide2-494008
  DATASET           FD002
  GCP_PROJECT       aide2-494008
  REDIS_HOST        from Memorystore — terraform output redis_host
  REDIS_PORT        6379
  DB_HOST           from Cloud SQL  — terraform output db_private_ip
  DB_PORT           5432
  DB_NAME           phmdb
  DB_USER           phmadmin
  DB_PASSWORD       from feature-platform-secrets
  WINDOW_SIZE       24
  RUL_CAP           125
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Callable, Iterator

import redis
import psycopg2
from google.cloud import storage
from google.cloud.exceptions import NotFound

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GCS_RAW_BUCKET = os.environ.get("GCS_RAW_BUCKET", "phm-raw-data-aide2-494008")
DATASET        = os.environ.get("DATASET",         "FD002")
GCP_PROJECT    = os.environ.get("GCP_PROJECT",     "aide2-494008")

REDIS_HOST     = os.environ.get("REDIS_HOST",  "localhost")
REDIS_PORT     = int(os.environ.get("REDIS_PORT", "6379"))
WINDOW_SIZE    = int(os.environ.get("WINDOW_SIZE",  "24"))

DB_HOST        = os.environ.get("DB_HOST",     "localhost")
DB_PORT        = int(os.environ.get("DB_PORT", "5432"))
DB_NAME        = os.environ.get("DB_NAME",     "phmdb")
DB_USER        = os.environ.get("DB_USER",     "phmadmin")
DB_PASSWORD    = os.environ.get("DB_PASSWORD", "")

RUL_CAP        = int(os.environ.get("RUL_CAP", "125"))

VALID_PREFIX   = f"validated_engine_cycles/{DATASET}/"
CHECKPOINT_KEY = f"checkpoints/{DATASET}/feature_processed_keys.json"

# ---------------------------------------------------------------------------
# FD002 feature selection
# ---------------------------------------------------------------------------
COLS_TO_DROP = {
    "OperSet1", "OperSet2", "OperSet3",
    "SensorMes1", "SensorMes5", "SensorMes10",
    "SensorMes18", "SensorMes19",
}

FEATURE_COLS = (
    [f"OperSet{i}"   for i in range(1, 4)  if f"OperSet{i}"   not in COLS_TO_DROP]
    + [f"SensorMes{j}" for j in range(1, 22) if f"SensorMes{j}" not in COLS_TO_DROP]
)

# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------
gcs = storage.Client(project=GCP_PROJECT)


def gcs_list_validated_keys() -> list[str]:
    blobs = gcs.bucket(GCS_RAW_BUCKET).list_blobs(prefix=VALID_PREFIX)
    return [
        b.name for b in blobs
        if b.name.endswith(".json") and "checkpoint" not in b.name
    ]


def gcs_read_event(key: str) -> dict:
    blob = gcs.bucket(GCS_RAW_BUCKET).blob(key)
    return json.loads(blob.download_as_text())


def load_processed_keys() -> set:
    try:
        blob = gcs.bucket(GCS_RAW_BUCKET).blob(CHECKPOINT_KEY)
        return set(json.loads(blob.download_as_text()).get("processed", []))
    except NotFound:
        return set()


def save_processed_keys(keys: set) -> None:
    gcs.bucket(GCS_RAW_BUCKET).blob(CHECKPOINT_KEY).upload_from_string(
        json.dumps({
            "processed":  list(keys),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }),
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def extract_features(event: dict) -> dict:
    sensors = event.get("sensors", {})
    return {col: sensors[col] for col in FEATURE_COLS if col in sensors}


def cap_rul(rul: int) -> int:
    return min(rul, RUL_CAP)


# ---------------------------------------------------------------------------
# Connections — lazy singletons
# ---------------------------------------------------------------------------
_redis_conn = None
_pg_conn    = None


def get_redis() -> redis.Redis:
    global _redis_conn
    if _redis_conn is None:
        _redis_conn = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT,
            decode_responses=True, socket_connect_timeout=5,
        )
    return _redis_conn


def get_pg_conn():
    global _pg_conn
    if _pg_conn is None or _pg_conn.closed:
        _pg_conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT,
            dbname=DB_NAME, user=DB_USER,
            password=DB_PASSWORD, connect_timeout=10,
        )
    return _pg_conn


# ---------------------------------------------------------------------------
# PostgreSQL schema bootstrap
# ---------------------------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS engine_features (
    id           SERIAL PRIMARY KEY,
    dataset      VARCHAR(10)  NOT NULL,
    unit_id      INTEGER      NOT NULL,
    cycle        INTEGER      NOT NULL,
    rul          INTEGER,
    rul_capped   INTEGER,
    features     JSONB        NOT NULL,
    processed_at TIMESTAMPTZ  DEFAULT NOW(),
    source_key   TEXT,
    UNIQUE (dataset, unit_id, cycle)
);
CREATE INDEX IF NOT EXISTS idx_engine_features_unit
    ON engine_features (dataset, unit_id, cycle);
"""


def ensure_schema() -> None:
    conn = get_pg_conn()
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    log.info("PostgreSQL schema ready — table: engine_features")


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------
def sink_redis(unit_id: int, features: dict, cycle: int) -> None:
    """Rolling 24-cycle window per engine."""
    r      = get_redis()
    key    = f"engine:{unit_id}:window"
    raw    = r.get(key)
    window = json.loads(raw) if raw else []

    window.append({"cycle": cycle, "features": features})
    if len(window) > WINDOW_SIZE:
        window = window[-WINDOW_SIZE:]

    r.set(key, json.dumps(window))
    r.set(f"engine:{unit_id}:latest", json.dumps({
        "cycle":      cycle,
        "features":   features,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }))
    log.debug(f"Redis updated — engine:{unit_id} window_len={len(window)}")


def sink_postgres(event: dict, features: dict, source_key: str) -> None:
    """Upsert one feature row into engine_features."""
    conn = get_pg_conn()
    sql  = """
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
            "unit_id":    int(event["unit_id"]),
            "cycle":      int(event["cycle"]),
            "rul":        None,   # computed at training time from max_cycle
            "rul_capped": None,
            "features":   json.dumps(features),
            "source_key": source_key,
        })


# ---------------------------------------------------------------------------
# Flink-style micro-batch pipeline
# Mirrors the Flink operator model: Source => Map => Sinks
# Each method corresponds to a Flink concept:
#   source()   — DataStream source (GCS blob list)
#   map()      — stateless transformation operator
#   execute()  — triggers the pipeline, equivalent to env.execute()
# ---------------------------------------------------------------------------
class MicroBatchPipeline:
    """
    Lightweight Flink-style pipeline for micro-batch processing.
    Designed to be drop-in replaceable with a real PyFlink pipeline
    once PyFlink resolves Python 3.11 compatibility.

    Usage mirrors Flink's DataStream API:
      pipeline = MicroBatchPipeline(name="phm-stream-processor")
      pipeline.source(records)      # DataStream source
               .map(transform_fn)   # stateless map operator
               .execute()           # trigger execution
    """

    def __init__(self, name: str):
        self.name       = name
        self._source    = []
        self._map_fn: Callable | None = None
        self.processed  = 0
        self.errors     = 0

    def source(self, records: list) -> "MicroBatchPipeline":
        """Register the input collection — equivalent to env.from_collection()."""
        self._source = records
        log.info(f"[{self.name}] Source registered — {len(records)} records")
        return self

    def map(self, fn: Callable) -> "MicroBatchPipeline":
        """Register a stateless map operator."""
        self._map_fn = fn
        return self

    def execute(self) -> "MicroBatchPipeline":
        """
        Execute the pipeline — equivalent to env.execute().
        Applies the map function to each record in the source collection.
        Commits PostgreSQL every 50 records to keep transactions small.
        """
        log.info(f"[{self.name}] Pipeline executing — parallelism=1 (local mode)")

        for record in self._source:
            try:
                self._map_fn(record)
                self.processed += 1

                if self.processed % 50 == 0:
                    get_pg_conn().commit()
                    log.info(f"[{self.name}] Progress — {self.processed} records processed")

            except Exception as e:
                log.error(f"[{self.name}] Map operator error: {e}")
                self.errors += 1

        # Final commit
        if _pg_conn and not _pg_conn.closed:
            _pg_conn.commit()

        log.info(
            f"[{self.name}] Execution complete — "
            f"processed={self.processed} errors={self.errors}"
        )
        return self


# ---------------------------------------------------------------------------
# Map operator function — applied to each record by the pipeline
# ---------------------------------------------------------------------------
def process_record(record: dict) -> None:
    """
    Stateless map operator — equivalent to a Flink MapFunction.
    Receives one event payload, applies feature engineering,
    writes to both sinks.
    """
    event    = record["event"]
    key      = record["key"]
    unit_id  = int(event["unit_id"])
    cycle    = int(event["cycle"])
    features = extract_features(event)

    sink_redis(unit_id, features, cycle)
    sink_postgres(event, features, key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info(
        f"Stream processor starting — "
        f"dataset={DATASET} window={WINDOW_SIZE} "
        f"mode=micro-batch (Flink-compatible API)"
    )

    ensure_schema()

    # Collect pending keys from GCS
    all_keys       = gcs_list_validated_keys()
    processed_keys = load_processed_keys()
    pending_keys   = [k for k in all_keys if k not in processed_keys]

    log.info(f"Total={len(all_keys)} Pending={len(pending_keys)}")

    if not pending_keys:
        log.info("All validated events already processed.")
        return

    # Load event payloads from GCS — Source stage
    records     = []
    load_errors = 0
    for key in pending_keys:
        try:
            event = gcs_read_event(key)
            records.append({"event": event, "key": key})
        except Exception as e:
            log.error(f"Failed to load {key}: {e}")
            load_errors += 1

    log.info(f"Loaded {len(records)} events ({load_errors} load errors)")

    if not records:
        log.info("No events loaded. Exiting.")
        return

    # Execute pipeline — Source => Map => Sinks
    pipeline = MicroBatchPipeline(name="phm-stream-processor")
    pipeline.source(records) \
            .map(process_record) \
            .execute()

    # Close connections
    if _pg_conn and not _pg_conn.closed:
        _pg_conn.close()

    # Update checkpoint
    for record in records:
        processed_keys.add(record["key"])
    save_processed_keys(processed_keys)

    log.info(
        f"Stream processor done — "
        f"processed={pipeline.processed} "
        f"errors={pipeline.errors + load_errors} "
        f"feature_cols={len(FEATURE_COLS)}"
    )


if __name__ == "__main__":
    main()