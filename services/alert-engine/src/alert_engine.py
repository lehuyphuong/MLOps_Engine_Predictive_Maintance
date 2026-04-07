"""
alert_engine.py — Alert Rule Engine
Polls S3 inference-results/ for new engine_id · RUL · anomaly_score · anomaly_flag events.
Evaluates two alert rules:
  Rule 1 — RUL threshold  : RUL < RUL_ALERT_THRESHOLD (default 20 cycles)
  Rule 2 — Anomaly flag   : anomaly_flag == True

For each triggered rule:
  - Sends event to Elasticsearch alert index
  - Logs alert with severity level

Runs as a Kubernetes CronJob (every 5 min, after model-serving writes results).

Environment variables:
  S3_RAW_BUCKET          phm-raw-data
  DATASET                FD002
  AWS_REGION             ap-southeast-1
  RUL_ALERT_THRESHOLD    20
  ES_HOST                Elasticsearch ClusterIP
  ES_PORT                9200
  ES_INDEX               phm-alerts
"""

import os
import json
import logging
import boto3
from datetime import datetime, timezone
from botocore.exceptions import ClientError
from elasticsearch import Elasticsearch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
S3_RAW_BUCKET       = os.environ.get("S3_RAW_BUCKET",       "phm-raw-data")
DATASET             = os.environ.get("DATASET",              "FD002")
AWS_REGION          = os.environ.get("AWS_REGION",           "ap-southeast-1")
RUL_ALERT_THRESHOLD = int(os.environ.get("RUL_ALERT_THRESHOLD", "20"))
ES_HOST             = os.environ.get("ES_HOST",              "elasticsearch-svc")
ES_PORT             = int(os.environ.get("ES_PORT",          "9200"))
ES_INDEX            = os.environ.get("ES_INDEX",             "phm-alerts")

RESULTS_PREFIX  = f"inference-results/{DATASET}/"
CHECKPOINT_KEY  = f"checkpoints/{DATASET}/alert_processed_keys.json"

s3 = boto3.client("s3", region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def load_processed_keys() -> set:
    try:
        obj  = s3.get_object(Bucket=S3_RAW_BUCKET, Key=CHECKPOINT_KEY)
        data = json.loads(obj["Body"].read())
        return set(data.get("processed", []))
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return set()
        raise


def save_processed_keys(keys: set) -> None:
    s3.put_object(
        Bucket=S3_RAW_BUCKET,
        Key=CHECKPOINT_KEY,
        Body=json.dumps({
            "processed":  list(keys),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }),
        ContentType="application/json",
    )


# ---------------------------------------------------------------------------
# Alert Rule Engine
# ---------------------------------------------------------------------------
def evaluate_rules(event: dict) -> list[dict]:
    """
    Evaluate alert rules against an inference event.
    Returns list of triggered alerts (empty if no rules fired).
    """
    alerts    = []
    engine_id = event.get("engine_id")
    cycle     = event.get("cycle")
    rul       = event.get("RUL")
    anomaly_score = event.get("anomaly_score", 0.0)
    anomaly_flag  = event.get("anomaly_flag",  False)
    timestamp     = event.get("timestamp", datetime.now(timezone.utc).isoformat())

    # Rule 1 — RUL threshold
    if rul is not None and rul < RUL_ALERT_THRESHOLD:
        severity = "critical" if rul < 10 else "warning"
        alerts.append({
            "alert_type":   "rul_threshold",
            "severity":     severity,
            "engine_id":    engine_id,
            "cycle":        cycle,
            "RUL":          rul,
            "anomaly_score": anomaly_score,
            "anomaly_flag": anomaly_flag,
            "rule":         f"RUL < {RUL_ALERT_THRESHOLD}",
            "message":      f"Engine {engine_id} approaching failure — RUL={rul:.1f} cycles",
            "dataset":      DATASET,
            "event_time":   timestamp,
            "indexed_at":   datetime.now(timezone.utc).isoformat(),
        })
        log.warning(
            f"[{severity.upper()}] Engine {engine_id} cycle {cycle} — "
            f"RUL={rul:.1f} < threshold={RUL_ALERT_THRESHOLD}"
        )

    # Rule 2 — Anomaly flag
    if anomaly_flag:
        alerts.append({
            "alert_type":   "anomaly_detected",
            "severity":     "warning",
            "engine_id":    engine_id,
            "cycle":        cycle,
            "RUL":          rul,
            "anomaly_score": anomaly_score,
            "anomaly_flag": anomaly_flag,
            "rule":         "anomaly_flag == True",
            "message":      (
                f"Engine {engine_id} anomaly detected — "
                f"score={anomaly_score:.4f} (HPC/fan degradation)"
            ),
            "dataset":      DATASET,
            "event_time":   timestamp,
            "indexed_at":   datetime.now(timezone.utc).isoformat(),
        })
        log.warning(
            f"[WARNING] Engine {engine_id} cycle {cycle} — "
            f"anomaly_flag=True score={anomaly_score:.4f}"
        )

    return alerts


# ---------------------------------------------------------------------------
# Elasticsearch — Alert Index
# ---------------------------------------------------------------------------
def get_es_client() -> Elasticsearch:
    return Elasticsearch(
        f"http://{ES_HOST}:{ES_PORT}",
        request_timeout=10,
        retry_on_timeout=True,
        max_retries=3,
    )


def ensure_index(es: Elasticsearch) -> None:
    """Create alert index with mapping if it doesn't exist."""
    if es.indices.exists(index=ES_INDEX):
        return

    mapping = {
        "mappings": {
            "properties": {
                "alert_type":    {"type": "keyword"},
                "severity":      {"type": "keyword"},
                "engine_id":     {"type": "integer"},
                "cycle":         {"type": "integer"},
                "RUL":           {"type": "float"},
                "anomaly_score": {"type": "float"},
                "anomaly_flag":  {"type": "boolean"},
                "rule":          {"type": "keyword"},
                "message":       {"type": "text"},
                "dataset":       {"type": "keyword"},
                "event_time":    {"type": "date"},
                "indexed_at":    {"type": "date"},
            }
        },
        "settings": {
            "number_of_shards":   1,
            "number_of_replicas": 0,   # single node — no replicas needed
        }
    }

    es.indices.create(index=ES_INDEX, body=mapping)
    log.info(f"Created Elasticsearch index: {ES_INDEX}")


def index_alert(es: Elasticsearch, alert: dict) -> None:
    """Index a single alert document."""
    es.index(index=ES_INDEX, document=alert)
    log.info(
        f"Indexed alert — type={alert['alert_type']} "
        f"engine={alert['engine_id']} severity={alert['severity']}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info(
        f"Alert engine starting — dataset={DATASET} "
        f"rul_threshold={RUL_ALERT_THRESHOLD}"
    )

    # Connect to Elasticsearch
    es = get_es_client()
    try:
        info = es.info()
        log.info(f"Elasticsearch connected — version={info['version']['number']}")
    except Exception as e:
        log.error(f"Cannot connect to Elasticsearch at {ES_HOST}:{ES_PORT} — {e}")
        raise

    ensure_index(es)

    # List new inference result files from S3
    paginator = s3.get_paginator("list_objects_v2")
    all_keys  = [
        obj["Key"]
        for page in paginator.paginate(Bucket=S3_RAW_BUCKET, Prefix=RESULTS_PREFIX)
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".json") and "checkpoint" not in obj["Key"]
    ]

    if not all_keys:
        log.info("No inference results found. Nothing to evaluate.")
        return

    processed_keys = load_processed_keys()
    pending_keys   = [k for k in all_keys if k not in processed_keys]

    log.info(f"Total={len(all_keys)} Pending={len(pending_keys)}")

    if not pending_keys:
        log.info("All results already processed.")
        return

    total_alerts = 0
    errors       = 0

    for key in pending_keys:
        try:
            obj   = s3.get_object(Bucket=S3_RAW_BUCKET, Key=key)
            event = json.loads(obj["Body"].read())

            alerts = evaluate_rules(event)
            for alert in alerts:
                index_alert(es, alert)
                total_alerts += 1

            processed_keys.add(key)

        except Exception as e:
            log.error(f"Failed to process {key}: {e}")
            errors += 1

    save_processed_keys(processed_keys)

    log.info(
        f"Alert engine done — "
        f"processed={len(pending_keys) - errors} "
        f"alerts_fired={total_alerts} "
        f"errors={errors}"
    )


if __name__ == "__main__":
    main()