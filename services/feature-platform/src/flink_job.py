"""
flink_job.py — PyFlink job submitted to the remote Flink cluster.

Invoked by stream_processor.py via:
  flink run -m <jobmanager>:<port> -py flink_job.py --input /tmp/stream_batch.json

This script is the actual Flink job. It runs in the context of the Flink
CLI which sets execution.target=remote automatically when -m is provided.
get_execution_environment() in this context returns the remote environment
connected to the JobManager, and the job appears in the Flink Web UI.

The FeatureMapFunction writes to Redis (online store) and PostgreSQL
(offline store). All env vars are inherited from the CronJob pod's
environment, which gets them from the feature-platform-config ConfigMap.
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import psycopg2
import redis as redis_lib

from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment, RuntimeExecutionMode
from pyflink.datastream.functions import MapFunction, RuntimeContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — inherited from CronJob pod environment (ConfigMap + Secret)
# ---------------------------------------------------------------------------
REDIS_HOST  = os.environ.get("REDIS_HOST",  "localhost")
REDIS_PORT  = int(os.environ.get("REDIS_PORT", "6379"))
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "24"))

DB_HOST     = os.environ.get("DB_HOST",     "localhost")
DB_PORT     = int(os.environ.get("DB_PORT", "5432"))
DB_NAME     = os.environ.get("DB_NAME",     "phmdb")
DB_USER     = os.environ.get("DB_USER",     "phmadmin")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

DATASET     = os.environ.get("DATASET", "FD002")

COLS_TO_DROP = {
    "OperSet1", "OperSet2", "OperSet3",
    "SensorMes1", "SensorMes5", "SensorMes10",
    "SensorMes18", "SensorMes19",
}

FEATURE_COLS = (
    [f"OperSet{i}"   for i in range(1, 4)  if f"OperSet{i}"   not in COLS_TO_DROP]
    + [f"SensorMes{j}" for j in range(1, 22) if f"SensorMes{j}" not in COLS_TO_DROP]
)


def extract_features(event: dict) -> dict:
    sensors = event.get("sensors", {})
    return {col: sensors[col] for col in FEATURE_COLS if col in sensors}


# ---------------------------------------------------------------------------
# Flink MapFunction
# ---------------------------------------------------------------------------
class FeatureMapFunction(MapFunction):

    def open(self, ctx: RuntimeContext) -> None:
        self._redis = redis_lib.Redis(
            host=REDIS_HOST, port=REDIS_PORT,
            decode_responses=True, socket_connect_timeout=5,
        )
        self._pg = psycopg2.connect(
            host=DB_HOST, port=DB_PORT,
            dbname=DB_NAME, user=DB_USER,
            password=DB_PASSWORD, connect_timeout=10,
        )
        self._pg.autocommit = False
        self._count = 0
        log.info("FeatureMapFunction.open() — connections established")

    def map(self, value: str) -> str:
        record  = json.loads(value)
        event   = record["event"]
        key     = record["key"]
        unit_id = int(event["unit_id"])
        cycle   = int(event["cycle"])
        features = extract_features(event)

        self._sink_redis(unit_id, features, cycle)
        self._sink_postgres(event, features, key)

        self._count += 1
        if self._count % 50 == 0:
            self._pg.commit()
            log.info("Progress — %d records processed", self._count)

        return value

    def close(self) -> None:
        try:
            if self._pg and not self._pg.closed:
                self._pg.commit()
                self._pg.close()
        except Exception as e:
            log.warning("PostgreSQL close error: %s", e)
        log.info("FeatureMapFunction.close() — %d records total", self._count)

    def _sink_redis(self, unit_id: int, features: dict, cycle: int) -> None:
        key    = f"engine:{unit_id}:window"
        raw    = self._redis.get(key)
        window = json.loads(raw) if raw else []

        window.append({"cycle": cycle, "features": features})
        if len(window) > WINDOW_SIZE:
            window = window[-WINDOW_SIZE:]

        self._redis.set(key, json.dumps(window))
        self._redis.set(f"engine:{unit_id}:latest", json.dumps({
            "cycle":      cycle,
            "features":   features,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }))

    def _sink_postgres(self, event: dict, features: dict, source_key: str) -> None:
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
        with self._pg.cursor() as cur:
            cur.execute(sql, {
                "dataset":    DATASET,
                "unit_id":    int(event["unit_id"]),
                "cycle":      int(event["cycle"]),
                "rul":        None,
                "rul_capped": None,
                "features":   json.dumps(features),
                "source_key": source_key,
            })


# ---------------------------------------------------------------------------
# Entry point — called by `flink run -py flink_job.py --input <path>`
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to JSON batch file written by stream_processor.py")
    args = parser.parse_args()

    with open(args.input) as f:
        records = json.load(f)

    log.info("flink_job.py starting — %d records from %s", len(records), args.input)

    # In this context (invoked via `flink run -m host:port -py ...`)
    # get_execution_environment() returns the REMOTE environment
    # connected to the JobManager specified by -m.
    # The job appears in the Flink Web UI automatically.
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_runtime_mode(RuntimeExecutionMode.BATCH)
    env.set_parallelism(1)

    serialised = [json.dumps(r) for r in records]

    (
        env
        .from_collection(serialised, type_info=Types.STRING())
        .map(FeatureMapFunction(), output_type=Types.STRING())
        .name("phm-feature-pipeline")
    )

    env.execute("phm-feature-pipeline")
    log.info("flink_job.py done — %d records processed", len(records))


if __name__ == "__main__":
    main()
