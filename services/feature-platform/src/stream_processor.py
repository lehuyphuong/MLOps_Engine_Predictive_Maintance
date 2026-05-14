"""
stream_processor.py — Feature Platform Stream Processor

Architecture note — why flink run instead of the Python API:
  PyFlink 1.18.1's Python StreamExecutionEnvironment has no
  create_remote_environment() method (Java-only) and
  get_execution_environment(Configuration) always returns a local
  environment regardless of execution.target in the Configuration —
  it calls JStreamExecutionEnvironment.getExecutionEnvironment() which
  reads the execution context, not the REST address.

  The correct way to submit a PyFlink job to a remote standalone cluster
  is the Flink CLI:
    flink run -m <host>:<port> -py <script.py>
  This is what pyflink_gateway_server.py does internally when
  cluster_type=remote. It invokes $FLINK_HOME/bin/flink run, which
  connects to the JobManager REST API and submits the job graph.

  The submitter pod (this CronJob) runs the Flink CLI as a subprocess.
  The actual operator execution (FeatureMapFunction) runs on the
  TaskManager pod. The job appears in the Flink Web UI.

Two-file design:
  stream_processor.py  (this file) — CronJob entry point.
    1. Reads GCS, loads a capped batch of validated events.
    2. Writes them to a temp JSON file.
    3. Invokes flink_job.py via `flink run -py` for remote submission.
    4. Updates the GCS checkpoint.

  flink_job.py  — the actual PyFlink job script submitted to Flink.
    Reads the temp JSON file, builds the DataStream pipeline,
    runs FeatureMapFunction (writes Redis + PostgreSQL), exits.
    This file runs on the TaskManager side.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage
from google.cloud.exceptions import NotFound
from pyflink.find_flink_home import _find_flink_home

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GCS_RAW_BUCKET        = os.environ.get("GCS_RAW_BUCKET", "phm-raw-data-aide2-494008")
DATASET               = os.environ.get("DATASET",         "FD002")
GCP_PROJECT           = os.environ.get("GCP_PROJECT",     "aide2-494008")
FLINK_JOBMANAGER_HOST = os.environ.get("FLINK_JOBMANAGER_HOST", "flink-jobmanager")
FLINK_JOBMANAGER_PORT = os.environ.get("FLINK_JOBMANAGER_PORT", "8081")
BATCH_SIZE            = int(os.environ.get("STREAM_BATCH_SIZE", "50"))

VALID_PREFIX   = f"validated_engine_cycles/{DATASET}/"
CHECKPOINT_KEY = f"checkpoints/{DATASET}/feature_processed_keys.json"

# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------
_gcs_client = None


def get_gcs():
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client(project=GCP_PROJECT)
    return _gcs_client


def gcs_list_validated_keys() -> list[str]:
    blobs = get_gcs().bucket(GCS_RAW_BUCKET).list_blobs(prefix=VALID_PREFIX)
    return [
        b.name for b in blobs
        if b.name.endswith(".json") and "checkpoint" not in b.name
    ]


def gcs_read_event(key: str) -> dict:
    return json.loads(
        get_gcs().bucket(GCS_RAW_BUCKET).blob(key).download_as_text()
    )


def load_processed_keys() -> set:
    try:
        blob = get_gcs().bucket(GCS_RAW_BUCKET).blob(CHECKPOINT_KEY)
        return set(json.loads(blob.download_as_text()).get("processed", []))
    except NotFound:
        return set()


def save_processed_keys(keys: set) -> None:
    get_gcs().bucket(GCS_RAW_BUCKET).blob(CHECKPOINT_KEY).upload_from_string(
        json.dumps({
            "processed":  list(keys),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }),
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# Main — CronJob entry point
# ---------------------------------------------------------------------------
def main():
    log.info(
        "Stream processor starting — dataset=%s batch=%d "
        "jobmanager=%s:%s",
        DATASET, BATCH_SIZE,
        FLINK_JOBMANAGER_HOST, FLINK_JOBMANAGER_PORT,
    )

    # Collect pending keys
    all_keys       = gcs_list_validated_keys()
    processed_keys = load_processed_keys()
    pending_keys   = [k for k in all_keys if k not in processed_keys]

    log.info("Total=%d Pending=%d Batch=%d",
             len(all_keys), len(pending_keys), BATCH_SIZE)

    if not pending_keys:
        log.info("All validated events already processed.")
        return

    batch_keys = pending_keys[:BATCH_SIZE]
    remaining  = len(pending_keys) - len(batch_keys)
    log.info("Processing batch of %d keys (%d remaining for next run)",
             len(batch_keys), remaining)

    # Load batch from GCS
    records     = []
    load_errors = 0
    for key in batch_keys:
        try:
            event = gcs_read_event(key)
            records.append({"event": event, "key": key})
        except Exception as e:
            log.error("Failed to load %s: %s", key, e)
            load_errors += 1

    log.info("Loaded %d events (%d load errors)", len(records), load_errors)

    if not records:
        log.info("No events loaded. Exiting.")
        return

    # Write batch to a temp file for flink_job.py to read
    # The temp file is written to /tmp which is on the same pod's filesystem.
    # flink_job.py is submitted via `flink run -py` and runs in the same pod
    # process space (the TaskManager executes Python operators via py4j).
    batch_file = "/tmp/stream_batch.json"
    with open(batch_file, "w") as f:
        json.dump(records, f)
    log.info("Batch written to %s (%d records)", batch_file, len(records))

    # Submit job to Flink via CLI
    flink_home  = _find_flink_home()
    flink_bin   = os.path.join(flink_home, "bin", "flink")
    job_script  = str(Path(__file__).parent / "flink_job.py")
    jobmanager  = f"{FLINK_JOBMANAGER_HOST}:{FLINK_JOBMANAGER_PORT}"

    cmd = [
        flink_bin, "run",
        "-m",  jobmanager,        # JobManager REST address
        "-py", job_script,        # Python script to submit
        # Pass batch file path and env vars as program arguments
        "--input",   batch_file,
    ]

    log.info("Submitting to Flink: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    log.info("flink run stdout:\n%s", result.stdout)

    if result.returncode != 0:
        log.error("flink run failed (exit %d):\n%s", result.returncode, result.stderr)
        sys.exit(result.returncode)

    log.info("Flink job submitted successfully")

    # Checkpoint
    for r in records:
        processed_keys.add(r["key"])
    save_processed_keys(processed_keys)

    log.info(
        "Stream processor done — processed=%d load_errors=%d "
        "remaining_pending=%d",
        len(records) - load_errors, load_errors, remaining,
    )


if __name__ == "__main__":
    main()