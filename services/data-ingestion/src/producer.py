"""
producer.py — Telemetry Producer
Reads simulated_FD002.txt from GCS row by row and publishes each cycle
as a JSON event to GCS (gs://phm-raw-data-.../raw_engine_cycles/<dataset>/...).

Runs as a Kubernetes CronJob. On each trigger it picks up from the last
recorded cycle (stored in a checkpoint file on GCS) so it never re-sends
the same row twice.

Environment variables (set in configmap.yaml):
  GCS_BUCKET      raw data bucket       (phm-raw-data-aide2-494008)
  GCS_DATA_KEY    path to source file   (cmapss-data/simulated_FD002.txt)
  DATASET         sub-dataset label     (FD002)
  CYCLES_PER_RUN  rows per CronJob run  (default: 50)
  GCP_PROJECT     aide2-494008
"""

import io
import os
import json
import logging
import pandas as pd
from datetime import datetime, timezone
from google.cloud import storage
from google.cloud.exceptions import NotFound

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

GCS_BUCKET     = os.environ.get("GCS_BUCKET",     "phm-raw-data-aide2-494008")
DATASET        = os.environ.get("DATASET",         "FD002")
CYCLES_PER_RUN = int(os.environ.get("CYCLES_PER_RUN", "50"))
GCP_PROJECT    = os.environ.get("GCP_PROJECT",     "aide2-494008")
GCS_DATA_KEY   = os.environ.get("GCS_DATA_KEY",    f"cmapss-data/simulated_{DATASET}.txt")

CHECKPOINT_KEY = f"checkpoints/{DATASET}/last_row.json"

COLUMN_NAMES = (
    ["UnitNumber", "TimeInCycles"]
    + [f"OperSet{i}"   for i in range(1, 4)]
    + [f"SensorMes{j}" for j in range(1, 22)]
)

# ---------------------------------------------------------------------------
# GCS client
# ---------------------------------------------------------------------------
gcs = storage.Client(project=GCP_PROJECT)


def load_dataframe() -> pd.DataFrame:
    """Read the simulated source file directly from GCS."""
    log.info(f"Loading gs://{GCS_BUCKET}/{GCS_DATA_KEY}")
    blob = gcs.bucket(GCS_BUCKET).blob(GCS_DATA_KEY)
    data = blob.download_as_bytes()
    df   = pd.read_csv(
        io.BytesIO(data),
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )
    log.info(f"Loaded {len(df)} rows from gs://{GCS_BUCKET}/{GCS_DATA_KEY}")
    return df


def load_checkpoint() -> int:
    """Return the index of the last row that was successfully emitted.
    Returns -1 if no checkpoint exists (fresh start)."""
    try:
        blob = gcs.bucket(GCS_BUCKET).blob(CHECKPOINT_KEY)
        data = json.loads(blob.download_as_text())
        last = data.get("last_row_index", -1)
        log.info(f"Checkpoint loaded — last emitted row index: {last}")
        return last
    except NotFound:
        log.info("No checkpoint found — starting from row 0")
        return -1


def save_checkpoint(last_row_index: int) -> None:
    payload = {
        "last_row_index": last_row_index,
        "dataset":        DATASET,
        "updated_at":     datetime.now(timezone.utc).isoformat(),
    }
    blob = gcs.bucket(GCS_BUCKET).blob(CHECKPOINT_KEY)
    blob.upload_from_string(
        json.dumps(payload),
        content_type="application/json",
    )
    log.info(f"Checkpoint saved — last row index: {last_row_index}")


def emit_cycle(row: dict, row_index: int) -> None:
    """Write a single cycle event as JSON to GCS."""
    unit  = row["UnitNumber"]
    cycle = row["TimeInCycles"]
    ts    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")

    event = {
        "event_type": "engine_cycle",
        "dataset":    DATASET,
        "unit_id":    int(unit),
        "cycle":      int(cycle),
        "row_index":  row_index,
        "emitted_at": datetime.now(timezone.utc).isoformat(),
        "sensors": {
            k: float(v) for k, v in row.items()
            if k not in ("UnitNumber", "TimeInCycles")
        },
    }

    unit = int(unit)
    key  = f"raw_engine_cycles/{DATASET}/unit_{unit:03d}/{ts}_{row_index}.json"

    blob = gcs.bucket(GCS_BUCKET).blob(key)
    blob.upload_from_string(
        json.dumps(event),
        content_type="application/json",
    )
    log.debug(f"Emitted unit={unit} cycle={cycle} -> gs://{GCS_BUCKET}/{key}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info(f"Producer starting — dataset={DATASET} cycles_per_run={CYCLES_PER_RUN}")

    df         = load_dataframe()
    total_rows = len(df)

    last_index  = load_checkpoint()
    start_index = last_index + 1

    if start_index >= total_rows:
        log.info("All rows have already been emitted. Nothing to do.")
        return

    end_index = min(start_index + CYCLES_PER_RUN, total_rows)
    batch     = df.iloc[start_index:end_index]

    emitted = 0
    for abs_index, (_, row) in enumerate(batch.iterrows(), start=start_index):
        emit_cycle(row.to_dict(), abs_index)
        emitted += 1

    save_checkpoint(end_index - 1)

    remaining = total_rows - end_index
    log.info(
        f"Done — emitted {emitted} cycles "
        f"(rows {start_index}..{end_index - 1}). "
        f"{remaining} rows remaining."
    )


if __name__ == "__main__":
    main()
