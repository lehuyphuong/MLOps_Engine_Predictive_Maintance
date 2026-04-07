"""
producer.py — Telemetry Producer
Reads simulated_FD002.txt from S3 row by row and publishes each cycle
as a JSON event to S3 (s3://phm-raw-data/raw_engine_cycles/<dataset>/...).

Runs as a Kubernetes CronJob. On each trigger it picks up from the last
recorded cycle (stored in a checkpoint file on S3) so it never re-sends
the same row twice.

Environment variables (set in configmap.yaml):
  S3_BUCKET       raw data bucket       (phm-raw-data)
  S3_DATA_KEY     path to source file   (cmapss-data/simulated_FD002.txt)
  DATASET         sub-dataset label     (FD002)
  CYCLES_PER_RUN  rows per CronJob run  (default: 50)
  AWS_REGION      ap-southeast-1
"""

import io
import os
import json
import logging
import boto3
import pandas as pd
from datetime import datetime, timezone
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

S3_BUCKET      = os.environ.get("S3_BUCKET",      "phm-raw-data")
DATASET        = os.environ.get("DATASET",         "FD002")
CYCLES_PER_RUN = int(os.environ.get("CYCLES_PER_RUN", "50"))
AWS_REGION     = os.environ.get("AWS_REGION",      "ap-southeast-1")
S3_DATA_KEY    = os.environ.get("S3_DATA_KEY",     f"cmapss-data/simulated_{DATASET}.txt")

CHECKPOINT_KEY = f"checkpoints/{DATASET}/last_row.json"

COLUMN_NAMES = (
    ["UnitNumber", "TimeInCycles"]
    + [f"OperSet{i}"   for i in range(1, 4)]
    + [f"SensorMes{j}" for j in range(1, 22)]
)

# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
s3 = boto3.client("s3", region_name=AWS_REGION)


def load_dataframe() -> pd.DataFrame:
    """Read the simulated source file directly from S3."""
    log.info(f"Loading s3://{S3_BUCKET}/{S3_DATA_KEY}")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_DATA_KEY)
    df  = pd.read_csv(
        io.BytesIO(obj["Body"].read()),
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )
    log.info(f"Loaded {len(df)} rows from s3://{S3_BUCKET}/{S3_DATA_KEY}")
    return df


def load_checkpoint() -> int:
    """Return the index of the last row that was successfully emitted.
    Returns -1 if no checkpoint exists (fresh start)."""
    try:
        obj  = s3.get_object(Bucket=S3_BUCKET, Key=CHECKPOINT_KEY)
        data = json.loads(obj["Body"].read())
        last = data.get("last_row_index", -1)
        log.info(f"Checkpoint loaded — last emitted row index: {last}")
        return last
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            log.info("No checkpoint found — starting from row 0")
            return -1
        raise


def save_checkpoint(last_row_index: int) -> None:
    payload = {
        "last_row_index": last_row_index,
        "dataset":        DATASET,
        "updated_at":     datetime.now(timezone.utc).isoformat(),
    }
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=CHECKPOINT_KEY,
        Body=json.dumps(payload),
        ContentType="application/json",
    )
    log.info(f"Checkpoint saved — last row index: {last_row_index}")


def emit_cycle(row: dict, row_index: int) -> None:
    """Write a single cycle event as JSON to S3."""
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
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(event),
        ContentType="application/json",
    )
    log.debug(f"Emitted unit={unit} cycle={cycle} -> s3://{S3_BUCKET}/{key}")


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