"""
validation_service.py — Validation Service
Reads raw cycle events from S3 (raw_engine_cycles/), validates each batch
using Great Expectations Core, then routes to:
  valid   -> s3://phm-raw-data/validated_engine_cycles/
  invalid -> s3://phm-data-quality/invalid_engine_cycles/

Four expectation suites matching the diagram:
  1. schema  — required fields present, correct types
  2. null    — no missing sensor values
  3. range   — sensor readings within C-MAPSS FD002 observed bounds
  4. order   — cycle >= 1, unit_id >= 1

Environment variables:
  S3_RAW_BUCKET      phm-raw-data
  S3_QUALITY_BUCKET  phm-data-quality
  DATASET            FD002
  AWS_REGION         ap-southeast-1
"""

import os
import json
import logging
import boto3
import pandas as pd
import great_expectations as gx
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
from datetime import datetime, timezone
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

S3_RAW_BUCKET     = os.environ.get("S3_RAW_BUCKET",     "phm-raw-data")
S3_QUALITY_BUCKET = os.environ.get("S3_QUALITY_BUCKET", "phm-data-quality")
DATASET           = os.environ.get("DATASET",            "FD002")
AWS_REGION        = os.environ.get("AWS_REGION",         "ap-southeast-1")

RAW_PREFIX     = f"raw_engine_cycles/{DATASET}/"
VALID_PREFIX   = f"validated_engine_cycles/{DATASET}/"
INVALID_PREFIX = f"invalid_engine_cycles/{DATASET}/"
CHECKPOINT_KEY = f"checkpoints/{DATASET}/validated_keys.json"

s3 = boto3.client("s3", region_name=AWS_REGION)

SENSOR_COLS     = [f"SensorMes{j}" for j in range(1, 22)]
OPSET_COLS      = [f"OperSet{i}"   for i in range(1, 4)]
ALL_SENSOR_KEYS = OPSET_COLS + SENSOR_COLS

# C-MAPSS FD002 observed sensor bounds
SENSOR_RANGES = {
    "OperSet1":    (0.0,    100.0),  "OperSet2":    (0.0,    1.0),
    "OperSet3":    (0.0,    100.0),  "SensorMes1":  (400.0,  550.0),
    "SensorMes2":  (530.0,  650.0),  "SensorMes3":  (1000.0, 1700.0),
    "SensorMes4":  (1000.0, 1500.0), "SensorMes5":  (0.0,    20.0),
    "SensorMes6":  (0.0,    30.0),   "SensorMes7":  (100.0,  400.0),
    "SensorMes8":  (1800.0, 2400.0), "SensorMes9":  (7000.0, 9500.0),
    "SensorMes10": (0.5,    1.5),    "SensorMes11": (30.0,   50.0),
    "SensorMes12": (100.0,  400.0),  "SensorMes13": (2000.0, 2600.0),
    "SensorMes14": (7000.0, 9500.0), "SensorMes15": (8.0,    13.0),
    "SensorMes16": (0.0,    1.0),    "SensorMes17": (200.0,  500.0),
    "SensorMes18": (1800.0, 2400.0), "SensorMes19": (50.0,   110.0),
    "SensorMes20": (10.0,   25.0),   "SensorMes21": (5.0,    15.0),
}


def build_expectation_suite() -> ExpectationSuite:
    """
    Builds four suites as per architecture diagram:
      schema  -> required columns exist
      null    -> no null sensor values
      range   -> sensor values within C-MAPSS FD002 bounds
      order   -> unit_id >= 1, cycle >= 1
    """
    suite = ExpectationSuite(expectation_suite_name="phm_cycle_validation")

    # 1. Schema
    for col in ["unit_id", "cycle"] + ALL_SENSOR_KEYS:
        suite.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_to_exist",
            kwargs={"column": col},
            meta={"suite": "schema"},
        ))

    # 2. Null
    for col in ALL_SENSOR_KEYS:
        suite.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": col},
            meta={"suite": "null"},
        ))

    # 3. Range
    for col, (lo, hi) in SENSOR_RANGES.items():
        suite.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": col, "min_value": lo, "max_value": hi},
            meta={"suite": "range"},
        ))

    # 4. Order
    for col in ["unit_id", "cycle"]:
        suite.add_expectation(ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": col, "min_value": 1},
            meta={"suite": "order"},
        ))

    return suite


def event_to_dataframe(event: dict) -> pd.DataFrame:
    """Flatten nested sensors dict into a single-row DataFrame for GX."""
    sensors = event.get("sensors", {})
    row = {"unit_id": event.get("unit_id"), "cycle": event.get("cycle"), **sensors}
    return pd.DataFrame([row])


def validate_event(event: dict, suite: ExpectationSuite, context) -> tuple[bool, list[str]]:
    """Run all four GX suites against one event. Returns (is_valid, failures)."""
    df         = event_to_dataframe(event)
    datasource = context.sources.add_or_update_pandas(name="cycle_event")
    asset      = datasource.add_dataframe_asset(name="event_row")
    batch_req  = asset.build_batch_request(dataframe=df)
    validator  = context.get_validator(batch_request=batch_req, expectation_suite=suite)
    results    = validator.validate()

    failures = []
    for result in results.results:
        if not result.success:
            suite_name  = result.expectation_config.meta.get("suite", "")
            expectation = result.expectation_config.expectation_type
            col         = result.expectation_config.kwargs.get("column", "")
            failures.append(f"{suite_name}:{expectation}:{col}")

    return results.success, failures


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


def route_event(event: dict, source_key: str, is_valid: bool, failures: list) -> None:
    filename = source_key.split("/")[-1]
    unit     = event.get("unit_id", "unknown")
    ts       = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")

    if is_valid:
        dest_key = f"{VALID_PREFIX}unit_{unit:03d}/{filename}"
        s3.put_object(
            Bucket=S3_RAW_BUCKET,
            Key=dest_key,
            Body=json.dumps(event),
            ContentType="application/json",
        )
        log.debug(f"VALID   -> s3://{S3_RAW_BUCKET}/{dest_key}")
    else:
        bad_event = {
            **event,
            "validation_failures": failures,
            "flagged_at":          datetime.now(timezone.utc).isoformat(),
            "source_key":          source_key,
        }
        dest_key = f"{INVALID_PREFIX}unit_{unit}/{ts}_{filename}"
        s3.put_object(
            Bucket=S3_QUALITY_BUCKET,
            Key=dest_key,
            Body=json.dumps(bad_event),
            ContentType="application/json",
        )
        log.warning(f"INVALID -> s3://{S3_QUALITY_BUCKET}/{dest_key} failures={failures}")


def main():
    log.info(f"Validation service starting — dataset={DATASET}")

    context = gx.get_context(mode="ephemeral")
    suite   = build_expectation_suite()
    log.info("GX suite ready — schema / null / range / order")

    paginator = s3.get_paginator("list_objects_v2")
    all_keys  = [
        obj["Key"]
        for page in paginator.paginate(Bucket=S3_RAW_BUCKET, Prefix=RAW_PREFIX)
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".json") and "checkpoint" not in obj["Key"]
    ]

    if not all_keys:
        log.info("No raw events found. Nothing to validate.")
        return

    processed_keys = load_processed_keys()
    pending_keys   = [k for k in all_keys if k not in processed_keys]

    log.info(f"Total={len(all_keys)} Pending={len(pending_keys)}")
    if not pending_keys:
        log.info("All events already validated.")
        return

    valid_count = invalid_count = 0

    for key in pending_keys:
        try:
            obj      = s3.get_object(Bucket=S3_RAW_BUCKET, Key=key)
            event    = json.loads(obj["Body"].read())
            is_valid, failures = validate_event(event, suite, context)
            route_event(event, key, is_valid, failures)
            valid_count   += is_valid
            invalid_count += not is_valid
            processed_keys.add(key)
        except Exception as e:
            log.error(f"Failed to process {key}: {e}")

    save_processed_keys(processed_keys)
    log.info(
        f"Done — valid={valid_count} invalid={invalid_count} "
        f"total={valid_count + invalid_count}"
    )


if __name__ == "__main__":
    main()