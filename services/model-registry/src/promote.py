"""
promote.py — Model Registry: Validation + Promotion
Runs as a second container in the training Job (sidecar pattern).
Waits for the training container to finish, then validates and promotes.

Flow:
  1. Poll MLflow until a new run appears in both experiments
  2. Fetch metrics from latest RUL run (mean_rmse, mean_mae, mean_s_score)
  3. RMSE gate  — reject if mean_rmse > RMSE_THRESHOLD
  4. Drift check — reject if mean_rmse degraded > DRIFT_TOLERANCE vs current production
  5. Register both models (RUL + anomaly) in MLflow Model Registry
  6. Transition RUL model: None → Staging → Production
  7. Transition anomaly model: None → Staging → Production
  8. Write promotion manifest to S3 — model-serving reads this to load artifacts

Environment variables:
  MLFLOW_TRACKING_URI   postgresql+psycopg2://...
  S3_ARTIFACTS_BUCKET   phm-model-artifacts
  S3_MLFLOW_BUCKET      phm-mlflow-artifacts
  DATASET               FD002
  RMSE_THRESHOLD        0.30
  DRIFT_TOLERANCE       0.05  (5% degradation allowed vs current production)
  POLL_INTERVAL         30    (seconds between MLflow polls)
  POLL_TIMEOUT          3600  (max seconds to wait for training to finish)
  AWS_REGION            ap-southeast-1
"""

import io
import os
import json
import time
import logging
import boto3
from datetime import datetime, timezone
from urllib.parse import quote_plus
import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET           = os.environ.get("DATASET",            "FD002")
AWS_REGION        = os.environ.get("AWS_REGION",         "ap-southeast-1")
S3_ARTIFACTS_BUCKET = os.environ.get("S3_ARTIFACTS_BUCKET", "phm-model-artifacts")
RMSE_THRESHOLD    = float(os.environ.get("RMSE_THRESHOLD",  "0.30"))
DRIFT_TOLERANCE   = float(os.environ.get("DRIFT_TOLERANCE", "0.05"))
POLL_INTERVAL     = int(os.environ.get("POLL_INTERVAL",     "30"))
POLL_TIMEOUT      = int(os.environ.get("POLL_TIMEOUT",      "3600"))

_db_host_raw = os.environ.get("DB_HOST", "localhost")
DB_HOST      = _db_host_raw.split(":")[0]
DB_PORT      = int(_db_host_raw.split(":")[1]) if ":" in _db_host_raw \
               else int(os.environ.get("DB_PORT", "5432"))
DB_NAME      = os.environ.get("DB_NAME",     "phmdb")
DB_USER      = os.environ.get("DB_USER",     "phmadmin")
DB_PASSWORD  = os.environ.get("DB_PASSWORD", "")

MLFLOW_TRACKING_URI = (
    f"postgresql+psycopg2://{DB_USER}:{quote_plus(DB_PASSWORD)}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# MLflow experiment and model names
RUL_EXPERIMENT     = f"phm-rul-xgboost-{DATASET}-offline"
ANOMALY_EXPERIMENT = f"phm-anomaly-autoencoder-{DATASET}"
RUL_MODEL_NAME     = f"phm-rul-xgboost-{DATASET}"
ANOMALY_MODEL_NAME = f"phm-anomaly-autoencoder-{DATASET}"

s3     = boto3.client("s3", region_name=AWS_REGION)
client = None   # initialized in main after tracking URI is set


# ---------------------------------------------------------------------------
# Poll MLflow for latest completed run
# ---------------------------------------------------------------------------
def get_latest_run(experiment_name: str) -> dict | None:
    """Return the most recent FINISHED run for an experiment, or None."""
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        return runs[0] if runs else None
    except Exception as e:
        log.warning(f"Error fetching runs for {experiment_name}: {e}")
        return None


def wait_for_runs() -> tuple[object, object]:
    """
    Poll until both RUL and anomaly experiments have a finished run.
    Returns (rul_run, anomaly_run).
    """
    log.info(f"Waiting for training runs to complete "
             f"(timeout={POLL_TIMEOUT}s, interval={POLL_INTERVAL}s)...")

    deadline = time.time() + POLL_TIMEOUT

    while time.time() < deadline:
        rul_run     = get_latest_run(RUL_EXPERIMENT)
        anomaly_run = get_latest_run(ANOMALY_EXPERIMENT)

        if rul_run and anomaly_run:
            log.info(f"Training runs found:")
            log.info(f"  RUL run_id     : {rul_run.info.run_id}")
            log.info(f"  Anomaly run_id : {anomaly_run.info.run_id}")
            return rul_run, anomaly_run

        missing = []
        if not rul_run:     missing.append(RUL_EXPERIMENT)
        if not anomaly_run: missing.append(ANOMALY_EXPERIMENT)
        log.info(f"Still waiting for: {missing} — polling again in {POLL_INTERVAL}s")
        time.sleep(POLL_INTERVAL)

    raise TimeoutError(
        f"Training did not complete within {POLL_TIMEOUT}s. "
        "Check model-training job logs."
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def rmse_gate(rul_run) -> tuple[bool, str]:
    """Check if mean_rmse is below threshold."""
    mean_rmse = rul_run.data.metrics.get("mean_rmse")
    if mean_rmse is None:
        return False, "mean_rmse metric not found in run"

    if mean_rmse > RMSE_THRESHOLD:
        return False, (
            f"RMSE gate FAILED — mean_rmse={mean_rmse:.4f} "
            f"> threshold={RMSE_THRESHOLD}"
        )

    log.info(f"RMSE gate PASSED — mean_rmse={mean_rmse:.4f} <= {RMSE_THRESHOLD}")
    return True, f"mean_rmse={mean_rmse:.4f}"


def drift_check(rul_run) -> tuple[bool, str]:
    """
    Compare new model RMSE against currently registered production model.
    If no production model exists yet, skip drift check and allow promotion.
    """
    try:
        prod_versions = client.get_latest_versions(
            RUL_MODEL_NAME, stages=["Production"]
        )
    except Exception:
        # Model not registered yet — first promotion, skip drift check
        log.info("Drift check skipped — no production model registered yet")
        return True, "first_promotion"

    if not prod_versions:
        log.info("Drift check skipped — no production model registered yet")
        return True, "first_promotion"

    # Get RMSE of current production model
    prod_run_id = prod_versions[0].run_id
    prod_run    = client.get_run(prod_run_id)
    prod_rmse   = prod_run.data.metrics.get("mean_rmse", float("inf"))
    new_rmse    = rul_run.data.metrics.get("mean_rmse", float("inf"))

    degradation = new_rmse - prod_rmse

    if degradation > DRIFT_TOLERANCE:
        return False, (
            f"Drift check FAILED — new_rmse={new_rmse:.4f} "
            f"prod_rmse={prod_rmse:.4f} "
            f"degradation={degradation:.4f} > tolerance={DRIFT_TOLERANCE}"
        )

    log.info(f"Drift check PASSED — new={new_rmse:.4f} "
             f"prod={prod_rmse:.4f} delta={degradation:+.4f}")
    return True, f"delta={degradation:+.4f}"


# ---------------------------------------------------------------------------
# Registration and promotion
# ---------------------------------------------------------------------------
def register_and_promote(run, model_name: str, artifact_path: str) -> str:
    """
    Register a run's artifact in MLflow Model Registry and
    transition it to Production via Staging.
    Returns the new model version string.
    """
    model_uri = f"runs:/{run.info.run_id}/{artifact_path}"

    # Register — creates version 1 (or next version if already exists)
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = mv.version
    log.info(f"Registered {model_name} version {version}")

    # Transition: None → Staging
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Staging",
        archive_existing_versions=False,
    )
    log.info(f"{model_name} v{version} → Staging")

    # Transition: Staging → Production
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production",
        archive_existing_versions=True,   # archive previous production
    )
    log.info(f"{model_name} v{version} → Production")

    return version


# ---------------------------------------------------------------------------
# Write promotion manifest to S3 (model-serving reads this)
# ---------------------------------------------------------------------------
def write_promotion_manifest(
    rul_run,
    anomaly_run,
    rul_version: str,
    anomaly_version: str,
    validation_notes: dict,
) -> str:
    manifest = {
        "promoted_at":       datetime.now(timezone.utc).isoformat(),
        "dataset":           DATASET,
        "rul_model": {
            "name":          RUL_MODEL_NAME,
            "version":       rul_version,
            "run_id":        rul_run.info.run_id,
            "s3_uri":        rul_run.data.params.get("model_s3_uri", ""),
            "mean_rmse":     rul_run.data.metrics.get("mean_rmse"),
            "mean_mae":      rul_run.data.metrics.get("mean_mae"),
            "mean_s_score":  rul_run.data.metrics.get("mean_s_score"),
        },
        "anomaly_model": {
            "name":          ANOMALY_MODEL_NAME,
            "version":       anomaly_version,
            "run_id":        anomaly_run.info.run_id,
            "s3_uri":        anomaly_run.data.params.get("artifacts_s3_uri", ""),
            "threshold":     anomaly_run.data.metrics.get("threshold"),
            "detection_f1":  anomaly_run.data.metrics.get("detection_f1"),
        },
        "validation": validation_notes,
    }

    key = f"registry/{DATASET}/promotion_manifest.json"
    s3.put_object(
        Bucket=S3_ARTIFACTS_BUCKET,
        Key=key,
        Body=json.dumps(manifest, indent=2),
        ContentType="application/json",
    )

    s3_uri = f"s3://{S3_ARTIFACTS_BUCKET}/{key}"
    log.info(f"Promotion manifest written to {s3_uri}")
    return s3_uri


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global client

    log.info(f"Model registry starting — dataset={DATASET} "
             f"rmse_threshold={RMSE_THRESHOLD} drift_tolerance={DRIFT_TOLERANCE}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Wait for both training runs to finish
    rul_run, anomaly_run = wait_for_runs()

    # RMSE gate
    rmse_ok, rmse_note = rmse_gate(rul_run)
    if not rmse_ok:
        log.error(f"PROMOTION REJECTED — {rmse_note}")
        return

    # Drift check
    drift_ok, drift_note = drift_check(rul_run)
    if not drift_ok:
        log.error(f"PROMOTION REJECTED — {drift_note}")
        return

    log.info("Validation PASSED — proceeding with promotion")

    # Register and promote both models
    rul_version     = register_and_promote(rul_run,     RUL_MODEL_NAME,     "xgboost_model")
    anomaly_version = register_and_promote(anomaly_run, ANOMALY_MODEL_NAME, "anomaly_model")

    # Write promotion manifest for model-serving
    manifest_uri = write_promotion_manifest(
        rul_run, anomaly_run,
        rul_version, anomaly_version,
        validation_notes={
            "rmse_gate":   rmse_note,
            "drift_check": drift_note,
        },
    )

    log.info("=" * 60)
    log.info("PROMOTION COMPLETE")
    log.info(f"  RUL model     : {RUL_MODEL_NAME} v{rul_version} → Production")
    log.info(f"  Anomaly model : {ANOMALY_MODEL_NAME} v{anomaly_version} → Production")
    log.info(f"  Manifest      : {manifest_uri}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()