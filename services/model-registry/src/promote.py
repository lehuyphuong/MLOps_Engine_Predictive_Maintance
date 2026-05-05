"""
promote.py — Model Registry: Validation + Promotion

Flow:
  1. Poll MLflow until a new run appears in both experiments
  2. Fetch metrics from latest RUL run (mean_rmse, mean_mae, mean_s_score)
  3. RMSE gate  — reject if mean_rmse > RMSE_THRESHOLD
  4. Drift check — reject if mean_rmse degraded > DRIFT_TOLERANCE vs production
  5. Register both models in MLflow Model Registry
  6. Transition: None => Staging => Production
  7. Write promotion manifest to GCS — model-serving reads this to load artifacts

Environment variables:
  MLFLOW_TRACKING_URI      postgresql+psycopg2://...
  GCS_ARTIFACTS_BUCKET     phm-model-artifacts-aide2-494008
  GCP_PROJECT              aide2-494008
  DATASET                  FD002
  RMSE_THRESHOLD           0.30
  DRIFT_TOLERANCE          0.05
  POLL_INTERVAL            30
  POLL_TIMEOUT             3600
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from urllib.parse import quote_plus

import mlflow
from mlflow.tracking import MlflowClient
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET              = os.environ.get("DATASET",              "FD002")
GCP_PROJECT          = os.environ.get("GCP_PROJECT",           "aide2-494008")
GCS_ARTIFACTS_BUCKET = os.environ.get("GCS_ARTIFACTS_BUCKET",  "phm-model-artifacts-aide2-494008")
RMSE_THRESHOLD       = float(os.environ.get("RMSE_THRESHOLD",  "0.30"))
DRIFT_TOLERANCE      = float(os.environ.get("DRIFT_TOLERANCE", "0.05"))
POLL_INTERVAL        = int(os.environ.get("POLL_INTERVAL",     "30"))
POLL_TIMEOUT         = int(os.environ.get("POLL_TIMEOUT",      "3600"))

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

RUL_EXPERIMENT     = f"phm-rul-xgboost-{DATASET}-offline"
ANOMALY_EXPERIMENT = f"phm-anomaly-autoencoder-{DATASET}"
RUL_MODEL_NAME     = f"phm-rul-xgboost-{DATASET}"
ANOMALY_MODEL_NAME = f"phm-anomaly-autoencoder-{DATASET}"

# ---------------------------------------------------------------------------
# GCS client  (replaces boto3.client("s3"))
# ---------------------------------------------------------------------------
gcs    = storage.Client(project=GCP_PROJECT)
client = None   # MLflow client — initialised in main


def gcs_write_json(bucket: str, key: str, data: dict) -> str:
    gcs.bucket(bucket).blob(key).upload_from_string(
        json.dumps(data, indent=2),
        content_type="application/json",
    )
    return f"gs://{bucket}/{key}"


# ---------------------------------------------------------------------------
# Poll MLflow  — unchanged logic
# ---------------------------------------------------------------------------
def get_latest_run(experiment_name: str):
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


def wait_for_runs() -> tuple:
    log.info(f"Waiting for training runs "
             f"(timeout={POLL_TIMEOUT}s interval={POLL_INTERVAL}s)...")
    deadline = time.time() + POLL_TIMEOUT

    while time.time() < deadline:
        rul_run     = get_latest_run(RUL_EXPERIMENT)
        anomaly_run = get_latest_run(ANOMALY_EXPERIMENT)

        if rul_run and anomaly_run:
            log.info(f"Runs found — RUL={rul_run.info.run_id} "
                     f"Anomaly={anomaly_run.info.run_id}")
            return rul_run, anomaly_run

        missing = ([RUL_EXPERIMENT]     if not rul_run     else []) + \
                  ([ANOMALY_EXPERIMENT] if not anomaly_run else [])
        log.info(f"Still waiting for: {missing} — retrying in {POLL_INTERVAL}s")
        time.sleep(POLL_INTERVAL)

    raise TimeoutError(
        f"Training did not complete within {POLL_TIMEOUT}s. "
        "Check model-training job logs."
    )


# ---------------------------------------------------------------------------
# Validation — unchanged logic
# ---------------------------------------------------------------------------
def rmse_gate(rul_run) -> tuple[bool, str]:
    mean_rmse = rul_run.data.metrics.get("mean_rmse")
    if mean_rmse is None:
        return False, "mean_rmse metric not found in run"
    if mean_rmse > RMSE_THRESHOLD:
        return False, (f"RMSE gate FAILED — mean_rmse={mean_rmse:.4f} "
                       f"> threshold={RMSE_THRESHOLD}")
    log.info(f"RMSE gate PASSED — mean_rmse={mean_rmse:.4f}")
    return True, f"mean_rmse={mean_rmse:.4f}"


def drift_check(rul_run) -> tuple[bool, str]:
    try:
        prod_versions = client.get_latest_versions(
            RUL_MODEL_NAME, stages=["Production"]
        )
    except Exception:
        log.info("Drift check skipped — no production model yet")
        return True, "first_promotion"

    if not prod_versions:
        log.info("Drift check skipped — no production model yet")
        return True, "first_promotion"

    prod_rmse   = client.get_run(prod_versions[0].run_id) \
                        .data.metrics.get("mean_rmse", float("inf"))
    new_rmse    = rul_run.data.metrics.get("mean_rmse", float("inf"))
    degradation = new_rmse - prod_rmse

    if degradation > DRIFT_TOLERANCE:
        return False, (f"Drift FAILED — new={new_rmse:.4f} "
                       f"prod={prod_rmse:.4f} delta={degradation:.4f}")

    log.info(f"Drift check PASSED — delta={degradation:+.4f}")
    return True, f"delta={degradation:+.4f}"


# ---------------------------------------------------------------------------
# Registration and promotion — unchanged logic
# ---------------------------------------------------------------------------
def register_and_promote(run, model_name: str, artifact_path: str) -> str:
    mv      = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/{artifact_path}",
        name=model_name
    )
    version = mv.version
    log.info(f"Registered {model_name} v{version}")

    client.transition_model_version_stage(
        name=model_name, version=version, stage="Staging",
        archive_existing_versions=False,
    )
    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production",
        archive_existing_versions=True,
    )
    log.info(f"{model_name} v{version} => Production")
    return version


# ---------------------------------------------------------------------------
# Write promotion manifest to GCS  (replaces s3.put_object)
# ---------------------------------------------------------------------------
def write_promotion_manifest(rul_run, anomaly_run,
                             rul_version, anomaly_version,
                             validation_notes) -> str:
    manifest = {
        "promoted_at":   datetime.now(timezone.utc).isoformat(),
        "dataset":       DATASET,
        "rul_model": {
            "name":         RUL_MODEL_NAME,
            "version":      rul_version,
            "run_id":       rul_run.info.run_id,
            "gcs_uri":      rul_run.data.params.get("model_gcs_uri", ""),
            "mean_rmse":    rul_run.data.metrics.get("mean_rmse"),
            "mean_mae":     rul_run.data.metrics.get("mean_mae"),
            "mean_s_score": rul_run.data.metrics.get("mean_s_score"),
        },
        "anomaly_model": {
            "name":         ANOMALY_MODEL_NAME,
            "version":      anomaly_version,
            "run_id":       anomaly_run.info.run_id,
            "gcs_uri":      anomaly_run.data.params.get("artifacts_gcs_uri", ""),
            "threshold":    anomaly_run.data.metrics.get("threshold"),
            "detection_f1": anomaly_run.data.metrics.get("detection_f1"),
        },
        "validation": validation_notes,
    }

    key     = f"registry/{DATASET}/promotion_manifest.json"
    gcs_uri = gcs_write_json(GCS_ARTIFACTS_BUCKET, key, manifest)
    log.info(f"Promotion manifest written to {gcs_uri}")
    return gcs_uri


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global client

    log.info(f"Model registry starting — dataset={DATASET} "
             f"rmse_threshold={RMSE_THRESHOLD}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    rul_run, anomaly_run = wait_for_runs()

    rmse_ok, rmse_note = rmse_gate(rul_run)
    if not rmse_ok:
        log.error(f"PROMOTION REJECTED — {rmse_note}"); return

    drift_ok, drift_note = drift_check(rul_run)
    if not drift_ok:
        log.error(f"PROMOTION REJECTED — {drift_note}"); return

    log.info("Validation PASSED — proceeding with promotion")

    rul_version     = register_and_promote(rul_run,     RUL_MODEL_NAME,
                                           "xgboost_model")
    anomaly_version = register_and_promote(anomaly_run, ANOMALY_MODEL_NAME,
                                           "anomaly_model")

    manifest_uri = write_promotion_manifest(
        rul_run, anomaly_run, rul_version, anomaly_version,
        {"rmse_gate": rmse_note, "drift_check": drift_note},
    )

    log.info("=" * 60)
    log.info("PROMOTION COMPLETE")
    log.info(f"  RUL model     : {RUL_MODEL_NAME} v{rul_version} => Production")
    log.info(f"  Anomaly model : {ANOMALY_MODEL_NAME} v{anomaly_version} => Production")
    log.info(f"  Manifest      : {manifest_uri}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()