"""
validate.py — Model Validation Gate

Runs after train_RUL.py and train_anomaly.py complete.
Two gates must both pass before promote.py is allowed to run:

  1. RMSE gate
     Reads the latest MLflow run in the phm-rul-xgboost-FD002-* experiment.
     Fails if mean_rmse >= RMSE_THRESHOLD (default 0.30).

  2. Drift check
     Compares the mean of each feature column in the current training data
     against the baseline stored in the previous production MLflow run.
     Fails if any feature mean has drifted more than DRIFT_TOLERANCE (0.05).
     On first run (no previous production model) the drift check is skipped.

Exit codes:
  0 — both gates passed   → Airflow marks task SUCCESS → promote runs
  1 — at least one failed → Airflow marks task FAILED  → promote skipped

Environment variables (from model-training-config ConfigMap):
  RMSE_THRESHOLD    0.30
  DRIFT_TOLERANCE   0.05
  TRAINING_MODE     offline | online
  DATASET           FD002
  DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD
"""

import io
import json
import logging
import os
import sys
from urllib.parse import quote_plus

import mlflow
import numpy as np
import pandas as pd
import psycopg2
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET          = os.environ.get("DATASET",         "FD002")
TRAINING_MODE    = os.environ.get("TRAINING_MODE",   "offline")
RMSE_THRESHOLD   = float(os.environ.get("RMSE_THRESHOLD",  "0.30"))
DRIFT_TOLERANCE  = float(os.environ.get("DRIFT_TOLERANCE", "0.05"))
GCS_RAW_BUCKET   = os.environ.get("GCS_RAW_BUCKET",  "phm-raw-data-aide2-494008")
GCP_PROJECT      = os.environ.get("GCP_PROJECT",     "aide2-494008")

_db_host_raw = os.environ.get("DB_HOST", "localhost")
DB_HOST      = _db_host_raw.split(":")[0]
DB_PORT      = int(_db_host_raw.split(":")[1]) if ":" in _db_host_raw \
               else int(os.environ.get("DB_PORT", "5432"))
DB_NAME      = os.environ.get("DB_NAME",     "phmdb")
DB_USER      = os.environ.get("DB_USER",     "phmadmin")
DB_PASSWORD  = os.environ.get("DB_PASSWORD", "")

MLFLOW_DB_NAME  = os.environ.get("MLFLOW_DB_NAME", "mlflowdb")

MLFLOW_TRACKING_URI = (
    f"postgresql+psycopg2://{DB_USER}:{quote_plus(DB_PASSWORD)}"
    f"@{DB_HOST}:{DB_PORT}/{MLFLOW_DB_NAME}"
)

COLS_TO_DROP = [
    "OperSet1", "OperSet2", "OperSet3",
    "SensorMes1", "SensorMes5", "SensorMes10",
    "SensorMes18", "SensorMes19",
]
COLUMN_NAMES = (
    ["UnitNumber", "TimeInCycles"]
    + [f"OperSet{i}" for i in range(1, 4)]
    + [f"SensorMes{j}" for j in range(1, 22)]
)


# ── Gate 1: RMSE ──────────────────────────────────────────────────────────────

def check_rmse() -> tuple[bool, float]:
    """
    Find the most recent completed RUL training run and read its mean_rmse.
    Returns (passed, mean_rmse).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment_name = f"phm-rul-xgboost-{DATASET}-{TRAINING_MODE}"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        log.error("MLflow experiment '%s' not found. "
                  "Has train_RUL.py run at least once?", experiment_name)
        return False, float("inf")

    runs = mlflow.search_runs(
        experiment_ids = [experiment.experiment_id],
        filter_string  = "status = 'FINISHED'",
        order_by       = ["start_time DESC"],
        max_results    = 1,
    )

    if runs.empty:
        log.error("No finished runs in experiment '%s'.", experiment_name)
        return False, float("inf")

    mean_rmse = float(runs.iloc[0].get("metrics.mean_rmse", float("inf")))
    passed    = mean_rmse < RMSE_THRESHOLD

    log.info(
        "RMSE gate — mean_rmse=%.4f threshold=%.2f passed=%s",
        mean_rmse, RMSE_THRESHOLD, passed,
    )
    return passed, mean_rmse


# ── Gate 2: Feature drift ─────────────────────────────────────────────────────

def load_current_feature_means() -> dict[str, float]:
    """
    Load training data and compute per-feature column means.
    Mirrors the feature selection from train_RUL.py exactly.
    """
    if TRAINING_MODE == "offline":
        gcs    = storage.Client(project=GCP_PROJECT)
        data   = gcs.bucket(GCS_RAW_BUCKET).blob(
            f"offline/train_{DATASET}.txt"
        ).download_as_bytes()
        df = pd.read_csv(
            io.BytesIO(data), sep=r"\s+", header=None, names=COLUMN_NAMES
        )
    else:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD, connect_timeout=10,
        )
        df = pd.read_sql(
            "SELECT * FROM engine_features WHERE dataset = %s", conn,
            params=(DATASET,)
        )
        conn.close()

    feature_cols = [c for c in df.columns
                    if c not in COLS_TO_DROP
                    and c not in ("UnitNumber", "TimeInCycles", "RUL",
                                  "unit_id", "cycle", "id", "processed_at",
                                  "source_key", "dataset")]

    return {col: float(df[col].mean()) for col in feature_cols if col in df.columns}


def load_baseline_feature_means() -> dict[str, float] | None:
    """
    Read feature means logged by the previous production run from MLflow.
    Returns None if no production run exists yet (first run — skip drift check).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment_name = f"phm-rul-xgboost-{DATASET}-{TRAINING_MODE}"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    # Look for runs tagged as production baseline
    runs = mlflow.search_runs(
        experiment_ids = [experiment.experiment_id],
        filter_string  = "tags.production_baseline = 'true' and status = 'FINISHED'",
        order_by       = ["start_time DESC"],
        max_results    = 1,
    )

    if runs.empty:
        log.info("No production baseline run found — drift check skipped (first run).")
        return None

    run_id = runs.iloc[0]["run_id"]
    client = mlflow.tracking.MlflowClient()

    try:
        artifact_path = client.download_artifacts(run_id, "feature_means.json")
        with open(artifact_path) as f:
            return json.load(f)
    except Exception as e:
        log.warning("Could not load feature_means.json from baseline run: %s", e)
        return None


def check_drift(current_means: dict, baseline_means: dict) -> tuple[bool, float]:
    """
    Compute max absolute mean shift across all common features.
    Returns (passed, max_drift).
    """
    drifts = []
    for col, baseline_val in baseline_means.items():
        if col in current_means:
            drift = abs(current_means[col] - baseline_val)
            drifts.append(drift)

    if not drifts:
        log.warning("No common features between current and baseline — drift check skipped.")
        return True, 0.0

    max_drift = float(max(drifts))
    passed    = max_drift < DRIFT_TOLERANCE

    log.info(
        "Drift check — max_drift=%.4f tolerance=%.2f passed=%s",
        max_drift, DRIFT_TOLERANCE, passed,
    )
    return passed, max_drift


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info(
        "Validation starting — dataset=%s mode=%s "
        "rmse_threshold=%.2f drift_tolerance=%.2f",
        DATASET, TRAINING_MODE, RMSE_THRESHOLD, DRIFT_TOLERANCE,
    )

    results = {}

    # Gate 1: RMSE
    rmse_passed, mean_rmse = check_rmse()
    results["rmse_gate"] = {"mean_rmse": mean_rmse, "passed": rmse_passed}

    # Gate 2: Drift
    current_means  = load_current_feature_means()
    baseline_means = load_baseline_feature_means()

    if baseline_means is None:
        # First run — no baseline to compare against
        drift_passed, max_drift = True, 0.0
        results["drift_check"] = {"skipped": True, "reason": "no_baseline"}
    else:
        drift_passed, max_drift = check_drift(current_means, baseline_means)
        results["drift_check"] = {"max_drift": max_drift, "passed": drift_passed}

    # Log current feature means to MLflow so the next run can use them as baseline
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = f"phm-rul-xgboost-{DATASET}-{TRAINING_MODE}"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs.empty and rmse_passed and drift_passed:
            run_id = runs.iloc[0]["run_id"]
            client = mlflow.tracking.MlflowClient()
            means_path = "/tmp/feature_means.json"
            with open(means_path, "w") as f:
                json.dump(current_means, f)
            client.log_artifact(run_id, means_path)
            client.set_tag(run_id, "production_baseline", "true")
            log.info("Tagged run %s as production_baseline.", run_id)

    # Verdict
    overall = rmse_passed and drift_passed
    log.info("Validation result: %s", "PASSED" if overall else "FAILED")
    log.info("Details: %s", json.dumps(results, indent=2))

    if not overall:
        log.error(
            "Validation FAILED — promote will be skipped. "
            "Fix the issues above and re-trigger the DAG."
        )
        sys.exit(1)

    log.info("Validation PASSED — promote will run next.")
    sys.exit(0)


if __name__ == "__main__":
    main()