"""
train_RUL.py - XGBoost RUL Training
Mirrors existing preprocessing script exactly, adds MLflow tracking
and saves model artifact to GCS.

Two training modes controlled by TRAINING_MODE env var:

  offline - reads train_FD002.txt + test_FD002.txt + RUL_FD002.txt from GCS
             full dataset, best for initial model
             Upload files first:
               gsutil cp data/CMaps/train_FD002.txt gs://phm-raw-data-aide2-494008/offline/
               gsutil cp data/CMaps/test_FD002.txt  gs://phm-raw-data-aide2-494008/offline/
               gsutil cp data/CMaps/RUL_FD002.txt   gs://phm-raw-data-aide2-494008/offline/

  online  - reads engine_features table from PostgreSQL
            uses labeled rows (rul IS NOT NULL) for incremental retraining
            as live stream data accumulates in feature-platform namespace

Pipeline (matches existing script):
  1. Load data
  2. Generate RUL labels
  3. Feature selection - drop low-correlation sensors for FD002
  4. MinMaxScaler normalisation
  5. Random sample selection (50 train / 25 test per engine)
  6. XGBoost with GridSearchCV x 10 runs
  7. Evaluate: MSE - RMSE - MAE - MAPE - S-score
  8. Log to MLflow - params, metrics, model artifact
  9. Save best model => gs://phm-model-artifacts-aide2-494008/

Environment variables:
  TRAINING_MODE        offline | online  (default: offline)
  DATASET              FD002
  GCS_RAW_BUCKET       phm-raw-data-aide2-494008
  GCS_ARTIFACTS_BUCKET phm-model-artifacts-aide2-494008
  GCS_MLFLOW_BUCKET    phm-mlflow-artifacts-aide2-494008
  GCP_PROJECT          aide2-494008
  DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD
  N_RUNS               10
  RUL_CAP              125
"""

import io
import os
import json
import time
import logging
import pickle
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, timezone
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from urllib.parse import quote_plus
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAINING_MODE        = os.environ.get("TRAINING_MODE",        "offline")
DATASET              = os.environ.get("DATASET",               "FD002")
GCS_RAW_BUCKET       = os.environ.get("GCS_RAW_BUCKET",        "phm-raw-data-aide2-494008")
GCS_ARTIFACTS_BUCKET = os.environ.get("GCS_ARTIFACTS_BUCKET",  "phm-model-artifacts-aide2-494008")
GCS_MLFLOW_BUCKET    = os.environ.get("GCS_MLFLOW_BUCKET",     "phm-mlflow-artifacts-aide2-494008")
GCP_PROJECT          = os.environ.get("GCP_PROJECT",            "aide2-494008")
N_RUNS               = int(os.environ.get("N_RUNS",             "10"))
RUL_CAP              = int(os.environ.get("RUL_CAP",            "125"))

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

# ---------------------------------------------------------------------------
# FD002 feature selection
# ---------------------------------------------------------------------------
COLS_TO_DROP = [
    "OperSet1", "OperSet2", "OperSet3",
    "SensorMes1", "SensorMes5", "SensorMes10",
    "SensorMes18", "SensorMes19",
]

COLUMN_NAMES = (
    ["UnitNumber", "TimeInCycles"]
    + [f"OperSet{i}"   for i in range(1, 4)]
    + [f"SensorMes{j}" for j in range(1, 22)]
)

# ---------------------------------------------------------------------------
# GCS client  (replaces boto3.client("s3"))
# ---------------------------------------------------------------------------
gcs = storage.Client(project=GCP_PROJECT)


def gcs_read_bytes(bucket: str, key: str) -> bytes:
    return gcs.bucket(bucket).blob(key).download_as_bytes()


def gcs_write_bytes(bucket: str, key: str, data: bytes,
                    content_type: str = "application/octet-stream") -> str:
    gcs.bucket(bucket).blob(key).upload_from_string(
        data, content_type=content_type
    )
    return f"gs://{bucket}/{key}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_offline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train + test + RUL from GCS offline prefix."""
    log.info("Loading offline data from GCS...")

    def read_txt(key):
        return pd.read_csv(
            io.BytesIO(gcs_read_bytes(GCS_RAW_BUCKET, key)),
            sep=r"\s+", header=None, names=COLUMN_NAMES
        )

    def read_rul(key):
        return pd.read_csv(
            io.BytesIO(gcs_read_bytes(GCS_RAW_BUCKET, key)),
            sep=r"\s+", header=None, names=["RUL_FD"]
        )

    train = read_txt(f"offline/train_{DATASET}.txt")
    test  = read_txt(f"offline/test_{DATASET}.txt")
    rul   = read_rul(f"offline/RUL_{DATASET}.txt")

    log.info(f"Loaded train={len(train)} rows, test={len(test)} rows")
    return train, test, rul


def load_online() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load from PostgreSQL engine_features - labeled rows only."""
    log.info("Loading online data from PostgreSQL engine_features...")

    conn  = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD, connect_timeout=10,
    )
    query = """
        SELECT unit_id, cycle, rul, features
        FROM engine_features
        WHERE dataset = %s AND rul IS NOT NULL
        ORDER BY unit_id, cycle
    """
    df = pd.read_sql(query, conn, params=(DATASET,))
    conn.close()

    if len(df) == 0:
        raise ValueError(
            "No labeled rows found in engine_features. "
            "Run offline training first to populate RUL labels."
        )

    features_df = pd.json_normalize(df["features"].apply(json.loads))
    df = pd.concat([df[["unit_id", "cycle", "rul"]], features_df], axis=1)
    df = df.rename(columns={
        "unit_id": "UnitNumber", "cycle": "TimeInCycles", "rul": "RUL"
    })

    units       = df["UnitNumber"].unique()
    train_units = units[:int(len(units) * 0.8)]
    train = df[df["UnitNumber"].isin(train_units)].copy()
    test  = df[~df["UnitNumber"].isin(train_units)].copy()

    log.info(f"Online data - train={len(train)} rows, test={len(test)} rows")
    return train, test


# ---------------------------------------------------------------------------
# RUL generation (matches existing script exactly)
# ---------------------------------------------------------------------------
def rul_train_generation(df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = df.groupby("UnitNumber")["TimeInCycles"].max().rename("max")
    df = df.join(max_cycles, on="UnitNumber")
    df["RUL"] = df["max"] - df["TimeInCycles"]
    return df.drop(columns="max")


def rul_test_generation(test: pd.DataFrame, rul: pd.DataFrame) -> pd.DataFrame:
    rul["UnitNumber"] = rul.index + 1
    test = test.merge(rul, on="UnitNumber", how="left")
    max_cycle = test.groupby("UnitNumber")["TimeInCycles"].max().rename("max")
    test = test.join(max_cycle, on="UnitNumber")
    test["RUL"] = test["RUL_FD"] + test["max"] - test["TimeInCycles"]
    return test.drop(columns=["max", "RUL_FD"])


def cap_rul(df: pd.DataFrame) -> pd.DataFrame:
    df["RUL"] = df["RUL"].clip(upper=RUL_CAP)
    return df


def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])


def selection_aleatoire(df: pd.DataFrame, sample_size: int,
                        rand_state: int) -> pd.DataFrame:
    selected = []
    for value in df["UnitNumber"].unique():
        rows = df[df["UnitNumber"] == value]
        selected.append(rows.sample(n=min(sample_size, len(rows)),
                                    random_state=rand_state))
    return pd.concat(selected)


def normalised_df(train: pd.DataFrame,
                  test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    scaler      = MinMaxScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    test_scaled  = pd.DataFrame(scaler.fit_transform(test),  columns=test.columns)
    return train_scaled, test_scaled, scaler


def compute_s_score(rul_true, rul_pred) -> float:
    diff = rul_pred - rul_true
    return float(np.sum(
        np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    ))


# ---------------------------------------------------------------------------
# Save model artifact to GCS  (replaces s3.put_object)
# ---------------------------------------------------------------------------
def save_model_to_gcs(model, scaler, run_id: str, mean_rmse: float) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    key       = f"xgboost/{DATASET}/{timestamp}_rmse{mean_rmse:.4f}_run{run_id[:8]}.pkl"

    artifact = {
        "model":               model,
        "scaler":              scaler,
        "dataset":             DATASET,
        "feature_cols_dropped": COLS_TO_DROP,
        "rul_cap":             RUL_CAP,
    }
    buf = io.BytesIO()
    pickle.dump(artifact, buf)
    buf.seek(0)

    gcs_uri = gcs_write_bytes(GCS_ARTIFACTS_BUCKET, key, buf.read())
    log.info(f"Model saved to {gcs_uri}")
    return gcs_uri


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"phm-rul-xgboost-{DATASET}-{TRAINING_MODE}")

    log.info(f"Starting training - mode={TRAINING_MODE} n_runs={N_RUNS}")
    t_start = time.time()

    mse_list = []; rmse_list = []; mae_list = []
    mape_list = []; s_list = []

    best_rmse = float("inf"); best_model = None; best_scaler = None

    param_grid = {
        "n_estimators":     [100],
        "max_depth":        [3],
        "learning_rate":    [0.01],
        "subsample":        [0.5],
        "colsample_bytree": [0.5],
    }

    with mlflow.start_run(
        run_name=f"{DATASET}_{TRAINING_MODE}_{N_RUNS}runs"
    ) as run:

        mlflow.log_params({
            "dataset":       DATASET,
            "training_mode": TRAINING_MODE,
            "n_runs":        N_RUNS,
            "rul_cap":       RUL_CAP,
            "train_samples": 50,
            "test_samples":  25,
            **{f"xgb_{k}": v[0] for k, v in param_grid.items()},
        })

        for j in range(1, N_RUNS + 1):
            log.info(f"Run {j}/{N_RUNS}")

            norm_train, norm_test, scaler = normalised_df(train_df, test_df)
            train_sel = selection_aleatoire(norm_train, 50, j)
            test_sel  = selection_aleatoire(norm_test,  25, j)

            X_train = train_sel.drop("RUL", axis=1)
            Y_train = train_sel["RUL"]
            X_test  = test_sel.drop("RUL", axis=1)
            Y_test  = test_sel["RUL"]

            model = XGBRegressor()
            gs    = GridSearchCV(model, param_grid, cv=5,
                                 scoring="neg_mean_squared_error")
            gs.fit(X_train, Y_train)

            y_pred  = gs.predict(X_test)
            mse     = mean_squared_error(Y_test, y_pred)
            rmse    = np.sqrt(mse)
            mae     = mean_absolute_error(Y_test, y_pred)
            mape    = float(np.mean(np.abs((Y_test - y_pred) / (Y_test + 1e-8))) * 100)
            s_score = compute_s_score(Y_test.values, y_pred)

            mse_list.append(mse);   rmse_list.append(rmse)
            mae_list.append(mae);   mape_list.append(mape)
            s_list.append(s_score)

            log.info(f"  RMSE={rmse:.4f} MAE={mae:.4f} "
                     f"MAPE={mape:.2f}% S={s_score:.2f}")

            if rmse < best_rmse:
                best_rmse = rmse; best_model = gs.best_estimator_
                best_scaler = scaler

        mean_rmse = float(np.mean(rmse_list))
        mlflow.log_metrics({
            "mean_mse":     float(np.mean(mse_list)),
            "mean_rmse":    mean_rmse,
            "mean_mae":     float(np.mean(mae_list)),
            "mean_mape":    float(np.mean(mape_list)),
            "mean_s_score": float(np.mean(s_list)),
            "best_rmse":    best_rmse,
        })

        gcs_uri = save_model_to_gcs(best_model, best_scaler,
                                    run.info.run_id, mean_rmse)
        mlflow.log_param("model_gcs_uri", gcs_uri)
        mlflow.xgboost.log_model(best_model, artifact_path="xgboost_model")

        elapsed = (time.time() - t_start) / 60
        log.info(f"Training complete in {elapsed:.2f} min")
        log.info(f"mean RMSE = {mean_rmse:.4f} ({mean_rmse * 100:.2f}%)")
        log.info(f"Best model saved to {gcs_uri}")
        log.info(f"MLflow run_id: {run.info.run_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info(f"Model training starting - mode={TRAINING_MODE} dataset={DATASET}")

    if TRAINING_MODE == "offline":
        train_raw, test_raw, rul = load_offline()
        train_df = rul_train_generation(train_raw)
        test_df  = rul_test_generation(test_raw, rul)
        train_df = cap_rul(feature_selection(train_df))
        test_df  = cap_rul(feature_selection(test_df))
        log.info(f"Train shape: {train_df.shape} Test shape: {test_df.shape}")

    elif TRAINING_MODE == "online":
        train_df, test_df = load_online()
        train_df = cap_rul(feature_selection(train_df))
        test_df  = cap_rul(feature_selection(test_df))

    else:
        raise ValueError(
            f"Unknown TRAINING_MODE: {TRAINING_MODE}. Use 'offline' or 'online'."
        )

    train(train_df, test_df)


if __name__ == "__main__":
    main()