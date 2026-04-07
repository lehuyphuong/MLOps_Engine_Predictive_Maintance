"""
train_RUL.py — XGBoost RUL Training
Mirrors existing preprocessing script exactly, adds MLflow tracking
and saves model artifact to S3.

Two training modes controlled by TRAINING_MODE env var:

  offline — reads train_FD002.txt + test_FD002.txt + RUL_FD002.txt from S3
             full dataset, best for initial model
             Upload files first:
               aws s3 cp data/CMaps/train_FD002.txt s3://phm-raw-data/offline/
               aws s3 cp data/CMaps/test_FD002.txt  s3://phm-raw-data/offline/
               aws s3 cp data/CMaps/RUL_FD002.txt   s3://phm-raw-data/offline/

  online  — reads engine_features table from PostgreSQL
            uses labeled rows (rul IS NOT NULL) for incremental retraining
            as live stream data accumulates in feature-platform namespace

Pipeline (matches existing script):
  1. Load data
  2. Generate RUL labels
  3. Feature selection — drop low-correlation sensors for FD002
  4. MinMaxScaler normalisation
  5. Random sample selection (50 train / 25 test per engine)
  6. XGBoost with GridSearchCV × 10 runs
  7. Evaluate: MSE · RMSE · MAE · MAPE · S-score
  8. Log to MLflow — params, metrics, model artifact
  9. Save best model → s3://phm-model-artifacts/

Environment variables:
  TRAINING_MODE     offline | online  (default: offline)
  DATASET           FD002
  S3_RAW_BUCKET     phm-raw-data
  S3_ARTIFACTS_BUCKET  phm-model-artifacts
  S3_MLFLOW_BUCKET  phm-mlflow-artifacts
  MLFLOW_TRACKING_URI  postgresql+psycopg2://...
  DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD
  AWS_REGION        ap-southeast-1
  N_RUNS            10  (number of random sample runs)
  RUL_CAP           125
"""

import io
import os
import json
import time
import logging
import pickle
import boto3
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAINING_MODE       = os.environ.get("TRAINING_MODE",        "offline")
DATASET             = os.environ.get("DATASET",               "FD002")
S3_RAW_BUCKET       = os.environ.get("S3_RAW_BUCKET",         "phm-raw-data")
S3_ARTIFACTS_BUCKET = os.environ.get("S3_ARTIFACTS_BUCKET",   "phm-model-artifacts")
S3_MLFLOW_BUCKET    = os.environ.get("S3_MLFLOW_BUCKET",      "phm-mlflow-artifacts")
AWS_REGION          = os.environ.get("AWS_REGION",             "ap-southeast-1")
N_RUNS              = int(os.environ.get("N_RUNS",             "10"))
RUL_CAP             = int(os.environ.get("RUL_CAP",            "125"))

_db_host_raw = os.environ.get("DB_HOST", "localhost")
DB_HOST      = _db_host_raw.split(":")[0]
DB_PORT      = int(_db_host_raw.split(":")[1]) if ":" in _db_host_raw else int(os.environ.get("DB_PORT", "5432"))
DB_NAME             = os.environ.get("DB_NAME",    "phmdb")
DB_USER             = os.environ.get("DB_USER",    "phmadmin")
DB_PASSWORD         = os.environ.get("DB_PASSWORD", "")

MLFLOW_TRACKING_URI = (
    f"postgresql+psycopg2://{DB_USER}:{quote_plus(DB_PASSWORD)}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ---------------------------------------------------------------------------
# FD002 feature selection — drop low-correlation sensors (from existing script)
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

s3 = boto3.client("s3", region_name=AWS_REGION)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_offline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train + test + RUL from S3 offline prefix."""
    log.info("Loading offline data from S3...")

    def read_txt(key):
        obj = s3.get_object(Bucket=S3_RAW_BUCKET, Key=key)
        return pd.read_csv(
            io.BytesIO(obj["Body"].read()),
            sep=r"\s+", header=None, names=COLUMN_NAMES
        )

    def read_rul(key):
        obj = s3.get_object(Bucket=S3_RAW_BUCKET, Key=key)
        return pd.read_csv(
            io.BytesIO(obj["Body"].read()),
            sep=r"\s+", header=None, names=["RUL_FD"]
        )

    train = read_txt(f"offline/train_{DATASET}.txt")
    test  = read_txt(f"offline/test_{DATASET}.txt")
    rul   = read_rul(f"offline/RUL_{DATASET}.txt")

    log.info(f"Loaded train={len(train)} rows, test={len(test)} rows")
    return train, test, rul


def load_online() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load from PostgreSQL engine_features table.
    Rows with rul IS NOT NULL are used for training.
    Splits 80/20 by unit_id for train/test.
    """
    log.info("Loading online data from PostgreSQL engine_features...")

    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, connect_timeout=10,
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
            "Run the offline training first to populate RUL labels."
        )

    # Expand features JSON into columns
    features_df = pd.json_normalize(df["features"].apply(json.loads))
    df = pd.concat([df[["unit_id", "cycle", "rul"]], features_df], axis=1)
    df = df.rename(columns={"unit_id": "UnitNumber", "cycle": "TimeInCycles", "rul": "RUL"})

    # 80/20 split by unit_id
    units      = df["UnitNumber"].unique()
    train_size = int(len(units) * 0.8)
    train_units = units[:train_size]

    train = df[df["UnitNumber"].isin(train_units)].copy()
    test  = df[~df["UnitNumber"].isin(train_units)].copy()

    log.info(f"Online data — train={len(train)} rows, test={len(test)} rows")
    return train, test


# ---------------------------------------------------------------------------
# RUL generation (matches existing script exactly)
# ---------------------------------------------------------------------------
def rul_train_generation(df: pd.DataFrame) -> pd.DataFrame:
    """RUL = max_cycle - current_cycle per engine."""
    max_cycles = df.groupby("UnitNumber")["TimeInCycles"].max().rename("max")
    df = df.join(max_cycles, on="UnitNumber")
    df["RUL"] = df["max"] - df["TimeInCycles"]
    df.drop(columns="max", inplace=True)
    return df


def rul_test_generation(test: pd.DataFrame, rul: pd.DataFrame) -> pd.DataFrame:
    """RUL = RUL_FD + max_test_cycle - current_cycle."""
    rul["UnitNumber"] = rul.index + 1
    test = test.merge(rul, on="UnitNumber", how="left")
    max_cycle = test.groupby("UnitNumber")["TimeInCycles"].max().rename("max")
    test = test.join(max_cycle, on="UnitNumber")
    test["RUL"] = test["RUL_FD"] + test["max"] - test["TimeInCycles"]
    test.drop(columns=["max", "RUL_FD"], inplace=True)
    return test


def cap_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Piecewise linear RUL cap — standard C-MAPSS practice."""
    df["RUL"] = df["RUL"].clip(upper=RUL_CAP)
    return df


# ---------------------------------------------------------------------------
# Feature selection (matches existing script for FD002)
# ---------------------------------------------------------------------------
def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in COLS_TO_DROP if c in df.columns]
    return df.drop(columns=existing)


# ---------------------------------------------------------------------------
# Random sample selection (matches existing script exactly)
# ---------------------------------------------------------------------------
def selection_aleatoire(df: pd.DataFrame, sample_size: int, rand_state: int) -> pd.DataFrame:
    unique_values = df["UnitNumber"].unique()
    selected_rows = []
    for value in unique_values:
        rows = df[df["UnitNumber"] == value]
        n    = min(sample_size, len(rows))
        selected_rows.append(rows.sample(n=n, random_state=rand_state))
    return pd.concat(selected_rows)


# ---------------------------------------------------------------------------
# Normalisation (matches existing script exactly)
# ---------------------------------------------------------------------------
def normalised_df(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler      = MinMaxScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    test_scaled  = pd.DataFrame(scaler.fit_transform(test),  columns=test.columns)
    return train_scaled, test_scaled, scaler


# ---------------------------------------------------------------------------
# S-score (matches existing script exactly)
# ---------------------------------------------------------------------------
def compute_s_score(rul_true, rul_pred) -> float:
    diff = rul_pred - rul_true
    return float(np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)))


# ---------------------------------------------------------------------------
# Save model artifact to S3
# ---------------------------------------------------------------------------
def save_model_to_s3(model, scaler, run_id: str, mean_rmse: float) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    key       = f"xgboost/{DATASET}/{timestamp}_rmse{mean_rmse:.4f}_run{run_id[:8]}.pkl"

    artifact = {"model": model, "scaler": scaler, "dataset": DATASET,
                "feature_cols_dropped": COLS_TO_DROP, "rul_cap": RUL_CAP}

    buf = io.BytesIO()
    pickle.dump(artifact, buf)
    buf.seek(0)

    s3.put_object(Bucket=S3_ARTIFACTS_BUCKET, Key=key,
                  Body=buf.read(), ContentType="application/octet-stream")

    s3_uri = f"s3://{S3_ARTIFACTS_BUCKET}/{key}"
    log.info(f"Model saved to {s3_uri}")
    return s3_uri


# ---------------------------------------------------------------------------
# Main training loop (mirrors existing script × 10 runs)
# ---------------------------------------------------------------------------
def train(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:

    log.info(f"MLFLOW_TRACKING_URI = {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"phm-rul-xgboost-{DATASET}-{TRAINING_MODE}")

    log.info(f"Starting training — mode={TRAINING_MODE} n_runs={N_RUNS}")

    t_start = time.time()

    mse_list  = []
    rmse_list = []
    mae_list  = []
    mape_list = []
    s_list    = []

    best_rmse  = float("inf")
    best_model = None
    best_scaler = None

    param_grid = {
        "n_estimators":    [100],
        "max_depth":       [3],
        "learning_rate":   [0.01],
        "subsample":       [0.5],
        "colsample_bytree":[0.5],
    }

    with mlflow.start_run(run_name=f"{DATASET}_{TRAINING_MODE}_{N_RUNS}runs") as run:

        # Log fixed params
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

            # Normalise
            norm_train, norm_test, scaler = normalised_df(train_df, test_df)

            # Random sample selection
            train_sel = selection_aleatoire(norm_train, 50, j)
            test_sel  = selection_aleatoire(norm_test,  25, j)

            # Split X/Y
            X_train = train_sel.drop("RUL", axis=1)
            Y_train = train_sel["RUL"]
            X_test  = test_sel.drop("RUL", axis=1)
            Y_test  = test_sel["RUL"]

            # Train
            model = XGBRegressor()
            gs    = GridSearchCV(model, param_grid, cv=5,
                                 scoring="neg_mean_squared_error")
            gs.fit(X_train, Y_train)

            # Evaluate
            y_pred   = gs.predict(X_test)
            mse      = mean_squared_error(Y_test, y_pred)
            rmse     = np.sqrt(mse)
            mae      = mean_absolute_error(Y_test, y_pred)
            mape     = float(np.mean(np.abs((Y_test - y_pred) / (Y_test + 1e-8))) * 100)
            s_score  = compute_s_score(Y_test.values, y_pred)

            mse_list.append(mse)
            rmse_list.append(rmse)
            mae_list.append(mae)
            mape_list.append(mape)
            s_list.append(s_score)

            log.info(f"  RMSE={rmse:.4f} MAE={mae:.4f} MAPE={mape:.2f}% S={s_score:.2f}")

            # Track best model
            if rmse < best_rmse:
                best_rmse   = rmse
                best_model  = gs.best_estimator_
                best_scaler = scaler

        # Log aggregate metrics
        mean_rmse = float(np.mean(rmse_list))
        mlflow.log_metrics({
            "mean_mse":    float(np.mean(mse_list)),
            "mean_rmse":   mean_rmse,
            "mean_mae":    float(np.mean(mae_list)),
            "mean_mape":   float(np.mean(mape_list)),
            "mean_s_score":float(np.mean(s_list)),
            "best_rmse":   best_rmse,
        })

        # Save best model
        s3_uri = save_model_to_s3(best_model, best_scaler, run.info.run_id, mean_rmse)
        mlflow.log_param("model_s3_uri", s3_uri)
        mlflow.xgboost.log_model(best_model, artifact_path="xgboost_model")

        elapsed = (time.time() - t_start) / 60
        log.info(f"Training complete in {elapsed:.2f} min")
        log.info(f"mean RMSE = {mean_rmse:.4f} ({mean_rmse * 100:.2f}%)")
        log.info(f"mean MAE  = {np.mean(mae_list) * 100:.2f}%")
        log.info(f"mean MAPE = {np.mean(mape_list):.2f}%")
        log.info(f"Best model saved to {s3_uri}")
        log.info(f"MLflow run_id: {run.info.run_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info(f"Model training starting — mode={TRAINING_MODE} dataset={DATASET}")

    if TRAINING_MODE == "offline":
        train_raw, test_raw, rul = load_offline()

        # Generate RUL labels (matches existing script)
        train_df = rul_train_generation(train_raw)
        test_df  = rul_test_generation(test_raw, rul)

        # Cap RUL
        train_df = cap_rul(train_df)
        test_df  = cap_rul(test_df)

        # Feature selection
        train_df = feature_selection(train_df)
        test_df  = feature_selection(test_df)

        log.info(f"Features: {list(train_df.columns)}")
        log.info(f"Train shape: {train_df.shape} Test shape: {test_df.shape}")

    elif TRAINING_MODE == "online":
        train_df, test_df = load_online()
        train_df = cap_rul(train_df)
        test_df  = cap_rul(test_df)
        train_df = feature_selection(train_df)
        test_df  = feature_selection(test_df)

        log.info(f"Online features: {list(train_df.columns)}")

    else:
        raise ValueError(f"Unknown TRAINING_MODE: {TRAINING_MODE}. Use 'offline' or 'online'.")

    train(train_df, test_df)


if __name__ == "__main__":
    main()