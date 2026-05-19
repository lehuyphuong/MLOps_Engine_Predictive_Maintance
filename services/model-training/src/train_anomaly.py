"""
train_anomaly.py — Anomaly Detection Model Training
Trains a PyTorch LSTM AutoEncoder on the normal engine pool (RUL > 150) from FD002.

Architecture:
  LSTM AutoEncoder — PyTorch
  - Trained on NORMAL pool only (RUL > 150 cycles)
  - Learns to reconstruct healthy sensor patterns
  - Reconstruction error threshold saved as artifact
  - At inference: high reconstruction error = anomaly detected

Pipeline:
  1. Load train_FD002.txt from GCS offline prefix
  2. Generate RUL labels
  3. Feature selection (same 16 features as RUL model)
  4. Split into normal pool (RUL > 150) and degradation pool
  5. Per-condition normalisation (FD002 has 6 operating conditions)
  6. Build overlapping sequences of length SEQUENCE_LENGTH
  7. Train PyTorch LSTM AutoEncoder on normal pool only
  8. Compute reconstruction error on both pools
  9. Set threshold = mean + 3*std of normal pool errors
  10. Evaluate: precision, recall, F1 on degradation pool detection
  11. Save model + threshold + scaler => GCS
  12. Log everything to MLflow

Environment variables:
  GCS_RAW_BUCKET         phm-raw-data-aide2-494008
  GCS_ARTIFACTS_BUCKET   phm-model-artifacts-aide2-494008
  GCP_PROJECT            aide2-494008
  DATASET                FD002
  NORMAL_RUL_THRESHOLD   150
  SEQUENCE_LENGTH        30
  EPOCHS                 50
  BATCH_SIZE             32
  LEARNING_RATE          0.001
  MLFLOW_TRACKING_URI    postgresql+psycopg2://...
"""

import io
import os
import pickle
import logging
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timezone
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from urllib.parse import quote_plus
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GCS_RAW_BUCKET       = os.environ.get("GCS_RAW_BUCKET",        "phm-raw-data-aide2-494008")
GCS_ARTIFACTS_BUCKET = os.environ.get("GCS_ARTIFACTS_BUCKET",  "phm-model-artifacts-aide2-494008")
GCP_PROJECT          = os.environ.get("GCP_PROJECT",            "aide2-494008")
DATASET              = os.environ.get("DATASET",                "FD002")
NORMAL_RUL_THRESHOLD = int(os.environ.get("NORMAL_RUL_THRESHOLD", "150"))
SEQUENCE_LENGTH      = int(os.environ.get("SEQUENCE_LENGTH",    "30"))
EPOCHS               = int(os.environ.get("EPOCHS",             "50"))
BATCH_SIZE           = int(os.environ.get("BATCH_SIZE",         "32"))
LEARNING_RATE        = float(os.environ.get("LEARNING_RATE",    "0.001"))

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# PyTorch LSTM AutoEncoder — unchanged from original
# ---------------------------------------------------------------------------
class LSTMAutoEncoder(nn.Module):
    def __init__(self, n_features: int, seq_len: int,
                 hidden1: int = 64, hidden2: int = 32, latent: int = 16):
        super().__init__()
        self.seq_len    = seq_len
        self.n_features = n_features
        self.latent     = latent

        self.enc_lstm1  = nn.LSTM(n_features, hidden1, batch_first=True)
        self.enc_drop1  = nn.Dropout(0.2)
        self.enc_lstm2  = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.enc_linear = nn.Linear(hidden2, latent)

        self.dec_linear = nn.Linear(latent, hidden2)
        self.dec_lstm1  = nn.LSTM(hidden2, hidden2, batch_first=True)
        self.dec_drop1  = nn.Dropout(0.2)
        self.dec_lstm2  = nn.LSTM(hidden2, hidden1, batch_first=True)
        self.dec_output = nn.Linear(hidden1, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _          = self.enc_lstm1(x)
        out             = self.enc_drop1(out)
        out, (h, _)     = self.enc_lstm2(out)
        bottleneck      = torch.relu(self.enc_linear(h[-1]))
        dec_in          = torch.relu(self.dec_linear(bottleneck))
        dec_in          = dec_in.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _          = self.dec_lstm1(dec_in)
        out             = self.dec_drop1(out)
        out, _          = self.dec_lstm2(out)
        return self.dec_output(out)


# ---------------------------------------------------------------------------
# Data loading  (replaces s3.get_object)
# ---------------------------------------------------------------------------
def load_train() -> pd.DataFrame:
    log.info(f"Loading train_{DATASET}.txt from GCS...")
    df = pd.read_csv(
        io.BytesIO(gcs_read_bytes(GCS_RAW_BUCKET, f"offline/train_{DATASET}.txt")),
        sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    log.info(f"Loaded {len(df)} rows, {df['UnitNumber'].nunique()} engines")
    return df


# ---------------------------------------------------------------------------
# Preprocessing — all unchanged from original
# ---------------------------------------------------------------------------
def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = df.groupby("UnitNumber")["TimeInCycles"].max().rename("max")
    df = df.join(max_cycles, on="UnitNumber")
    df["RUL"] = df["max"] - df["TimeInCycles"]
    return df.drop(columns="max")


def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c not in ("UnitNumber", "TimeInCycles", "RUL")]


def normalise_per_condition(df: pd.DataFrame,
                            feature_cols: list) -> tuple[pd.DataFrame, dict]:
    scalers = {}
    norm_df = df.copy()
    if "OperSet3" in df.columns:
        for cond in df["OperSet3"].unique():
            mask   = norm_df["OperSet3"] == cond
            scaler = MinMaxScaler()
            norm_df.loc[mask, feature_cols] = scaler.fit_transform(
                norm_df.loc[mask, feature_cols]
            )
            scalers[float(cond)] = scaler
        log.info(f"Normalised per condition — {len(scalers)} conditions")
    else:
        scaler = MinMaxScaler()
        norm_df[feature_cols] = scaler.fit_transform(norm_df[feature_cols])
        scalers["global"] = scaler
    return norm_df, scalers


def apply_scalers(df: pd.DataFrame, scalers: dict,
                  feature_cols: list) -> pd.DataFrame:
    norm_df = df.copy()
    if "global" in scalers:
        norm_df[feature_cols] = scalers["global"].transform(norm_df[feature_cols])
    elif "OperSet3" in df.columns:
        for cond, scaler in scalers.items():
            mask = norm_df["OperSet3"] == cond
            if mask.any():
                norm_df.loc[mask, feature_cols] = scaler.transform(
                    norm_df.loc[mask, feature_cols]
                )
    return norm_df


def build_sequences(df: pd.DataFrame, feature_cols: list,
                    seq_len: int) -> np.ndarray:
    sequences = []
    for _, group in df.groupby("UnitNumber"):
        data = group.sort_values("TimeInCycles")[feature_cols].values
        for i in range(len(data) - seq_len + 1):
            sequences.append(data[i: i + seq_len])
    return np.array(sequences, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training + evaluation — unchanged from original
# ---------------------------------------------------------------------------
def train_autoencoder(model, X_train, epochs, batch_size, lr) -> list:
    dataset   = TensorDataset(torch.tensor(X_train))
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.to(DEVICE); model.train()
    losses = []; best_loss = float("inf"); patience = 5; no_improve = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward(); optimizer.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= len(X_train)
        losses.append(epoch_loss)

        if epoch % 10 == 0:
            log.info(f"  Epoch {epoch:3d}/{epochs} — loss={epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    return losses


def get_reconstruction_errors(model, sequences: np.ndarray) -> np.ndarray:
    model.eval(); errors = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(torch.tensor(sequences)),
                                   batch_size=64, shuffle=False):
            batch = batch.to(DEVICE)
            mse   = torch.mean((batch - model(batch)) ** 2, dim=(1, 2))
            errors.extend(mse.cpu().numpy())
    return np.array(errors)


# ---------------------------------------------------------------------------
# Save artifacts to GCS  (replaces s3.put_object)
# ---------------------------------------------------------------------------
def save_artifacts(model, scalers, threshold, n_features, run_id) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    prefix    = f"autoencoder/{DATASET}/{timestamp}_run{run_id[:8]}"

    model_buf = io.BytesIO()
    torch.save({
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "seq_len":    SEQUENCE_LENGTH,
    }, model_buf)
    model_buf.seek(0)
    gcs_write_bytes(GCS_ARTIFACTS_BUCKET, f"{prefix}/autoencoder.pt",
                    model_buf.read())

    pkl_buf = io.BytesIO()
    pickle.dump({
        "scalers":              scalers,
        "threshold":            threshold,
        "dataset":              DATASET,
        "seq_len":              SEQUENCE_LENGTH,
        "n_features":           n_features,
        "normal_rul_threshold": NORMAL_RUL_THRESHOLD,
        "feature_cols_dropped": COLS_TO_DROP,
    }, pkl_buf)
    pkl_buf.seek(0)
    gcs_write_bytes(GCS_ARTIFACTS_BUCKET, f"{prefix}/anomaly_artifacts.pkl",
                    pkl_buf.read())

    gcs_uri = f"gs://{GCS_ARTIFACTS_BUCKET}/{prefix}/"
    log.info(f"Artifacts saved to {gcs_uri}")
    return gcs_uri


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info(f"Anomaly training starting — dataset={DATASET} "
             f"normal_threshold=RUL>{NORMAL_RUL_THRESHOLD} device={DEVICE}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"phm-anomaly-autoencoder-{DATASET}")

    df           = add_rul(load_train())
    df           = feature_selection(df)
    feature_cols = get_feature_cols(df)
    n_features   = len(feature_cols)
    log.info(f"Features: {n_features} — {feature_cols}")

    normal_pool = df[df["RUL"] > NORMAL_RUL_THRESHOLD].copy()
    degrad_pool = df[df["RUL"] <= NORMAL_RUL_THRESHOLD].copy()
    log.info(f"Normal pool: {len(normal_pool)} rows | "
             f"Degradation pool: {len(degrad_pool)} rows")

    normal_norm, scalers = normalise_per_condition(normal_pool, feature_cols)
    degrad_norm          = apply_scalers(degrad_pool, scalers, feature_cols)

    X_normal = build_sequences(normal_norm, feature_cols, SEQUENCE_LENGTH)
    X_degrad = build_sequences(degrad_norm, feature_cols, SEQUENCE_LENGTH)
    log.info(f"Normal sequences: {X_normal.shape} | "
             f"Degraded sequences: {X_degrad.shape}")

    if len(X_normal) == 0:
        raise ValueError("No normal sequences — increase NORMAL_RUL_THRESHOLD")

    with mlflow.start_run(run_name=f"autoencoder_{DATASET}") as run:
        mlflow.log_params({
            "dataset":              DATASET,
            "normal_rul_threshold": NORMAL_RUL_THRESHOLD,
            "sequence_length":      SEQUENCE_LENGTH,
            "n_features":           n_features,
            "epochs":               EPOCHS,
            "batch_size":           BATCH_SIZE,
            "learning_rate":        LEARNING_RATE,
            "normal_pool_size":     len(X_normal),
            "degraded_pool_size":   len(X_degrad),
            "architecture":         "LSTM-AE-64-32-16",
            "framework":            "pytorch",
        })

        model  = LSTMAutoEncoder(n_features, SEQUENCE_LENGTH)
        losses = train_autoencoder(model, X_normal, EPOCHS,
                                   BATCH_SIZE, LEARNING_RATE)
        log.info(f"Training complete — {len(losses)} epochs, "
                 f"final loss={losses[-1]:.6f}")

        normal_errors = get_reconstruction_errors(model, X_normal)
        degrad_errors = get_reconstruction_errors(model, X_degrad)

        threshold = float(np.mean(normal_errors) + 3 * np.std(normal_errors))
        log.info(f"Threshold: {threshold:.6f}")

        y_true    = np.ones(len(degrad_errors), dtype=int)
        y_pred    = (degrad_errors > threshold).astype(int)
        report    = classification_report(y_true, y_pred,
                                          output_dict=True, zero_division=0)
        precision = report.get("1", {}).get("precision", 0.0)
        recall    = report.get("1", {}).get("recall",    0.0)
        f1        = report.get("1", {}).get("f1-score",  0.0)
        log.info(f"Detection — precision={precision:.3f} "
                 f"recall={recall:.3f} f1={f1:.3f}")

        mlflow.log_metrics({
            "threshold":           threshold,
            "normal_mean_error":   float(np.mean(normal_errors)),
            "normal_std_error":    float(np.std(normal_errors)),
            "degraded_mean_error": float(np.mean(degrad_errors)),
            "epochs_trained":      len(losses),
            "final_train_loss":    losses[-1],
            "detection_precision": precision,
            "detection_recall":    recall,
            "detection_f1":        f1,
        })

        gcs_uri = save_artifacts(model, scalers, threshold,
                                 n_features, run.info.run_id)
        mlflow.log_param("artifacts_gcs_uri", gcs_uri)

        log.info(f"Anomaly model complete — threshold={threshold:.6f} "
                 f"f1={f1:.3f} artifacts={gcs_uri}")


if __name__ == "__main__":
    main()