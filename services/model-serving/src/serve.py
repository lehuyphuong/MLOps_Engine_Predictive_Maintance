"""
serve.py — Model Serving (FastAPI)
Inference Orchestrator — triggered per cycle event.

Three components per diagram:
  1. Inference Orchestrator  — FastAPI, triggered per cycle event
  2. RUL Prediction Model    — 24-cycle sequence input from Redis window
  3. Anomaly Detection Model — latest single sensor reading

Output per inference:
  engine_id · RUL · anomaly_score · anomaly_flag
  Written to GCS for alert-naming namespace to consume.

Endpoints:
  POST /infer/{unit_id}         -> full inference (RUL + anomaly)
  POST /infer/{unit_id}/rul     -> RUL only
  POST /infer/{unit_id}/anomaly -> anomaly only
  GET  /health                  -> liveness probe
  GET  /model/info              -> loaded model versions

  POST /trigger                 -> called by stream_processor after each batch
                                   triggers inference for all active engine IDs
                                   THIS FIXES the missing pipeline link between
                                   feature-platform and model-serving

Environment variables:
  GCS_ARTIFACTS_BUCKET    phm-model-artifacts-aide2-494008
  GCS_RAW_BUCKET          phm-raw-data-aide2-494008
  GCP_PROJECT             aide2-494008
  REDIS_HOST              Memorystore endpoint
  REDIS_PORT              6379
  DATASET                 FD002
  WINDOW_SIZE             30
  PORT                    8080
"""

import io
import os
import json
import pickle
import logging
import redis
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GCS_ARTIFACTS_BUCKET = os.environ.get("GCS_ARTIFACTS_BUCKET", "phm-model-artifacts-aide2-494008")
GCS_RAW_BUCKET       = os.environ.get("GCS_RAW_BUCKET",       "phm-raw-data-aide2-494008")
GCP_PROJECT          = os.environ.get("GCP_PROJECT",           "aide2-494008")
DATASET              = os.environ.get("DATASET",               "FD002")
REDIS_HOST           = os.environ.get("REDIS_HOST",            "localhost")
REDIS_PORT           = int(os.environ.get("REDIS_PORT",        "6379"))
WINDOW_SIZE          = int(os.environ.get("WINDOW_SIZE",       "30"))

MANIFEST_KEY   = f"registry/{DATASET}/promotion_manifest.json"
RESULTS_PREFIX = f"inference-results/{DATASET}/"

COLS_TO_DROP = [
    "OperSet1", "OperSet2", "OperSet3",
    "SensorMes1", "SensorMes5", "SensorMes10",
    "SensorMes18", "SensorMes19",
]

DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# GCS client  (replaces boto3.client("s3"))
# ---------------------------------------------------------------------------
gcs = storage.Client(project=GCP_PROJECT)


def gcs_read_bytes(bucket: str, key: str) -> bytes:
    return gcs.bucket(bucket).blob(key).download_as_bytes()


def gcs_write_json(bucket: str, key: str, data: dict) -> None:
    gcs.bucket(bucket).blob(key).upload_from_string(
        json.dumps(data),
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# PyTorch LSTM AutoEncoder — must match train_anomaly.py exactly
# ---------------------------------------------------------------------------
class LSTMAutoEncoder(nn.Module):
    def __init__(self, n_features: int, seq_len: int,
                 hidden1: int = 64, hidden2: int = 32, latent: int = 16):
        super().__init__()
        self.seq_len    = seq_len
        self.n_features = n_features
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
        out, _      = self.enc_lstm1(x)
        out          = self.enc_drop1(out)
        out, (h, _) = self.enc_lstm2(out)
        bottleneck   = torch.relu(self.enc_linear(h[-1]))
        dec_in       = torch.relu(self.dec_linear(bottleneck))
        dec_in       = dec_in.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _       = self.dec_lstm1(dec_in)
        out          = self.dec_drop1(out)
        out, _       = self.dec_lstm2(out)
        return self.dec_output(out)


# ---------------------------------------------------------------------------
# Model store — loads artifacts from GCS on startup
# ---------------------------------------------------------------------------
class ModelStore:
    def __init__(self):
        self.rul_model         = None
        self.rul_scaler        = None
        self.anomaly_model     = None
        self.anomaly_scalers   = None
        self.anomaly_threshold = None
        self.anomaly_seq_len   = 30
        self.n_features        = None
        self.manifest          = None
        self.loaded_at         = None

    def load(self):
        log.info("Loading promotion manifest from GCS...")

        # Load manifest  (replaces s3.get_object for manifest)
        self.manifest = json.loads(
            gcs_read_bytes(GCS_ARTIFACTS_BUCKET, MANIFEST_KEY)
        )
        log.info(f"Manifest loaded — promoted_at={self.manifest['promoted_at']}")

        # Load XGBoost RUL model  (replaces s3.get_object for rul pkl)
        rul_key = self.manifest["rul_model"]["gcs_uri"].replace(
            f"gs://{GCS_ARTIFACTS_BUCKET}/", ""
        )
        artifact        = pickle.loads(gcs_read_bytes(GCS_ARTIFACTS_BUCKET, rul_key))
        self.rul_model  = artifact["model"]
        self.rul_scaler = artifact["scaler"]
        log.info(
            f"XGBoost RUL model loaded — "
            f"RMSE={self.manifest['rul_model']['mean_rmse']:.4f}"
        )

        # Load AutoEncoder anomaly artifacts  (replaces two s3.get_object calls)
        prefix = self.manifest["anomaly_model"]["gcs_uri"].replace(
            f"gs://{GCS_ARTIFACTS_BUCKET}/", ""
        )
        pkl                    = pickle.loads(
            gcs_read_bytes(GCS_ARTIFACTS_BUCKET, f"{prefix}anomaly_artifacts.pkl")
        )
        self.anomaly_scalers   = pkl["scalers"]
        self.anomaly_threshold = pkl["threshold"]
        self.anomaly_seq_len   = pkl.get("seq_len", 30)
        self.n_features        = pkl["n_features"]

        # Load PyTorch weights
        pt_data = torch.load(
            io.BytesIO(gcs_read_bytes(GCS_ARTIFACTS_BUCKET, f"{prefix}autoencoder.pt")),
            map_location=DEVICE
        )
        ae = LSTMAutoEncoder(self.n_features, self.anomaly_seq_len)
        ae.load_state_dict(pt_data["state_dict"])
        ae.eval()
        self.anomaly_model = ae
        log.info(
            f"AutoEncoder loaded — "
            f"threshold={self.anomaly_threshold:.6f} "
            f"seq_len={self.anomaly_seq_len} "
            f"n_features={self.n_features}"
        )

        self.loaded_at = datetime.now(timezone.utc).isoformat()
        log.info("All models ready")


# ---------------------------------------------------------------------------
# Redis — reads feature windows written by feature-platform
# ---------------------------------------------------------------------------
class RedisStore:
    def __init__(self):
        self.r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=5,
        )

    def get_window(self, unit_id: int) -> list:
        raw = self.r.get(f"engine:{unit_id}:window")
        return json.loads(raw) if raw else []

    def get_latest(self, unit_id: int) -> Optional[dict]:
        raw = self.r.get(f"engine:{unit_id}:latest")
        return json.loads(raw) if raw else None

    def get_all_active_engines(self) -> list[int]:
        """
        Scan Redis for all engine window keys and return their unit IDs.
        Used by the /trigger endpoint to run inference for all active engines.
        Pattern: engine:*:window
        """
        keys     = self.r.keys("engine:*:window")
        unit_ids = []
        for key in keys:
            try:
                # key format: engine:{unit_id}:window
                unit_ids.append(int(key.split(":")[1]))
            except (IndexError, ValueError):
                pass
        return sorted(unit_ids)


# ---------------------------------------------------------------------------
# RUL Prediction — unchanged logic from original
# ---------------------------------------------------------------------------
def infer_rul(model_store: ModelStore, window: list) -> Optional[float]:
    if len(window) < WINDOW_SIZE:
        return None

    latest   = window[-1]
    features = latest["features"]
    cycle    = latest["cycle"]
    clean    = {k: v for k, v in features.items() if k not in COLS_TO_DROP}
    df       = pd.DataFrame([{"UnitNumber": 0, "TimeInCycles": cycle, **clean}])

    try:
        df_scaled = pd.DataFrame(
            model_store.rul_scaler.transform(df),
            columns=df.columns
        )
    except Exception as e:
        log.warning(f"Scaler error — using raw features: {e}")
        df_scaled = df.copy()

    # Drop only RUL — UnitNumber and TimeInCycles must stay.
    # XGBoost was trained on the full scaled DataFrame including those columns
    # (train_RUL.py calls scaler.fit_transform(train) before the X/Y split,
    # so the booster's feature list includes UnitNumber and TimeInCycles).
    X   = df_scaled.drop(columns=["RUL"], errors="ignore")
    rul = float(model_store.rul_model.predict(X)[0])
    return max(0.0, rul)


# ---------------------------------------------------------------------------
# Anomaly Detection — unchanged logic from original
# ---------------------------------------------------------------------------
def infer_anomaly(model_store: ModelStore,
                  window: list) -> tuple[float, bool]:
    seq_len = model_store.anomaly_seq_len
    if len(window) < seq_len:
        return 0.0, False

    recent = window[-seq_len:]
    try:
        seq = np.array(
            [[list(e["features"].values())[:model_store.n_features]
              for e in recent]],
            dtype=np.float32,
        )
    except Exception as e:
        log.warning(f"Sequence build error: {e}")
        return 0.0, False

    with torch.no_grad():
        tensor = torch.tensor(seq)
        recon  = model_store.anomaly_model(tensor)
        error  = float(torch.mean((tensor - recon) ** 2).item())

    return error, error > model_store.anomaly_threshold


# ---------------------------------------------------------------------------
# Write inference result to GCS  (replaces s3.put_object)
# ---------------------------------------------------------------------------
def write_result(result: dict) -> None:
    ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    key = f"{RESULTS_PREFIX}unit_{result['engine_id']:03d}/{ts}.json"
    gcs_write_json(GCS_RAW_BUCKET, key, result)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app         = FastAPI(title="PHM Inference Orchestrator", version="1.0.0")
model_store = ModelStore()
redis_store: Optional[RedisStore] = None

# Expose /metrics endpoint for Prometheus scraping.
# Automatically tracks: request count, latency histograms, error rates
# per endpoint and HTTP method. Scraped by Prometheus every 15s.
Instrumentator().instrument(app).expose(app)


@app.on_event("startup")
async def startup():
    global redis_store
    model_store.load()
    redis_store = RedisStore()
    log.info("Inference Orchestrator ready")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class InferRequest(BaseModel):
    cycle: int


class TriggerRequest(BaseModel):
    cycle: int
    unit_ids: Optional[list[int]] = None   # if None, infer all active engines


class InferResponse(BaseModel):
    engine_id:     int
    cycle:         int
    RUL:           Optional[float]
    anomaly_score: float
    anomaly_flag:  bool
    timestamp:     str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status":    "ok",
        "loaded_at": model_store.loaded_at,
        "dataset":   DATASET,
    }


@app.get("/model/info")
def model_info():
    if not model_store.manifest:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "rul_model": {
            "name":      model_store.manifest["rul_model"]["name"],
            "version":   model_store.manifest["rul_model"]["version"],
            "mean_rmse": model_store.manifest["rul_model"]["mean_rmse"],
        },
        "anomaly_model": {
            "name":      model_store.manifest["anomaly_model"]["name"],
            "version":   model_store.manifest["anomaly_model"]["version"],
            "threshold": model_store.anomaly_threshold,
        },
        "loaded_at": model_store.loaded_at,
    }


@app.post("/infer/{unit_id}", response_model=InferResponse)
def infer(unit_id: int, req: InferRequest):
    """
    Inference Orchestrator — triggered per cycle event.
    Reads 24-cycle window from Redis written by feature-platform.
    Runs RUL Prediction + Anomaly Detection.
    Output: engine_id · RUL · anomaly_score · anomaly_flag
    """
    if model_store.rul_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    window = redis_store.get_window(unit_id)
    if not window:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No feature window for engine {unit_id}. "
                "Ensure feature-platform stream_processor has run."
            ),
        )

    rul                       = infer_rul(model_store, window)
    anomaly_score, anomaly_flag = infer_anomaly(model_store, window)

    result = {
        "engine_id":     unit_id,
        "cycle":         req.cycle,
        "RUL":           round(rul, 2) if rul is not None else None,
        "anomaly_score": round(anomaly_score, 6),
        "anomaly_flag":  anomaly_flag,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }

    try:
        write_result(result)
    except Exception as e:
        log.warning(f"GCS write failed: {e}")

    log.info(
        f"Engine {unit_id} cycle {req.cycle} — "
        f"RUL={result['RUL']} "
        f"anomaly_flag={anomaly_flag} "
        f"score={anomaly_score:.4f}"
    )
    return InferResponse(**result)


@app.post("/infer/{unit_id}/rul")
def infer_rul_only(unit_id: int, req: InferRequest):
    """RUL Prediction Model — 24-cycle sequence input."""
    window = redis_store.get_window(unit_id)
    if not window:
        raise HTTPException(status_code=404,
                            detail=f"No window for engine {unit_id}")
    rul = infer_rul(model_store, window)
    return {
        "engine_id": unit_id,
        "cycle":     req.cycle,
        "RUL":       round(rul, 2) if rul is not None else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/infer/{unit_id}/anomaly")
def infer_anomaly_only(unit_id: int, req: InferRequest):
    """Anomaly Detection Model — latest sensor => anomaly_score."""
    window = redis_store.get_window(unit_id)
    if not window:
        raise HTTPException(status_code=404,
                            detail=f"No window for engine {unit_id}")
    anomaly_score, anomaly_flag = infer_anomaly(model_store, window)
    return {
        "engine_id":     unit_id,
        "cycle":         req.cycle,
        "anomaly_score": round(anomaly_score, 6),
        "anomaly_flag":  anomaly_flag,
        "threshold":     model_store.anomaly_threshold,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }


@app.post("/trigger")
def trigger_batch(req: TriggerRequest):
    """
    Batch trigger endpoint — fixes the missing pipeline link.

    Called by stream_processor.py at the end of each CronJob run after
    writing features to Redis. Runs inference for all active engines
    (or a specified subset) in a single HTTP call.

    stream_processor adds this at the end of main():
      import requests
      requests.post(
          "http://model-serving-svc:80/trigger",
          json={"cycle": current_cycle},
          timeout=30,
      )
    """
    if model_store.rul_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Use provided unit_ids or discover all active engines from Redis
    unit_ids = req.unit_ids or redis_store.get_all_active_engines()

    if not unit_ids:
        return {"triggered": 0, "message": "No active engines in Redis"}

    results      = []
    errors       = []
    current_cycle = req.cycle

    for unit_id in unit_ids:
        try:
            window = redis_store.get_window(unit_id)
            if not window:
                continue

            rul                         = infer_rul(model_store, window)
            anomaly_score, anomaly_flag = infer_anomaly(model_store, window)

            result = {
                "engine_id":     unit_id,
                "cycle":         current_cycle,
                "RUL":           round(rul, 2) if rul is not None else None,
                "anomaly_score": round(anomaly_score, 6),
                "anomaly_flag":  bool(anomaly_flag),
                "timestamp":     datetime.now(timezone.utc).isoformat(),
            }

            try:
                write_result(result)
            except Exception as e:
                log.warning(f"GCS write failed for engine {unit_id}: {e}")

            results.append(result)

            log.info(
                f"[trigger] Engine {unit_id} cycle {current_cycle} — "
                f"RUL={result['RUL']} anomaly={anomaly_flag} "
                f"score={anomaly_score:.4f}"
            )

        except Exception as e:
            log.error(f"[trigger] Engine {unit_id} failed: {e}")
            errors.append({"unit_id": unit_id, "error": str(e)})

    return {
        "triggered":  len(results),
        "errors":     len(errors),
        "cycle":      current_cycle,
        "unit_ids":   [r["engine_id"] for r in results],
    }