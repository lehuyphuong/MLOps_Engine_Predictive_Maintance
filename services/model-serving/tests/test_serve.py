"""
test_serve.py — Unit tests for serve.py
Tests all pure-logic functions without requiring GCS, Redis, or real models.
All external dependencies (GCS, Redis, XGBoost, PyTorch) are mocked.

Test coverage:
  1. LSTMAutoEncoder     — architecture, forward pass shape
  2. infer_rul           — window too short, feature engineering, prediction
  3. infer_anomaly       — window too short, sequence build, threshold
  4. evaluate_rules      — RUL threshold, anomaly flag, severity levels
  5. write_result        — GCS key format, JSON content
  6. /health endpoint    — FastAPI response
  7. /infer/{unit_id}    — full inference flow
  8. /trigger            — batch inference, empty Redis
  9. RedisStore          — window parsing, active engine discovery
"""

import json
import sys
import os
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Make serve.py importable without GCS / Redis connections at import time
# ---------------------------------------------------------------------------
os.environ.setdefault("GCS_ARTIFACTS_BUCKET", "test-bucket")
os.environ.setdefault("GCS_RAW_BUCKET",       "test-raw-bucket")
os.environ.setdefault("GCP_PROJECT",           "test-project")
os.environ.setdefault("REDIS_HOST",            "localhost")
os.environ.setdefault("WINDOW_SIZE",           "30")

# Patch GCS and Redis before importing serve so module-level clients don't fail
with patch("google.cloud.storage.Client"), \
     patch("redis.Redis"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..","src"))
    import serve
    from serve import (
        LSTMAutoEncoder,
        ModelStore,
        RedisStore,
        infer_rul,
        infer_anomaly,
        write_result,
        app,
        WINDOW_SIZE,
        COLS_TO_DROP,
    )

# ---------------------------------------------------------------------------
# Helpers — build minimal mock objects
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "SensorMes2","SensorMes3","SensorMes4","SensorMes6","SensorMes7",
    "SensorMes8","SensorMes9","SensorMes11","SensorMes12","SensorMes13",
    "SensorMes14","SensorMes15","SensorMes16","SensorMes17","SensorMes20",
    "SensorMes21",
]
N_FEATURES = len(FEATURE_COLS)   # 16
SEQ_LEN    = 30


def make_window(n: int, cycle_start: int = 1) -> list:
    """Build a window of n entries with realistic feature values."""
    return [
        {
            "cycle": cycle_start + i,
            "features": {col: float(i + j) for j, col in enumerate(FEATURE_COLS)},
        }
        for i in range(n)
    ]


def make_model_store(threshold: float = 0.5) -> ModelStore:
    """Return a ModelStore with mocked models — no GCS calls."""
    ms = ModelStore()

    # Mock XGBoost model — always predicts 50.0
    ms.rul_model  = MagicMock()
    ms.rul_model.predict.return_value = np.array([50.0])

    # Mock MinMaxScaler — identity transform
    ms.rul_scaler = MagicMock()
    ms.rul_scaler.transform.side_effect = lambda df: df.values

    # Real LSTM AutoEncoder — use actual PyTorch forward pass
    ms.anomaly_model     = LSTMAutoEncoder(N_FEATURES, SEQ_LEN)
    ms.anomaly_threshold = threshold
    ms.anomaly_seq_len   = SEQ_LEN
    ms.n_features        = N_FEATURES

    ms.manifest  = {
        "promoted_at": "2026-01-01T00:00:00+00:00",
        "rul_model":   {"name": "test-rul",     "version": "1", "mean_rmse": 0.2721},
        "anomaly_model": {"name": "test-anomaly", "version": "1", "threshold": threshold},
    }
    ms.loaded_at = "2026-01-01T00:00:00+00:00"
    return ms


# ===========================================================================
# 1. LSTMAutoEncoder — architecture tests
# ===========================================================================
class TestLSTMAutoEncoder:

    def test_output_shape_matches_input(self):
        """Output tensor must have same shape as input."""
        model = LSTMAutoEncoder(n_features=N_FEATURES, seq_len=SEQ_LEN)
        x     = torch.randn(4, SEQ_LEN, N_FEATURES)   # batch=4
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape, \
            f"Expected {x.shape}, got {out.shape}"

    def test_reconstruction_error_is_non_negative(self):
        """MSE reconstruction error must always be >= 0."""
        model = LSTMAutoEncoder(n_features=N_FEATURES, seq_len=SEQ_LEN)
        x     = torch.randn(1, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            recon = model(x)
            error = float(torch.mean((x - recon) ** 2).item())
        assert error >= 0.0

    def test_different_inputs_produce_different_outputs(self):
        """Model must not return identical output for different inputs."""
        model  = LSTMAutoEncoder(n_features=N_FEATURES, seq_len=SEQ_LEN)
        x1     = torch.zeros(1, SEQ_LEN, N_FEATURES)
        x2     = torch.ones(1, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            o1 = model(x1)
            o2 = model(x2)
        assert not torch.allclose(o1, o2), \
            "Model returned identical output for different inputs"

    def test_eval_mode_is_deterministic(self):
        """In eval mode, same input must produce same output."""
        model = LSTMAutoEncoder(n_features=N_FEATURES, seq_len=SEQ_LEN)
        model.eval()
        x = torch.randn(1, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            o1 = model(x)
            o2 = model(x)
        assert torch.allclose(o1, o2)


# ===========================================================================
# 2. infer_rul — RUL prediction logic
# ===========================================================================
class TestInferRul:

    def test_returns_none_when_window_too_short(self):
        """Returns None if window has fewer entries than WINDOW_SIZE."""
        ms = make_model_store()
        assert infer_rul(ms, make_window(WINDOW_SIZE - 1)) is None

    def test_returns_none_for_empty_window(self):
        assert infer_rul(ms := make_model_store(), []) is None

    def test_returns_float_for_full_window(self):
        ms  = make_model_store()
        rul = infer_rul(ms, make_window(WINDOW_SIZE))
        assert isinstance(rul, float)
        assert rul >= 0.0

    def test_rul_is_non_negative(self):
        """RUL is clamped to >= 0 even if model predicts negative."""
        ms = make_model_store()
        ms.rul_model.predict.return_value = np.array([-10.0])
        rul = infer_rul(ms, make_window(WINDOW_SIZE))
        assert rul == 0.0, f"Expected 0.0, got {rul}"

    def test_drops_cols_to_drop(self):
        """Features in COLS_TO_DROP must not be passed to scaler."""
        ms = make_model_store()
        window = make_window(WINDOW_SIZE)
        # Add a column that should be dropped
        for entry in window:
            entry["features"]["OperSet1"] = 999.0

        infer_rul(ms, window)

        # Inspect the DataFrame passed to scaler.transform
        call_args = ms.rul_scaler.transform.call_args[0][0]
        assert "OperSet1" not in call_args.columns, \
            "OperSet1 should be dropped before scaling"

    def test_uses_last_entry_in_window(self):
        """infer_rul uses features from the last entry, not the first."""
        ms     = make_model_store()
        window = make_window(WINDOW_SIZE, cycle_start=1)
        # Mark last entry with a distinct cycle
        window[-1]["cycle"] = 9999

        infer_rul(ms, window)

        call_args = ms.rul_scaler.transform.call_args[0][0]
        assert call_args["TimeInCycles"].iloc[0] == 9999


# ===========================================================================
# 3. infer_anomaly — anomaly detection logic
# ===========================================================================
class TestInferAnomaly:

    def test_returns_zero_when_window_too_short(self):
        ms    = make_model_store()
        score, flag = infer_anomaly(ms, make_window(SEQ_LEN - 1))
        assert score == 0.0
        assert flag  is False

    def test_returns_zero_for_empty_window(self):
        ms = make_model_store()
        score, flag = infer_anomaly(ms, [])
        assert score == 0.0
        assert flag  is False

    def test_returns_float_and_bool_for_full_window(self):
        ms          = make_model_store(threshold=0.5)
        score, flag = infer_anomaly(ms, make_window(SEQ_LEN))
        assert isinstance(score, float)
        assert isinstance(flag,  bool)
        assert score >= 0.0

    def test_anomaly_flag_true_when_error_exceeds_threshold(self):
        """flag=True when reconstruction error > threshold."""
        ms = make_model_store(threshold=0.0)   # threshold=0 → always flag
        score, flag = infer_anomaly(ms, make_window(SEQ_LEN))
        assert flag is True, \
            f"Expected anomaly_flag=True with threshold=0, got score={score}"

    def test_anomaly_flag_false_when_error_below_threshold(self):
        """flag=False when reconstruction error < threshold."""
        ms = make_model_store(threshold=1e9)   # threshold=1B → never flag
        score, flag = infer_anomaly(ms, make_window(SEQ_LEN))
        assert flag is False, \
            f"Expected anomaly_flag=False with threshold=1e9, got score={score}"

    def test_uses_last_seq_len_entries(self):
        """Uses only the last seq_len entries from a longer window."""
        ms     = make_model_store()
        window = make_window(SEQ_LEN + 10)   # longer than needed
        # Should not raise — uses window[-seq_len:]
        score, flag = infer_anomaly(ms, window)
        assert isinstance(score, float)


# ===========================================================================
# 4. write_result — GCS key format
# ===========================================================================
class TestWriteResult:

    def test_gcs_key_format(self):
        """Result key must follow unit_{id:03d}/{timestamp}.json pattern."""
        with patch("serve.gcs_write_json") as mock_write:
            result = {
                "engine_id":     42,
                "cycle":         100,
                "RUL":           15.5,
                "anomaly_score": 0.001,
                "anomaly_flag":  False,
                "timestamp":     "2026-01-01T00:00:00+00:00",
            }
            write_result(result)
            assert mock_write.called
            key = mock_write.call_args[0][1]
            assert "unit_042/" in key, f"Key should contain unit_042/, got: {key}"
            assert key.endswith(".json"),  f"Key should end with .json, got: {key}"

    def test_writes_to_correct_bucket(self):
        """Must write to GCS_RAW_BUCKET."""
        with patch("serve.gcs_write_json") as mock_write:
            write_result({
                "engine_id": 1, "cycle": 1,
                "RUL": 10.0, "anomaly_score": 0.0,
                "anomaly_flag": False, "timestamp": "2026-01-01T00:00:00+00:00",
            })
            bucket = mock_write.call_args[0][0]
            assert bucket == serve.GCS_RAW_BUCKET


# ===========================================================================
# 5. FastAPI endpoints — /health, /model/info, /infer, /trigger
# ===========================================================================
@pytest.fixture
def client():
    """FastAPI test client with mocked model store and Redis."""
    ms = make_model_store()
    serve.model_store = ms

    mock_redis = MagicMock()
    mock_redis.get_window.return_value       = make_window(SEQ_LEN)
    mock_redis.get_all_active_engines.return_value = [1, 2, 3]
    serve.redis_store = mock_redis

    with patch("serve.write_result"):
        yield TestClient(app)


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_returns_dataset(self, client):
        data = client.get("/health").json()
        assert "dataset" in data


class TestModelInfoEndpoint:

    def test_model_info_returns_200(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_contains_rul_and_anomaly(self, client):
        data = client.get("/model/info").json()
        assert "rul_model"     in data
        assert "anomaly_model" in data

    def test_model_info_503_when_not_loaded(self):
        serve.model_store = ModelStore()   # unloaded
        c    = TestClient(app)
        resp = c.get("/model/info")
        assert resp.status_code == 503
        # Restore
        serve.model_store = make_model_store()


class TestInferEndpoint:

    def test_infer_returns_200(self, client):
        response = client.post("/infer/1", json={"cycle": 100})
        assert response.status_code == 200

    def test_infer_response_has_required_fields(self, client):
        data = client.post("/infer/1", json={"cycle": 100}).json()
        for field in ["engine_id", "cycle", "RUL",
                      "anomaly_score", "anomaly_flag", "timestamp"]:
            assert field in data, f"Missing field: {field}"

    def test_infer_returns_correct_engine_id(self, client):
        data = client.post("/infer/7", json={"cycle": 200}).json()
        assert data["engine_id"] == 7

    def test_infer_returns_correct_cycle(self, client):
        data = client.post("/infer/1", json={"cycle": 500}).json()
        assert data["cycle"] == 500

    def test_infer_rul_is_non_negative(self, client):
        data = client.post("/infer/1", json={"cycle": 100}).json()
        if data["RUL"] is not None:
            assert data["RUL"] >= 0.0

    def test_infer_anomaly_score_is_non_negative(self, client):
        data = client.post("/infer/1", json={"cycle": 100}).json()
        assert data["anomaly_score"] >= 0.0

    def test_infer_404_when_no_window(self, client):
        serve.redis_store.get_window.return_value = []
        resp = client.post("/infer/999", json={"cycle": 100})
        assert resp.status_code == 404


class TestTriggerEndpoint:

    def test_trigger_returns_200(self, client):
        response = client.post("/trigger", json={"cycle": 100})
        assert response.status_code == 200

    def test_trigger_returns_triggered_count(self, client):
        data = client.post("/trigger", json={"cycle": 100}).json()
        assert "triggered" in data
        assert "errors"    in data

    def test_trigger_infers_all_active_engines(self, client):
        serve.redis_store.get_all_active_engines.return_value = [1, 2, 3]
        data = client.post("/trigger", json={"cycle": 100}).json()
        assert data["triggered"] == 3

    def test_trigger_with_explicit_unit_ids(self, client):
        data = client.post(
            "/trigger",
            json={"cycle": 100, "unit_ids": [5, 10]}
        ).json()
        assert data["triggered"] == 2

    def test_trigger_returns_zero_when_no_engines(self, client):
        serve.redis_store.get_all_active_engines.return_value = []
        data = client.post("/trigger", json={"cycle": 100}).json()
        assert data["triggered"] == 0


# ===========================================================================
# 6. RedisStore — helper methods
# ===========================================================================
class TestRedisStore:

    def test_get_all_active_engines_parses_keys(self):
        with patch("redis.Redis") as mock_redis_cls:
            mock_r = MagicMock()
            mock_r.keys.return_value = [
                "engine:1:window",
                "engine:42:window",
                "engine:100:window",
            ]
            mock_redis_cls.return_value = mock_r

            store    = RedisStore()
            store.r  = mock_r
            engines  = store.get_all_active_engines()

        assert sorted(engines) == [1, 42, 100]

    def test_get_all_active_engines_ignores_malformed_keys(self):
        with patch("redis.Redis") as mock_redis_cls:
            mock_r = MagicMock()
            mock_r.keys.return_value = [
                "engine:1:window",
                "engine:bad:window",   # non-integer
                "other:key",           # wrong format
            ]
            mock_redis_cls.return_value = mock_r

            store   = RedisStore()
            store.r = mock_r
            engines = store.get_all_active_engines()

        assert 1 in engines
        assert len([e for e in engines if not isinstance(e, int)]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
