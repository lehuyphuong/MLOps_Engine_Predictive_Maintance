# PHM Engine Predictive Maintenance — Schema Design and Pipeline

## 1. Goal

Build a production-grade feature store and serving layer for aircraft turbofan RUL prediction and anomaly detection, supporting both real-time online inference and batch model training.

**Approach:** Online (Redis) + Offline (PostgreSQL) dual-store architecture with rolling window feature aggregation. No traditional Gold/Silver/Bronze lakehouse — replaced by a stream-first pipeline appropriate for time-series sensor telemetry.

**Coursework requirement:** Design and implement data pipelines end-to-end, with lineage visibility through Prometheus + Loki log aggregation and Grafana pipeline health dashboards.

**Storage requirement:** Raw and validated events stored in GCS object storage (equivalent to lakehouse Bronze/Silver). Computed features stored in PostgreSQL (equivalent to Gold).

**Naming conventions:**
- GCS prefixes: `raw_engine_cycles/` (Bronze), `validated_engine_cycles/` (Silver/valid), `invalid_engine_cycles/` (Silver/quarantine)
- PostgreSQL tables: `engine_features` (offline feature store), `alerts` (alert index)
- Redis keys: `engine:{unit_id}:window` (online rolling window)

### Input Data Profile

| Attribute | Value |
|---|---|
| Source | NASA C-MAPSS FD002 via `simulated_FD002.txt` + real-time producer |
| Key identifiers | `unit_id` (int), `cycle` (int), `dataset` (string) |
| Timestamp columns | `timestamp` (event publish time), `processed_at` (feature write time) |
| Columns per event | 26: unit_id, cycle, 3 operational settings, 21 sensor measurements |
| Data volume | ~67,340 total rows; 50 rows/CronJob run; 5-minute interval |
| Data velocity | ~10 rows/minute streaming rate |
| Null/duplicate patterns | <0.2% null rate from noise injection; no intentional duplicates; checkpoint prevents re-publishing |
| Schema evolution risks | Test set has no RUL field; RUL added from separate ground truth file at training time |
| Known data issues | Multi-modal sensor distributions from 6 flight conditions; Stage 1/Stage 2 boundary creates a RUL discontinuity |

**Assumptions:**
- Business objective: minimize unscheduled maintenance by predicting engine failure 20+ cycles in advance.
- Decision usage: RUL scores surfaced to maintenance engineers via dashboard; anomaly flags trigger email alerts.
- Service level: inference results available within 10 minutes of cycle event publication.
- Explainability: out of scope for current phase.
- Risk/governance: out of scope for current phase.

**SLA targets:**

| Target | Value | Achieved |
|---|---|---|
| Raw event freshness | ≤ 5 min from simulation | ✅ CronJob every 5 min |
| Validated event freshness | ≤ 10 min from raw publish | ✅ Offset 2 min, Flink offset 3 min |
| Feature freshness (Redis) | ≤ 10 min | ✅ Flink job runs every 5 min |
| Feature freshness (PostgreSQL) | ≤ 10 min | ✅ Same Flink job |
| Inference availability | ≥ 99% | ✅ FastAPI Deployment with liveness/readiness probes |
| Alert delivery latency | ≤ 5 min after inference | ✅ Alert engine CronJob every 5 min |

---

## 2. Storage Layer Design

### 2.1 GCS Bronze — Raw Events

**Path:** `gs://phm-raw-data-aide2-494008/raw_engine_cycles/FD002/{timestamp}_{unit_id}_{cycle}.json`

**Grain:** One JSON file per engine-cycle event.

**Schema:**

```json
{
  "unit_id":   int,
  "cycle":     int,
  "dataset":   string,
  "timestamp": ISO8601,
  "sensors":   { "SensorMes1": float, ..., "SensorMes21": float,
                 "OperSet1": float, "OperSet2": float, "OperSet3": float }
}
```

**Update strategy:** Append-only. Producer checkpoint (`checkpoints/FD002/last_row.json`) ensures exactly-once semantics per run.

### 2.2 GCS Silver — Validated Events

**Valid path:** `gs://phm-raw-data-aide2-494008/validated_engine_cycles/FD002/`
**Invalid path:** `gs://phm-data-quality-aide2-494008/invalid_engine_cycles/FD002/`

**Routing logic:** Four Great Expectations suites (schema, null, range, order). All four must pass for routing to valid; any failure routes to quarantine with a failure report attached.

### 2.3 Redis Online Feature Store

**Key format:** `engine:{unit_id}:window`
**Value:** JSON list of the last 30 cycle feature dictionaries (`WINDOW_SIZE=30`), ordered oldest-to-newest.
**TTL:** None (rolling overwrite on each Flink run).
**Purpose:** Serves the 30-cycle sequence input required by the RUL XGBoost model at inference time.

### 2.4 PostgreSQL Offline Feature Store — `engine_features`

**Database:** `phmdb` (Cloud SQL PostgreSQL 15, `db-g1-small`, 20 GB SSD).

```sql
CREATE TABLE engine_features (
    id           SERIAL PRIMARY KEY,
    unit_id      INTEGER NOT NULL,
    cycle        INTEGER NOT NULL,
    dataset      VARCHAR(10) NOT NULL,
    rul          FLOAT,
    rul_capped   FLOAT,
    features     JSONB NOT NULL,
    window_size  INTEGER,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source_key   TEXT,
    UNIQUE (unit_id, cycle, dataset)
);
CREATE INDEX idx_engine_features_unit_cycle ON engine_features (unit_id, cycle);
CREATE INDEX idx_engine_features_dataset    ON engine_features (dataset);
```

**Grain:** One row per (unit_id, cycle, dataset) combination.
**Purpose:** Training data source for both XGBoost RUL model (offline mode) and LSTM AutoEncoder (normal pool extraction).
**Update strategy:** Upsert on (unit_id, cycle, dataset) unique key — idempotent re-runs are safe.

### 2.5 PostgreSQL Alert Index — `alerts`

```sql
CREATE TABLE alerts (
    id           SERIAL PRIMARY KEY,
    engine_id    INTEGER NOT NULL,
    dataset      VARCHAR(10) NOT NULL,
    alert_type   VARCHAR(50) NOT NULL,   -- rul_threshold | anomaly_detected
    severity     VARCHAR(20) NOT NULL,   -- critical | warning
    rul          FLOAT,
    anomaly_score FLOAT,
    anomaly_flag BOOLEAN,
    rule_triggered VARCHAR(100),
    timestamp    TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Grain:** One row per alert event (multiple alerts can exist for the same engine at different times).
**Purpose:** Grafana dashboard source for Fleet Status and Anomaly Detection panels; email notification trigger.

---

## 3. Feature Table Design

### 3.1 Online Feature Window (Redis)

**Grain:** Engine-level, latest 30 cycles.
**Refresh:** Rolling overwrite every 5 minutes.
**Features per cycle:** 14 sensor columns selected by Pearson/Spearman/Kendall correlation with RUL (threshold 0.03 for FD002 multi-condition data):

`OperSet3, SensorMes2, SensorMes3, SensorMes4, SensorMes6, SensorMes7, SensorMes8, SensorMes9, SensorMes11, SensorMes12, SensorMes13, SensorMes14, SensorMes15, SensorMes16, SensorMes17, SensorMes20, SensorMes21`

### 3.2 Offline Feature Table (PostgreSQL)

**Grain:** (unit_id, cycle, dataset) — one row per engine-cycle.
**Features:** Same 14 columns stored as JSONB in the `features` column.
**Point-in-time correctness:** The `processed_at` timestamp allows filtering features available before any given reference time, preventing data leakage in training.
**Dedup:** UNIQUE constraint on (unit_id, cycle, dataset) with ON CONFLICT DO UPDATE semantics.

---

## 4. Pipeline Design

### 4.1 Pipeline Group 1 — Data Ingestion (data-ingestion namespace)

**Producer CronJob** (every 5 minutes):
1. Read GCS checkpoint → resume from last published row index.
2. Read next 50 rows from `simulated_FD002.txt`.
3. Publish each row as a JSON event to `raw_engine_cycles/FD002/`.
4. Update checkpoint.

**Validation CronJob** (every 5 minutes, offset +2 min):
1. List unprocessed keys in `raw_engine_cycles/FD002/` (GCS checkpoint).
2. Apply four GE suites sequentially.
3. Route valid → `validated_engine_cycles/`, invalid → `invalid_engine_cycles/` with failure report.
4. Update validation checkpoint.

**Update strategy:** Append-only to Bronze; idempotent routing to Silver via processed-key checkpoint.
**Backfill policy:** Reset checkpoint file to replay from any row index; no time limit.
**Late data handling:** Not applicable — producer is the sole source and is sequential.

### 4.2 Pipeline Group 2 — Feature Platform (feature-platform namespace)

**Stream Processor CronJob** (every 5 minutes, offset +3 min):
1. List unprocessed validated events (checkpoint `feature_processed_keys.json`).
2. Cap batch at 50 events to prevent OOM.
3. Write batch to `/tmp/stream_batch.json`.
4. Submit PyFlink job to remote Flink cluster via `flink run -m flink-jobmanager:6123`.

**PyFlink Job** (`phm-feature-pipeline`):
1. Read batch from collection source.
2. `FeatureMapFunction.open()`: connect to Redis and PostgreSQL.
3. For each event: extract 14 feature columns, update Redis rolling window (RPUSH/LTRIM to 30), upsert row to `engine_features`.
4. Job completes in FINISHED state (appears in Flink Web UI).

**Infrastructure:** Flink 1.18.1 standalone cluster — JobManager (Deployment) + TaskManager (StatefulSet, 1 replica). TaskManager uses headless Service (`flink-taskmanager-hl`) + ordinal DNS (`flink-taskmanager-0.flink-taskmanager-hl.feature-platform.svc.cluster.local`) to resolve the UnknownHostException that occurs with Deployment pods.

**Update strategy:** Idempotent RPUSH+LTRIM for Redis; ON CONFLICT DO UPDATE for PostgreSQL. Re-runs are safe.

**Quality gates per run:**
- Schema check: all expected feature columns present in event.
- Connection check: Redis and PostgreSQL connectivity verified in `open()`.
- Flink job completion: CronJob waits for FINISHED state; failure raises exception.

**Evidence:** Flink Web UI screenshot (`assets/imgs/Flink_UI.png`) showing job `phm-feature-pipeline` in FINISHED state, duration 7s, 1 task, parallelism 1.

### 4.3 Pipeline Group 3 — Model Training (model-training namespace)

Orchestrated by **Apache Airflow 2.9.2** (LocalExecutor, DAG: `model_training_pipeline`), scheduled daily at 02:00 UTC and triggerable manually.

**DAG topology:**

```
train_rul       ──┐
                  ├──► validate ──► promote
train_anomaly   ──┘
```

Each task runs as a `KubernetesPodOperator` pod with `service_account_name=model-training-sa` (Workload Identity for GCS access).

**Task: train_rul** (`train_RUL.py`):
1. Load `offline/train_FD002.txt` + `test_FD002.txt` + `RUL_FD002.txt` from GCS.
2. Generate RUL labels, cap at 125, drop 8 low-correlation columns.
3. MinMaxScaler normalization (fit on train, transform test separately).
4. 10-run loop: random sample selection (50 train / 25 test per engine per run) + XGBoost GridSearchCV (5-fold CV).
5. Compute MSE, RMSE, MAE, MAPE, S-score per run.
6. Log params + metrics to MLflow (`mlflowdb` database). Save best model to GCS.
7. **Result:** mean RMSE = 0.2721 (normalized), mean S-score logged.

**Task: train_anomaly** (`train_anomaly.py`):
1. Load training data from GCS.
2. Extract normal pool: engines with `RUL > 150` (pre-failure healthy baseline).
3. Build 30-cycle sliding windows per engine.
4. Train PyTorch LSTM AutoEncoder: `n_features → hidden1=64 → hidden2=32 → latent=16 → hidden2=32 → hidden1=64 → n_features` (50 epochs, batch=32, lr=0.001, Adam).
5. Compute reconstruction errors on normal pool. Threshold = `mean + 3×std` of normal pool errors.
6. Log to MLflow (`phm-anomaly-autoencoder-FD002` experiment). Save model + threshold to GCS.
7. **Result:** threshold = 0.117615; detection F1 logged.

**Task: validate** (`validate.py`):
1. **RMSE gate:** Read latest finished MLflow run in `phm-rul-xgboost-FD002-offline`. Fail if `mean_rmse ≥ 0.30`.
2. **Drift check:** Compare current training data feature means against previous production baseline (stored as `feature_means.json` MLflow artifact). Fail if max absolute mean shift > 0.05. Skip on first run (no baseline).
3. If both pass: tag run as `production_baseline=true`, upload `feature_means.json` artifact.
4. Exit 0 = PASSED (promote runs); Exit 1 = FAILED (promote skipped).

**Task: promote** (`promote.py`):
1. Fetch latest finished runs from both MLflow experiments.
2. Re-run RMSE gate + drift check (defensive double-check).
3. Register both models in MLflow Model Registry: `None → Staging → Production`.
4. Write promotion manifest to `gs://phm-model-artifacts-aide2-494008/registry/FD002/promotion_manifest.json`.
5. Manifest consumed by model-serving at startup for artifact loading.

**Evidence:** Airflow UI screenshot (`assets/imgs/Airfow_UI.png`) showing all four tasks green (success) in the latest DAG run. MLflow UI (`assets/imgs/MLflow_UI.png`) showing 3 experiments with multiple successful runs.

### 4.4 Pipeline Group 4 — Model Serving (model-serving namespace)

**FastAPI Deployment** (`serve.py`), always-running, triggered per cycle event:

| Endpoint | Input | Output |
|---|---|---|
| `POST /infer/{unit_id}` | `{cycle: int}` | `{rul: float, anomaly_score: float, anomaly_flag: bool}` |
| `POST /infer/{unit_id}/rul` | `{cycle: int}` | `{rul: float}` |
| `POST /infer/{unit_id}/anomaly` | `{cycle: int}` | `{anomaly_score: float, anomaly_flag: bool}` |
| `POST /trigger` | `{engine_ids: list}` | Batch inference for all active engines |
| `GET /health` | — | Liveness probe |

**Inference flow:**
1. Load model artifacts from GCS at startup (path from promotion manifest).
2. On `/trigger`: fetch Redis window for each engine → run XGBoost on 30-cycle sequence → run LSTM AE on latest single reading → write result JSON to `gs://phm-raw-data.../inference-results/FD002/`.
3. Prometheus metrics exported at `/metrics` for Grafana ingestion.

### 4.5 Pipeline Group 5 — Alert Engine (alert-naming namespace)

**Alert Engine CronJob** (every 5 minutes):
1. List unprocessed inference result files from GCS (checkpoint-based).
2. Evaluate Rule 1: `RUL < 20` → severity `warning`; `RUL < 10` → severity `critical`.
3. Evaluate Rule 2: `anomaly_flag == True` → severity `warning`.
4. Write alert row to PostgreSQL `alerts` table.
5. If `NOTIFY_ENABLED=true` and severity matches `NOTIFY_SEVERITY`: send email via SendGrid API.

**Evidence:** Email screenshot (`assets/imgs/mail_alert.png`) showing CRITICAL alert for Engine 999, RUL=0.41 cycles, anomaly_flag=True, delivered via SendGrid.

---

## 5. Refresh and Data Quality

### 5.1 Quality Gates Per Pipeline

| Pipeline | Gate | Action on failure |
|---|---|---|
| Producer | GCS write error | CronJob exits non-zero → Kubernetes restarts |
| Validation | GE suite failure | Route to quarantine bucket; continue processing others |
| Flink | PostgreSQL/Redis connection | Job fails FINISHED→FAILED; next CronJob retry |
| Training | RMSE > threshold | `validate` task exits 1; promote skipped |
| Serving | Model load failure | Pod restart; liveness probe triggers restart |
| Alert | PostgreSQL write failure | CronJob logs error; retry on next run |

### 5.2 Monitoring and Alerting

Prometheus scrapes FastAPI `/metrics` every 30 seconds. Loki + Promtail DaemonSet collects logs from all namespaces. Grafana (monitoring namespace, ingress-nginx-dev, basic auth) provides:

| Panel | Source | Alert condition |
|---|---|---|
| Pod Restarts | Prometheus | Any restart in last 6 hours |
| FastAPI Request Rate | Prometheus | < 0.01 req/min (inference stopped) |
| FastAPI Inference Latency p95 | Prometheus | > 200ms |
| Memory/CPU per namespace | Prometheus | > 90% of limit |
| Log Stream — Stream Processor | Loki | ERROR keyword |
| Log Stream — Model Serving | Loki | ERROR keyword |
| Log Stream — Alert Engine | Loki | ERROR keyword |
| Log Stream — Validation Errors | Loki | WARNING INVALID keyword |

**Evidence:** Monitoring UI screenshot (`assets/imgs/monitoring_UI.png`) showing all 8 panels active with real data.

### 5.3 Run Metadata

Each pipeline records:
- **Producer:** `last_row_index`, `rows_emitted`, `timestamp` in checkpoint JSON.
- **Validation:** `validated_keys` list in checkpoint JSON; failure reports per invalid event.
- **Flink:** Flink Web UI job history with start/end time, duration, state.
- **Airflow:** Full DAG run history in PostgreSQL (`phmdb/airflow` schema) with task state, start/end, log access.
- **MLflow:** Full experiment run history with params, metrics, artifact paths.

---

## 6. Warehouse Optimization

### 6.1 PostgreSQL Index Optimization

**Workload:** Model training reads all rows for a specific dataset from `engine_features` ordered by (unit_id, cycle). Alert dashboard queries `alerts` filtered by `alert_type` and time range.

**Optimizations applied:**

| Table | Optimization | Rationale |
|---|---|---|
| `engine_features` | Composite index on `(unit_id, cycle)` | Training data sort eliminates full-table scan |
| `engine_features` | Index on `dataset` | Multi-dataset support; filters 100% of rows by dataset |
| `engine_features` | UNIQUE on `(unit_id, cycle, dataset)` | Enforces idempotent upserts; doubles as covering index |
| `alerts` | Default SERIAL PK index | Dashboard queries ordered by `created_at DESC` |
| Airflow tables | Isolated in `airflow` schema | Prevents Alembic version conflict with MLflow tables in `public` |
| MLflow tables | Isolated in `mlflowdb` database | Separate database eliminates `alembic_version` collision entirely |

**Result:** Training data load for 260 engines × 258 cycles ≈ 67,000 rows completes in < 2 seconds with index scan vs ~8 seconds full table scan.

**Trade-off:** Two additional indexes increase write cost by ~15% per Flink upsert cycle. Acceptable given the 5-minute write interval.

### 6.2 Redis Key Design

**Key pattern:** `engine:{unit_id}:window` → JSON list (RPUSH/LTRIM to 30 elements).

This flat key design avoids nested hash lookups. Inference reads the full 30-element window in a single `LRANGE engine:{unit_id}:window 0 -1` call. With 260 engines × ~2 KB per window, total Redis footprint is ~520 KB — well within the 1 GB Memorystore allocation.

---

## 7. Deliverables

1. **Schema definitions:** `engine_features` and `alerts` tables created by `flink_job.py` `ensure_schema()` at startup.
2. **Pipeline code:** All five pipeline groups implemented in `services/` directory.
3. **Helm charts:** Each pipeline deployed as a Kubernetes workload in `charts/`.
4. **Monitoring evidence:** `assets/imgs/monitoring_UI.png` — all 8 Grafana panels active.
5. **Flink evidence:** `assets/imgs/Flink_UI.png` — `phm-feature-pipeline` FINISHED.
6. **MLflow evidence:** `assets/imgs/MLflow_UI.png` — 3 experiments, multiple successful runs.
7. **Airflow evidence:** `assets/imgs/Airfow_UI.png` — all 4 DAG tasks green.
8. **Alert evidence:** `assets/imgs/mail_alert.png` — critical email alert delivered.
