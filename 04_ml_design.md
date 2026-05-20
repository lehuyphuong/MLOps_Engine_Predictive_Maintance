# PHM Engine Predictive Maintenance — ML System Design

## 1. Goal

Build a two-model ML system for real-time aircraft turbofan engine health monitoring using the NASA C-MAPSS FD002 dataset. The system predicts Remaining Useful Life (RUL) and detects early-stage anomalies to enable proactive maintenance scheduling and prevent unscheduled engine removals.

**Why this task matters:** Unscheduled engine maintenance costs airlines $500K–$2M per incident. Accurate RUL prediction with 20+ cycle lead time allows maintenance to be planned during scheduled ground time, avoiding flight cancellations. Anomaly detection provides an additional early-warning signal before measurable RUL degradation is visible in sensor trends.

---

## 2. Prediction Setup and Modeling Plan

### 2.1 Model 1 — RUL Prediction (XGBoost Regressor)

**Entity:** Engine unit (`unit_id`)
**Target:** `RUL` — number of operational cycles remaining before failure
**Prediction time:** Per cycle, using the 30-cycle rolling feature window from Redis
**Label:** Computed from C-MAPSS ground truth: `max_cycle − current_cycle` (train), provided separately (test), capped at 125 cycles to reduce influence of early-life high-RUL values

**Training data:**
- Source: `gs://phm-raw-data-aide2-494008/offline/train_FD002.txt` (260 engines, ~54,000 rows)
- Feature selection: 14 columns retained after Pearson/Spearman/Kendall correlation analysis (threshold 0.03 for FD002 multi-condition data); 8 columns dropped
- Normalization: MinMaxScaler per run (fit on train sample, transform test sample independently)
- Sampling: 50 rows/engine for train, 25 rows/engine for test, random state varied across 10 runs

**Split strategy:** Random sample selection per engine across 10 independent runs. No time-based split for XGBoost because the model operates on per-cycle features rather than sequences. Each run uses a different random seed to estimate performance stability.

**Baseline model:** XGBoost Regressor with GridSearchCV (5-fold CV, neg MSE scoring). Hyperparameters: `n_estimators=100, max_depth=3, learning_rate=0.01, subsample=0.5, colsample_bytree=0.5`.

**Metrics:**

| Metric | Value | Acceptance threshold |
|---|---|---|
| Mean RMSE (normalized) | 0.2721 | < 0.30 (RMSE gate) |
| Mean MAE | Logged to MLflow | — |
| Mean MAPE | Logged to MLflow | — |
| Asymmetric S-score | Logged to MLflow | Lower is better (late predictions penalized 10× more than early) |

The S-score penalizes late predictions more heavily than early ones (`a₁=10, a₂=13`), matching the operational preference for conservative (early) RUL estimates over optimistic ones.

### 2.2 Model 2 — Anomaly Detection (LSTM AutoEncoder)

**Entity:** Engine unit (`unit_id`)
**Target:** Binary anomaly flag (`anomaly_flag = reconstruction_error > threshold`)
**Prediction time:** Per cycle, using the latest single-reading sensor vector

**Training data:**
- Normal pool: engines with `RUL > 150` from `train_FD002.txt` — the pre-failure healthy baseline (~70% of training data)
- Window construction: 30-cycle sliding windows per engine in the normal pool
- No split needed: AutoEncoder trained exclusively on normal data; anomaly = out-of-distribution reconstruction

**Architecture:** PyTorch LSTM AutoEncoder:
```
Encoder: n_features(14) => LSTM(64) => LSTM(32) => Linear(16)
Decoder: Linear(16) => LSTM(32) => LSTM(64) => Linear(n_features)
```

**Training:** 50 epochs, batch size 32, Adam optimizer (lr=0.001), MSE reconstruction loss.

**Threshold:** `mean + 3×std` of reconstruction errors on the normal pool. Result: **0.117615**.

**Metrics:**

| Metric | Value |
|---|---|
| Reconstruction threshold | 0.117615 |
| Normal pool error mean | Logged to MLflow |
| Normal pool error std | Logged to MLflow |

**Split strategy:** Train only on `RUL > 150` normal pool. No explicit test split for training — anomaly performance evaluated implicitly by monitoring false positive rate on normal pool cycles during inference.

**Assumptions:**
- Business objective: surface anomaly flag to maintenance engineers as an early warning complement to RUL prediction. Not used as a hard automated decision gate.
- Decision usage: RUL is the primary decision metric; anomaly flag is secondary confirmation.
- Service level: inference results available within 10 minutes of cycle publication; p95 latency < 200ms per `/infer` call.
- Explainability: out of scope for this phase.
- Risk/governance: out of scope for this phase.

**SLA targets:**

| Target | Value | Achieved |
|---|---|---|
| Online inference latency p95 | < 200ms | OK (Prometheus metrics confirm) |
| Feature freshness | < 10 min | OK (Flink CronJob every 5 min) |
| Model artifact load time | < 30s | OK (GCS download at startup) |
| Alert delivery | < 5 min after inference | OK (alert-engine CronJob every 5 min) |

---

## 3. High-Level ML Design

**End-to-end flow:**
1. **Data ingestion:** Telemetry producer publishes 50 engine cycles every 5 minutes to GCS raw bucket.
2. **Validation:** Great Expectations validates schema/null/range/order; routes valid => Silver bucket.
3. **Feature engineering:** PyFlink stream processor computes 30-cycle rolling windows => Redis (online) + PostgreSQL (offline).
4. **Training pipeline:** Airflow DAG triggers daily: `train_rul + train_anomaly (parallel) => validate => promote`.
5. **Model registry:** MLflow registers approved models as Production; writes promotion manifest to GCS.
6. **Serving:** FastAPI inference orchestrator loads artifacts from GCS; `/trigger` endpoint runs batch inference for all active engines after each Flink batch.
7. **Alerting:** Alert engine CronJob evaluates RUL < 20 and anomaly_flag == True; writes to PostgreSQL alerts table; sends email via SendGrid.
8. **Monitoring:** Prometheus + Loki + Grafana monitors all eight namespaces.
9. **CI/CD:** Jenkins pipeline automates image build + deployment of model-serving namespace, gated by 70% unit test coverage.
10. **Retraining:** Daily scheduled Airflow DAG run with drift-triggered failure gate.

### 3.1 Security Decision

**IAM and RBAC:**
- Each Kubernetes namespace has a dedicated Google Service Account (GSA) via Workload Identity. No static credentials in pods.
- GSA permissions follow least-privilege: `data-ingestion-sa` => raw/quality bucket objectAdmin; `model-serving-sa` => model-artifacts objectViewer + raw objectAdmin (inference results write); `model-training-sa` => raw objectViewer + model-artifacts/mlflow objectAdmin.
- Airflow admin password stored in Kubernetes Secret (`airflow-secrets`). Database password generated by Terraform (`random_password.db`, 16 chars, special characters URL-encoded to prevent injection).
- MLflow metadata stored in dedicated `mlflowdb` PostgreSQL database, isolated from Airflow metadata in `phmdb/airflow` schema.

**Secrets management:**
- All sensitive values (DB password, SendGrid API key) stored as Kubernetes Secrets, never in ConfigMaps or image layers.
- Terraform state stored in GCS with uniform bucket-level access enforcement.

**Encryption:**
- GCS buckets: `public_access_prevention = enforced`, `uniform_bucket_level_access = true`.
- Cloud SQL: private IP only (`ipv4_enabled = false`), VPC peering via `servicenetworking.googleapis.com`.
- In-transit: HTTPS for GCS/API calls; internal cluster traffic via VPC (no SSL between pods required for dev).

### 3.2 Resilience Decision

**Retry and backoff:**
- Airflow tasks: `retries=1`, `retry_delay=5 min`, `execution_timeout=3 hours`.
- Kubernetes CronJob pods: `restartPolicy=Never` with `backoffLimit=0` — failure detected immediately, no silent infinite retry.
- FastAPI: Kubernetes liveness probe (`/health`, 30s initial delay, 15s period, 3 failure threshold) + readiness probe restarts unhealthy pods.

**Idempotent jobs:**
- Producer: GCS checkpoint prevents re-publishing the same row.
- Flink: PostgreSQL UNIQUE constraint + ON CONFLICT DO UPDATE; Redis RPUSH/LTRIM is idempotent.
- Training: MLflow run IDs are unique per run; duplicate runs are new experiments, not overwrites.
- Scoring: inference results written to timestamped GCS keys; no overwrite conflict.

**Rollback strategy:**
- MLflow Model Registry keeps all versions. Reverting to a previous version requires only updating the promotion manifest in GCS and restarting the model-serving pod.
- Airflow DAG `catchup=False` prevents backfill accumulation after downtime.

### 3.3 Training/Inference Pattern Decision

**Training mode:** Daily scheduled (`0 2 * * *` UTC) + drift-triggered failure (validate task exits 1 when `DRIFT_TOLERANCE` exceeded). However, we turn off this and trigger manually

**Inference mode:** Near-real-time stream inference via `/trigger` endpoint. After each Flink batch completes, the stream processor calls `POST /trigger` with the list of active engine IDs. FastAPI runs inference for each engine using the Redis window and writes results to GCS for the alert engine.

**Trade-offs:**
- Batch precompute vs stream inference: stream chosen to minimize alert latency. Results available within seconds of Flink batch completion.
- Online XGBoost (CPU) vs GPU model: XGBoost is CPU-native and fits within the `e2-standard-4` node budget. The LSTM AutoEncoder uses CPU inference only (`DEVICE=cpu`) to avoid GPU node costs.
- Daily retraining vs continuous: daily is sufficient for engine degradation timescales (cycles last hours to days in real fleets). Continuous retraining would be wasteful given the low data velocity.

### 3.4 Storage Decision

| Layer | Storage | Purpose |
|---|---|---|
| Bronze raw events | GCS (`raw_engine_cycles/`) | Immutable append-only telemetry |
| Silver validated events | GCS (`validated_engine_cycles/`) | Cleaned, quality-checked inputs |
| Online feature store | Memorystore Redis 7.0 (1 GB) | 30-cycle rolling window per engine for inference |
| Offline feature store | Cloud SQL PostgreSQL 15 (`engine_features`) | Historical features + RUL labels for training |
| Model artifacts | GCS (`phm-model-artifacts-aide2-494008`) | Pickled XGBoost + PyTorch LSTM weights + threshold |
| MLflow metadata | Cloud SQL PostgreSQL 15 (`mlflowdb`) | Experiment runs, params, metrics, registered models |
| Airflow metadata | Cloud SQL PostgreSQL 15 (`phmdb/airflow` schema) | DAG run history, task states, logs |
| Inference results | GCS (`inference-results/FD002/`) | Per-engine RUL + anomaly_score + anomaly_flag |
| Alert index | Cloud SQL PostgreSQL 15 (`alerts` table) | Alert history for Grafana dashboard |
| Logs | Loki (in-cluster) | Pod logs from all namespaces |
| Metrics | Prometheus (in-cluster) | FastAPI latency, error rate, pod resource usage |

**Model artifact retention:** GCS object versioning enabled on `phm-model-artifacts` bucket. Previous model versions retained indefinitely (cost-acceptable for < 100 model files).

### 3.5 Routing/Gateway Decision

**External access:**
- Dashboard Grafana: exposed via `ingress-nginx-public` namespace (public LoadBalancer, no auth for demo).
- Monitoring Grafana: exposed via `ingress-nginx-dev` namespace (LoadBalancer, basic auth).
- FastAPI model-serving: internal `ClusterIP` service only — no external exposure. Called by stream processor via Kubernetes service DNS (`model-serving-svc.model-serving.svc.cluster.local`).
- Airflow webserver: port-forwarded locally (`kubectl port-forward svc/airflow-webserver 8090:8090`).
- Flink Web UI: port-forwarded locally (`kubectl port-forward svc/flink-jobmanager 8081:8081`).

**Versioned routing:** Single `Production` stage in MLflow Model Registry. No A/B or canary routing in current phase. Rollback is a manual manifest update + pod restart.

---

## 4. Infrastructure Plan

**Choice:** GKE Standard (Kubernetes) — justified because the system has 8+ independent microservices that need isolated scaling, failure isolation, and independent update cadence. A single VM cannot provide namespace-level resource quotas or rolling deployments.

**Cluster configuration:**

| Resource | Spec | Justification |
|---|---|---|
| GKE node machine type | `e2-standard-4` (4 vCPU, 16 GB RAM) | Sufficient for all 8 namespaces + Flink JM+TM + Airflow scheduler+webserver |
| Node pool min/max | 1–2 nodes (autoscaling) | Scale-out during training (train_rul + train_anomaly parallel pods) |
| Node disk | 50 GB `pd-balanced` | Container image storage for 8 service images |
| Cloud SQL | `db-g1-small` (1 vCPU, 1.7 GB RAM), 20 GB SSD | PostgreSQL for MLflow + Airflow + feature store + alerts |
| Memorystore Redis | 1 GB, BASIC tier | Rolling window cache: 260 engines × ~2 KB ≈ 520 KB peak |
| GCS buckets | 4 buckets (raw, model-artifacts, mlflow, data-quality) | Separation of concerns; versioning on model-artifacts |

**Node sizing rationale:**
- Flink cluster: JobManager (512Mi/500m) + TaskManager (2Gi/1000m) = ~2.5 GB RAM peak.
- Airflow: Scheduler (1Gi/500m) + Webserver (1Gi/500m) = 2 GB RAM.
- Training pods: train_rul (2Gi/1000m) + train_anomaly (2Gi/1000m) = 4 GB RAM (parallel during DAG run). This triggers autoscale to 2 nodes during training window.
- Model-serving: 512Mi/500m steady state; spikes during `/trigger` batch inference.
- All other services (producer, validator, alert-engine, Grafana, Prometheus, Loki): < 2 GB total.

**Autoscaling policy:** GKE cluster autoscaler scales node pool 1→2 when any pod is `Pending` for >30s. Returns to 1 node when utilization drops below 50% for 10 minutes. Training pods are the primary scale trigger.

---

## 5. Pipeline Design

### 5.1 ML Pipeline Group

**Pipeline A — Training Pipeline** (Airflow DAG `model_training_pipeline`, daily 02:00 UTC):

```
Task 1a: train_rul      — XGBoost 10-run loop, log to MLflow phm-rul-xgboost-FD002-offline
Task 1b: train_anomaly  — LSTM AE 50 epochs, log to MLflow phm-anomaly-autoencoder-FD002
         (1a and 1b run in PARALLEL via Airflow KubernetesPodOperator)
Task 2:  validate       — RMSE gate (< 0.30) + drift check (< 0.05 normalized shift)
Task 3:  promote        — MLflow registry transition + GCS manifest write
```

**Pipeline B — Batch Serving Pipeline** (triggered per Flink batch):

```
stream_processor.py completes Flink job
  => calls POST /trigger with active engine_ids
  => model-serving runs inference for each engine
  => writes results to GCS inference-results/
  => alert-engine CronJob picks up results (next 5-min window)
```

**Pipeline C — Retraining Trigger Pipeline** (embedded in Airflow DAG validate task):

```
validate task runs:
  => RMSE gate: check mean_rmse < 0.30
  => Drift gate: check max normalized shift < 0.05
  => If either fails: exit(1) => promote skipped => engineer notified via Airflow alert
  => If both pass: tag baseline, exit(0) => promote runs
```

Retraining is triggered by the next scheduled DAG run. There is no automated requeue — the engineer reviews the drift report in Airflow and MLflow, then manually triggers a DAG run with the updated training data if needed.

### 5.2 CI/CD Pipeline Design

**Jenkins Pipeline** (7 stages, runs on `push` to GitHub via webhook):

```
Stage 1: Checkout         — git clone
Stage 2: Unit Test        — pytest test_serve.py with 70% coverage gate
          (30 tests covering: load_artifacts, predict_rul, predict_anomaly,
           write_result, health endpoint, infer endpoint, trigger endpoint,
           error handling, Redis mock, GCS mock)
Stage 3: Authenticate GCP — gcloud auth activate-service-account
Stage 4: Build Image      — docker build model-serving image
Stage 5: Push Image       — docker push to Artifact Registry
Stage 6: Helm Dependency  — helm dependency build charts/model-serving
Stage 7: Deploy           — helm upgrade --install model-serving
```

**CI/CD for Data Pipelines:**
- Data ingestion, feature platform, model training, and alert engine images are built and pushed manually (not automated in CI). Jenkins scope is intentionally limited to the model-serving namespace as the primary deployment artifact.
- Extension path: add a second Jenkins pipeline for each service image on `services/*/Dockerfile` change.

**CI/CD for IaC:**
- Terraform is applied manually (`terraform apply`) from the developer's machine with GCP credentials.
- State stored remotely in GCS (`phm-tf-state-aide2-494008`) — team members share state automatically.
- Extension path: add `iac_validate_plan` Jenkins stage that runs `terraform plan` on PR and `terraform apply` on merge to main.

**Promotion flow:** dev (local test) => Jenkins CI gate (unit tests + coverage) => GCP Artifact Registry => GKE model-serving Deployment (rolling update, zero downtime).

**Rollback:** `helm rollback model-serving` reverts to the previous Helm release. Old Docker image retained in Artifact Registry.

---

## 6. Monitoring Plan

**Prometheus metrics (scraped from FastAPI `/metrics`):**
- `http_requests_total{method, endpoint, status_code}` — request rate per endpoint
- `http_request_duration_seconds{endpoint}` — p50/p95/p99 latency histograms
- `process_resident_memory_bytes` — per-pod memory
- `container_cpu_usage_seconds_total` — per-namespace CPU

**Loki log streams (Promtail DaemonSet, all namespaces):**
- Stream Processor logs: `[INFO] All validated events processed`, `[ERROR]` keywords
- Model Serving logs: access log per request, `[ERROR]` on model load failure
- Alert Engine logs: `[INFO] Alert written`, `[ERROR]` on PostgreSQL/SendGrid failure
- Validation Errors: `[WARNING] INVALID` events with GCS path and failure reason

**Grafana dashboards (2 instances):**

| Instance | Access | Panels |
|---|---|---|
| Dashboard (public) | `ingress-nginx-public` LoadBalancer | 11 panels: Fleet RUL status, RUL degradation trends, anomaly score timeline, model performance metrics (RMSE, MAE), recent alerts table |
| Monitoring (dev) | `ingress-nginx-dev` LoadBalancer (basic auth) | 9 panels: pod restarts, request rate, inference latency p50/p95/p99, memory/CPU per namespace, 4 Loki log streams |

**Alert conditions:**
- Any pod restart in last 6 hours => Grafana alert.
- FastAPI request rate < 0.01 req/min => inference pipeline stalled.
- Inference latency p95 > 200ms => SLA breach.
- Validation error rate > 5% of events => data quality degradation.
- Drift gate failure in Airflow => engineer reviews MLflow drift report.

**Drift indicators:**
- `validate.py` logs per-feature mean shift to MLflow artifact `feature_means.json`.
- Airflow task state history (stored in PostgreSQL) shows FAILED for validate on drift events.
- Model RMSE trend in MLflow experiment view — degradation over runs indicates concept drift.

**Evidence:** `assets/imgs/monitoring_UI.png` — PHM Pipeline Health dashboard with all 9 panels active showing real inference, validation, stream processor, and alert engine log streams.

---

## 7. Retraining Strategy

**Scheduled retraining:** Daily at 02:00 UTC via Airflow DAG (`0 2 * * *`). Sufficient for turbofan degradation timescales (real engines operate for weeks to months between failure).

**Triggered retraining conditions:**

| Trigger | Mechanism | Threshold |
|---|---|---|
| RMSE degradation | `validate` task RMSE gate | `mean_rmse ≥ 0.30` |
| Feature drift | `validate` task drift check | `max normalized shift ≥ 0.05` |
| Anomaly rate spike | Manual review of alert dashboard | No automated trigger in current phase |

**Operational rules:**
1. `validate` task exits 1 => Airflow marks DAG as FAILED => no promotion => engineer notified via Airflow alert email (configurable).
2. Engineer reviews MLflow experiment view to identify which metric failed.
3. If drift: investigate source data for condition mix change. Update `DRIFT_TOLERANCE` if drift is expected and benign.
4. If RMSE: check training data size (online mode may have insufficient labeled rows). Rerun with `TRAINING_MODE=offline` to use full C-MAPSS dataset.
5. Manual DAG trigger: `airflow dags trigger model_training_pipeline` — reruns with latest data.

**Candidate vs production comparison:**
- MLflow `validate` task reads `metrics.mean_rmse` from the latest FINISHED run.
- `promote` task re-validates before registry transition.
- New model version registered as Staging before Production — allows rollback by transitioning previous Production version back to Production.

**Rollback path:**
1. MLflow Model Registry retains all versions.
2. Update GCS promotion manifest to reference previous version GCS URI.
3. Restart model-serving pod to reload artifacts.
4. No downtime — rolling pod restart completes in < 30 seconds.

---

## 8. DeliverablesPipeline

1. **Goal:** Section 1 — RUL prediction and anomaly detection for proactive maintenance.
2. **Prediction and modeling plan:** Section 2 — XGBoost RMSE=0.2721, LSTM AE threshold=0.117615, S-score metric, split strategy.
3. **High-Level ML design:** Section 3 — end-to-end flow, security (Workload Identity, least-privilege IAM), resilience (idempotent pipelines, retry policy), training/inference pattern (daily scheduled + stream inference), storage (dual-store), routing (internal ClusterIP + external ingress).
4. **Low-Level ML design:** Section 4 — 6 core service classes with method signatures.
5. **Infrastructure plan:** Section 5 — GKE `e2-standard-4`, 1–2 nodes autoscaling, Cloud SQL `db-g1-small`, Redis 1 GB, 4 GCS buckets. Justification tied to workload (parallel training pods trigger scale-out).
6. **Pipeline plan:** Section 6 — Airflow DAG (train_rul + train_anomaly => validate => promote), stream serving pipeline, retraining trigger, Jenkins CI/CD (7 stages, 70% coverage gate), IaC (Terraform manual apply, GCS remote state).
7. **Monitoring plan:** Section 7 — Prometheus (4 metric types), Loki (4 log streams), Grafana (2 instances, 20 panels total), 5 alert conditions, drift indicators.
8. **Retraining plan:** Section 8 — daily scheduled, RMSE gate + drift gate triggers, MLflow version comparison, rollback via manifest update.
