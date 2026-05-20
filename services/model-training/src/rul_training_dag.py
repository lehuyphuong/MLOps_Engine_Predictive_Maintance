"""
rul_training_dag.py — Model Training Pipeline DAG

Replaces the three manual kubectl Jobs (rul-training-job, anomaly-training-job,
registry-job) with a proper Airflow DAG that enforces task dependencies,
provides retry logic, and records full run history in the Airflow UI.

DAG topology:
    train_rul + train_anomaly => validate => promote

Task details:
    train_rul     — runs train_RUL.py   (XGBoost, 10 runs, logs to MLflow)
    train_anomaly — runs train_anomaly.py (LSTM AE, logs to MLflow)
                    both run in PARALLEL — independent models, no data dependency
    validate      — runs validate.py
                    RMSE gate: mean_rmse < RMSE_THRESHOLD (0.30)
                    Drift check: feature drift < DRIFT_TOLERANCE (0.05)
                    Exits non-zero on failure → Airflow marks FAILED → promote skipped
    promote       — runs promote.py
                    Registers winning model in MLflow Model Registry
                    Pushes serving manifest to GCS for model-serving namespace

Schedule: daily at 02:00 UTC (off-peak, after overnight data accumulation)
Can be triggered manually from the Airflow UI or REST API.

Each task runs as a KubernetesPodOperator pod in the model-training namespace,
using the same Docker image and ConfigMap as the existing manual Jobs.
No Celery / Redis broker needed — KubernetesExecutor only.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.utils.trigger_rule import TriggerRule
from kubernetes.client import models as k8s

# Shared pod configuration
# All tasks use the same model-training image and inherit env from the
# existing model-training-config ConfigMap and model-training-secrets Secret.
# This means zero changes to train_RUL.py, train_anomaly.py, or promote.py.

IMAGE      = os.environ.get(
    "MODEL_TRAINING_IMAGE",
    "us-central1-docker.pkg.dev/aide2-494008/phm-model-training/model-training:latest",
)

# Fallback: if env var is empty string (not just missing), use the default
if not IMAGE or IMAGE.strip() == "":
    IMAGE = "us-central1-docker.pkg.dev/aide2-494008/phm-model-training/model-training:latest"
NAMESPACE  = "model-training"

ENV_FROM = [
    k8s.V1EnvFromSource(
        config_map_ref=k8s.V1ConfigMapEnvSource(name="model-training-config")
    )
]

DB_PASSWORD_ENV = k8s.V1EnvVar(
    name="DB_PASSWORD",
    value_from=k8s.V1EnvVarSource(
        secret_key_ref=k8s.V1SecretKeySelector(
            name="model-training-secrets",
            key="db-password",
        )
    ),
)

TRAINING_RESOURCES = k8s.V1ResourceRequirements(
    requests={"cpu": "500m",  "memory": "1Gi"},
    limits=  {"cpu": "1000m", "memory": "2Gi"},
)

LIGHT_RESOURCES = k8s.V1ResourceRequirements(
    requests={"cpu": "100m",  "memory": "256Mi"},
    limits=  {"cpu": "300m",  "memory": "512Mi"},
)

DEFAULT_ARGS = {
    "owner":             "mlops",
    "retries":           1,
    "retry_delay":       timedelta(minutes=5),
    "execution_timeout": timedelta(hours=3),
}

# DAG
with DAG(
    dag_id           = "model_training_pipeline",
    description      = "RUL XGBoost + Anomaly LSTM-AE → validate → promote",
    schedule_interval= "0 2 * * *",    # daily at 02:00 UTC
    start_date       = datetime(2026, 1, 1),
    catchup          = False,
    default_args     = DEFAULT_ARGS,
    tags             = ["mlops", "predictive-maintenance", "rul", "anomaly"],
    doc_md           = __doc__,
) as dag:

    # Task 1a: RUL Model Training
    train_rul = KubernetesPodOperator(
        task_id                = "train_rul",
        name                   = "train-rul",
        namespace              = NAMESPACE,
        image                  = IMAGE,
        image_pull_policy      = "Always",
        in_cluster             = True,
        service_account_name   = "model-training-sa",
        arguments              = ["src/train_RUL.py"],
        env_from               = ENV_FROM,
        env_vars               = [DB_PASSWORD_ENV],
        container_resources    = TRAINING_RESOURCES,
        is_delete_operator_pod = False,
        get_logs               = True,
        log_events_on_failure  = True,
    )

    # Task 1b: Anomaly Model Training (parallel with train_rul)
    train_anomaly = KubernetesPodOperator(
        task_id                = "train_anomaly",
        name                   = "train-anomaly",
        namespace              = NAMESPACE,
        image                  = IMAGE,
        image_pull_policy      = "Always",
        in_cluster             = True,
        service_account_name   = "model-training-sa",
        arguments              = ["src/train_anomaly.py"],
        env_from               = ENV_FROM,
        env_vars               = [
            DB_PASSWORD_ENV,
            k8s.V1EnvVar(name="NORMAL_RUL_THRESHOLD", value="150"),
            k8s.V1EnvVar(name="SEQUENCE_LENGTH",       value="30"),
            k8s.V1EnvVar(name="EPOCHS",                value="50"),
            k8s.V1EnvVar(name="BATCH_SIZE",            value="32"),
            k8s.V1EnvVar(name="LEARNING_RATE",         value="0.001"),
        ],
        container_resources    = TRAINING_RESOURCES,
        is_delete_operator_pod = False,
        get_logs               = True,
        log_events_on_failure  = True,
    )

    # Task 2: Validation
    validate = KubernetesPodOperator(
        task_id                = "validate",
        name                   = "validate-models",
        namespace              = NAMESPACE,
        image                  = IMAGE,
        image_pull_policy      = "Always",
        in_cluster             = True,
        service_account_name   = "model-training-sa",
        arguments              = ["src/validate.py"],
        env_from               = ENV_FROM,
        env_vars               = [DB_PASSWORD_ENV],
        container_resources    = LIGHT_RESOURCES,
        is_delete_operator_pod = True,
        get_logs               = True,
        log_events_on_failure  = True,
        trigger_rule           = TriggerRule.ALL_SUCCESS,
    )

    # Task 3: Promote
    promote = KubernetesPodOperator(
        task_id                = "promote",
        name                   = "promote-models",
        namespace              = NAMESPACE,
        image                  = "us-central1-docker.pkg.dev/aide2-494008/phm-model-registry/model-registry:latest",
        image_pull_policy      = "Always",
        in_cluster             = True,
        service_account_name   = "model-training-sa",
        arguments              = ["src/promote.py"],
        env_from               = ENV_FROM,
        env_vars               = [DB_PASSWORD_ENV],
        container_resources    = LIGHT_RESOURCES,
        is_delete_operator_pod = True,
        get_logs               = True,
        log_events_on_failure  = True,
    )

    [train_rul, train_anomaly] >> validate >> promote