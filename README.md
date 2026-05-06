# MLOps Engine Predictive Maintenance

## Overview

<!-- Architecture diagram placeholder -->

## Table of Contents

- [Introduction](#introduction)
- [Target Audience](#target-audience)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Guide to Setup](#guide-to-setup)
  - [Step 0: Authenticate with GCP](#step-0-authenticate-with-gcp)
  - [Step 1: Provision Infrastructure with Terraform](#step-1-provision-infrastructure-with-terraform)
  - [Step 2: Configure kubectl](#step-2-configure-kubectl)
  - [Step 3: Authenticate Docker to Artifact Registry](#step-3-authenticate-docker-to-artifact-registry)
  - [Step 4: Generate and Upload Dataset](#step-4-generate-and-upload-dataset)
  - [Step 5: Deploy data-ingestion Namespace](#step-5-deploy-data-ingestion-namespace)
  - [Step 6: Deploy feature-platform Namespace](#step-6-deploy-feature-platform-namespace)
  - [Step 7: Deploy model-training Namespace](#step-7-deploy-model-training-namespace)
  - [Step 8: Deploy model-serving Namespace](#step-8-deploy-model-serving-namespace)
  - [Step 9: Deploy alert-engine Namespace](#step-9-deploy-alert-engine-namespace)
  - [Step 10: Deploy dashboard and ingress-nginx Namespaces](#step-10-deploy-dashboard-and-ingress-nginx-namespaces)
  - [Step 11: Deploy monitoring Namespace](#step-11-deploy-monitoring-namespace)
  - [Step 12: Deploy Jenkins Locally](#step-12-deploy-jenkins-locally)
- [Simulating Events](#simulating-events)
  - [Simulate a Validation Error](#simulate-a-validation-error)
  - [Simulate an Anomaly Detection Event](#simulate-an-anomaly-detection-event)
  - [Simulate an Email Alert](#simulate-an-email-alert)
- [Conclusion](#conclusion)
- [Reference](#reference)
- [Citation](#citation)

---

## Introduction

MLOps Engine Predictive Maintenance is a cloud-native MLOps pipeline for predicting the Remaining Useful Life (RUL) of aircraft turbofan engines using the NASA C-MAPSS FD002 dataset. The system simulates a real-time telemetry stream, performs feature engineering, trains and promotes ML models, serves predictions via a REST API, and fires alerts when an engine approaches failure.

The entire stack runs on Google Kubernetes Engine (GKE) Standard, provisioned by Terraform, deployed via Helm, and observed through Prometheus, Loki, and Grafana. A Jenkins CI/CD pipeline automates image builds and deployment of the model-serving namespace, gated by a unit test coverage requirement.

The pipeline implements two predictive models. The first is an XGBoost regressor for RUL prediction, achieving a mean RMSE of 0.2721 on the normalised FD002 test set. The second is an LSTM AutoEncoder for anomaly detection, which computes a reconstruction error against a threshold calibrated on the normal operating pool.

---

## Target Audience

This repository is suited to ML engineers and MLOps practitioners who want to study or build a production-grade end-to-end ML pipeline on GCP, covering data ingestion, feature engineering, model training, online inference, alerting, observability, and CI/CD.

---

## Repository Structure

```
MLOps_Engine_Predictive_Maintenance/
    CICD/
        Dockerfile                         # Custom Jenkins image with Docker, kubectl, Helm, gcloud, Python
    charts/
        alert-engine/
            templates/
                configmap.yaml             # Alert engine environment variables
                cronjob-alert-engine.yaml  # CronJob running every 5 minutes
                serviceaccount.yaml
            Chart.yaml
            values.yaml
        dashboard/
            templates/
                configmap.yaml             # Dashboard environment variables
                grafana.yaml               # Grafana Deployment, PVC, ConfigMaps, Service
                ingress.yaml               # Public ingress via ingress-nginx-public
                serviceaccount.yaml
            Chart.yaml
            values.yaml
        data-ingestion/
            templates/
                configmap.yaml
                cronjob-producer.yaml      # Telemetry producer CronJob
                cronjob-validation.yaml    # Validation service CronJob
                serviceaccount.yaml
            Chart.yaml
            values.yaml
        feature-platform/
            templates/
                configmap.yaml
                cronjob-stream-processor.yaml  # Micro-batch feature pipeline CronJob
                serviceaccount.yaml
            Chart.yaml
            values.yaml
        model-serving/
            templates/
                configmap.yaml
                deployment.yaml            # FastAPI inference server Deployment
                serviceaccount.yaml
            Chart.yaml
            values.yaml
        model-training/
            templates/
                configmap.yaml
                job.yaml                   # RUL + Anomaly training Jobs + registry promotion Job
                serviceaccount.yaml
            Chart.yaml
            values.yaml
        monitoring/
            templates/
                grafana.yaml               # Monitoring Grafana with Prometheus + Loki datasources, PVC
                ingress.yaml               # Dev ingress with basic auth
                loki.yaml                  # Loki + Promtail DaemonSet
                prometheus.yaml            # Prometheus with self-scrape and model-serving target
                serviceaccount.yaml
            Chart.yaml
            values.yaml
    data/
        CMaps/
            RUL_FD002.txt
            simulated_FD002.txt            # Generated by generate_simulated_FD002.py
            test_FD002.txt
            train_FD002.txt
        README.md
    services/
        alert-engine/
            src/
                alert_engine.py            # Alert rule engine + SendGrid notifier
            Dockerfile
            requirements.txt
        data-ingestion/
            src/
                generate_simulated_FD002.py
                producer.py                # Telemetry producer reads from GCS and emits cycle events
                validation_service.py      # Great Expectations validation with 4 suites
            Dockerfile
            requirements.txt
        feature-platform/
            src/
                stream_processor.py        # MicroBatchPipeline writing to Redis + PostgreSQL
            Dockerfile
            requirements.txt
        model-registry/
            src/
                promote.py                 # RMSE gate + drift check + GCS manifest promotion
            Dockerfile
            requirements.txt
        model-serving/
            src/
                serve.py                   # FastAPI inference orchestrator with XGBoost + LSTM AE
                test_serve.py              # 30 unit tests across 9 test classes
            Dockerfile
            requirements.txt
        model-training/
            src/
                train_RUL.py               # XGBoost RUL regressor training
                train_anomaly.py           # LSTM AutoEncoder anomaly model training
            Dockerfile
            requirements.txt
    terraforms/
        main.tf                            # GKE, Cloud SQL, Memorystore, GCS, Artifact Registry
        outputs.tf
        variables.tf
    .gitignore
    Jenkinsfile                            # CI/CD pipeline: Unit Test > Build > Push > Deploy
    README.md
    auth                                   # htpasswd file for monitoring basic auth (not committed)
```

---

## Prerequisites

- GCP account with billing enabled (free trial provides approximately $300 over 90 days)
- `gcloud` CLI installed and authenticated
- `terraform` >= 1.5 installed
- `docker` installed and running
- `helm` >= 3.12 installed
- `kubectl` installed
- `python3` >= 3.11 installed
- `htpasswd` utility installed (`sudo apt install apache2-utils`)

---

## Guide to Setup

### Step 0: Authenticate with GCP

```bash
gcloud init
gcloud auth application-default login
```

---

### Step 1: Provision Infrastructure with Terraform

This step creates all GCP resources: GKE Standard cluster (`gke-phm`, `e2-standard-4` nodes), Cloud SQL PostgreSQL (`phmdb`), Memorystore Redis (1 GB), GCS buckets, and Artifact Registry repositories.

1. Create a GCS bucket for Terraform remote state:

```bash
gcloud storage buckets create gs://phm-tf-state-aide2-494008 \
  --location=us-central1 \
  --uniform-bucket-level-access
```

2. Initialise, plan, and apply:

```bash
terraform init
terraform plan
terraform apply
```

3. Retrieve credentials at any time:

```bash
# Database password
terraform output -raw db_password

# Redis host
terraform output -raw redis_host

# Database private IP
terraform output -raw db_private_ip
```

---

### Step 2: Configure kubectl

```bash
gcloud container clusters get-credentials gke-phm \
  --region us-central1 \
  --project aide2-494008

# Verify
kubectl get nodes
```

---

### Step 3: Authenticate Docker to Artifact Registry

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev

gcloud auth print-access-token | docker login \
  -u oauth2accesstoken \
  --password-stdin \
  us-central1-docker.pkg.dev
```

If you encounter permission errors, add your user to the docker group:

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

---

### Step 4: Generate and Upload Dataset

Generate the simulated FD002 telemetry file from the raw C-MAPSS training data, then upload it to GCS so the producer CronJob can read from it:

```bash
python services/data-ingestion/src/generate_simulated_FD002.py \
  --data-dir data/CMaps \
  --output-dir data/CMaps \
  --normal-cycles 100

gsutil cp data/CMaps/simulated_FD002.txt \
  gs://phm-raw-data-aide2-494008/cmapss-data/simulated_FD002.txt
```

---

### Step 5: Deploy data-ingestion Namespace

The data-ingestion namespace runs two CronJobs: the telemetry producer which emits 50 engine cycle events per run, and the validation service which applies four Great Expectations suites (schema, null, range, order) and routes events to the validated or invalid GCS bucket.

1. Build and push images:

```bash
cd services/data-ingestion

docker build -t \
  us-central1-docker.pkg.dev/aide2-494008/phm-data-ingestion/data-ingestion:latest .

docker tag \
  us-central1-docker.pkg.dev/aide2-494008/phm-data-ingestion/data-ingestion:latest \
  us-central1-docker.pkg.dev/aide2-494008/phm-validation-service/validation-service:latest

docker push \
  us-central1-docker.pkg.dev/aide2-494008/phm-data-ingestion/data-ingestion:latest

docker push \
  us-central1-docker.pkg.dev/aide2-494008/phm-validation-service/validation-service:latest

cd ../..
```

2. Install chart:

```bash
helm install data-ingestion charts/data-ingestion \
  -n data-ingestion --create-namespace
```

3. Verify and trigger manually:

```bash
kubectl get cronjobs -n data-ingestion

kubectl create job --from=cronjob/telemetry-producer test-run -n data-ingestion
kubectl logs -n data-ingestion -l app=telemetry-producer -f

kubectl create job --from=cronjob/validation-service test-val -n data-ingestion
kubectl logs -n data-ingestion -l app=validation-service -f

# Cleanup
kubectl delete jobs --all -n data-ingestion
```

---

### Step 6: Deploy feature-platform Namespace

The feature-platform namespace runs a stream processor CronJob that reads validated cycle events from GCS, applies FD002 feature selection (dropping 8 low-correlation sensors), and writes a rolling 30-cycle window per engine to Redis (online store) and a full historical record to PostgreSQL (offline store).

1. Create secret and build image:

```bash
kubectl create namespace feature-platform
kubectl create secret generic feature-platform-secrets \
  --from-literal=db-password="$(terraform -chdir=terraforms output -raw db_password)" \
  -n feature-platform

cd services/feature-platform
docker build -t \
  us-central1-docker.pkg.dev/aide2-494008/phm-stream-processor/stream-processor:latest .
docker push \
  us-central1-docker.pkg.dev/aide2-494008/phm-stream-processor/stream-processor:latest
cd ../..
```

2. Fill in Terraform outputs into `charts/feature-platform/values.yaml`:

```bash
terraform -chdir=terraforms output redis_host
terraform -chdir=terraforms output db_private_ip
```

3. Install chart:

```bash
helm install feature-platform charts/feature-platform \
  -n feature-platform --create-namespace
```

4. Verify Redis and PostgreSQL data:

```bash
# Check Redis windows
kubectl run redis-cli --image=redis:7 -n feature-platform --rm -it \
  --restart=Never -- redis-cli -h 10.72.0.3 KEYS "engine:*:window"

# Check PostgreSQL features
kubectl run pg-client --image=postgres:15 -n feature-platform --rm -it \
  --restart=Never \
  --env="PGPASSWORD=$(terraform -chdir=terraforms output -raw db_password)" \
  -- psql -h $(terraform -chdir=terraforms output -raw db_private_ip) \
     -U phmadmin -d phmdb \
     -c "SELECT dataset, unit_id, cycle FROM engine_features LIMIT 5;"
```

---

### Step 7: Deploy model-training Namespace

The model-training namespace runs three Kubernetes Jobs: RUL model training (XGBoost, RMSE=0.2721), anomaly model training (LSTM AutoEncoder, threshold=0.117615), and model registry promotion (RMSE gate + drift check, writes promotion manifest to GCS).

1. Upload training data to GCS:

```bash
gsutil cp data/CMaps/train_FD002.txt gs://phm-raw-data-aide2-494008/offline/
gsutil cp data/CMaps/test_FD002.txt  gs://phm-raw-data-aide2-494008/offline/
gsutil cp data/CMaps/RUL_FD002.txt   gs://phm-raw-data-aide2-494008/offline/
```

2. Build and push images:

```bash
cd services/model-training
docker build -t \
  us-central1-docker.pkg.dev/aide2-494008/phm-model-training/model-training:latest .
docker push \
  us-central1-docker.pkg.dev/aide2-494008/phm-model-training/model-training:latest

cd ../model-registry
docker build -t \
  us-central1-docker.pkg.dev/aide2-494008/phm-model-registry/model-registry:latest .
docker push \
  us-central1-docker.pkg.dev/aide2-494008/phm-model-registry/model-registry:latest
cd ../..
```

3. Create namespace and secret, then install:

```bash
kubectl create namespace model-training
kubectl create secret generic model-training-secrets \
  --from-literal=db-password="$(terraform -chdir=terraforms output -raw db_password)" \
  -n model-training

helm install model-training charts/model-training \
  -n model-training --create-namespace
```

4. Watch training jobs:

```bash
kubectl get jobs -n model-training
kubectl logs -n model-training -l app=model-training -f
kubectl logs -n model-training -l app=model-registry -f

# Verify model artifacts in GCS
gsutil ls -r gs://phm-model-artifacts-aide2-494008/
```

5. Optional: access MLflow UI:

```bash
DB_HOST=$(terraform -chdir=terraforms output -raw db_private_ip)
DB_PASS=$(terraform -chdir=terraforms output -raw db_password)

kubectl run mlflow-ui \
  --image=ghcr.io/mlflow/mlflow:v2.10.0 \
  -n model-training \
  --env="DB_HOST=${DB_HOST}" \
  --env="DB_PASS=${DB_PASS}" \
  --command -- sh -c \
  'pip install --quiet psycopg2-binary && \
   python3 -c "
import os, subprocess
from urllib.parse import quote_plus
host = os.environ[\"DB_HOST\"]
pwd  = quote_plus(os.environ[\"DB_PASS\"])
uri  = f\"postgresql+psycopg2://phmadmin:{pwd}@{host}:5432/phmdb\"
subprocess.run([
    \"mlflow\", \"server\",
    \"--backend-store-uri\", uri,
    \"--default-artifact-root\", \"gs://phm-mlflow-artifacts-aide2-494008\",
    \"--host\", \"0.0.0.0\",
    \"--port\", \"5000\"
])"'

kubectl port-forward pod/mlflow-ui 5000:5000 -n model-training
# Open: http://localhost:5000
```

---

### Step 8: Deploy model-serving Namespace

The model-serving namespace runs a FastAPI inference server that loads the XGBoost RUL model and LSTM AutoEncoder from GCS at startup, reads 30-cycle windows from Redis, and exposes endpoints for per-engine inference (`/infer/{unit_id}`), anomaly-only inference (`/infer/{unit_id}/anomaly`), and batch trigger (`/trigger`). Results are written to GCS and Prometheus metrics are exposed at `/metrics`.

1. Fill in Redis host from Terraform into `charts/model-serving/values.yaml`, then build and push:

```bash
terraform -chdir=terraforms output redis_host

cd services/model-serving
docker build -t \
  us-central1-docker.pkg.dev/aide2-494008/phm-model-serving/model-serving:latest .
docker push \
  us-central1-docker.pkg.dev/aide2-494008/phm-model-serving/model-serving:latest
cd ../..
```

2. Install chart:

```bash
helm install model-serving charts/model-serving \
  -n model-serving --create-namespace
```

3. Verify and test:

```bash
# Wait for pod to be ready (model loading takes ~30s)
kubectl get pods -n model-serving -w

# Check models loaded
kubectl logs -n model-serving -l app=model-serving

# Health check
kubectl run curl-test --image=curlimages/curl -n model-serving --rm -it \
  --restart=Never -- curl -s http://model-serving-svc:80/health

# Test single engine inference
kubectl run curl-test --image=curlimages/curl -n model-serving --rm -it \
  --restart=Never -- \
  curl -s -X POST http://model-serving-svc:80/infer/1 \
  -H "Content-Type: application/json" -d '{"cycle": 100}'

# Batch trigger for all active engines
kubectl run curl-test --image=curlimages/curl -n model-serving --rm -it \
  --restart=Never -- \
  curl -s -X POST http://model-serving-svc:80/trigger \
  -H "Content-Type: application/json" -d '{"cycle": 100}'

# Verify results in GCS
gsutil ls -r gs://phm-raw-data-aide2-494008/inference-results/FD002/
```

---

### Step 9: Deploy alert-engine Namespace

The alert-engine namespace runs a CronJob every 5 minutes that polls GCS inference results, evaluates two rules (RUL < 20 cycles and anomaly_flag = True), writes alert rows to PostgreSQL, and sends email notifications via SendGrid.

1. Build and push image:

```bash
cd services/alert-engine
docker build -t \
  us-central1-docker.pkg.dev/aide2-494008/phm-alert-engine/alert-engine:latest .
docker push \
  us-central1-docker.pkg.dev/aide2-494008/phm-alert-engine/alert-engine:latest
cd ../..
```

2. Create namespace and secrets:

```bash
kubectl create namespace alert-naming

kubectl create secret generic alert-engine-secrets \
  --from-literal=db-password="$(terraform -chdir=terraforms output -raw db_password)" \
  --from-literal=sendgrid-api-key="YOUR_SENDGRID_API_KEY" \
  -n alert-naming
```

3. Update `charts/alert-engine/templates/configmap.yaml` with notifier settings:

```yaml
NOTIFY_ENABLED:   "true"
NOTIFY_SEVERITY:  "critical"
ALERT_FROM_EMAIL: "your-verified-email@gmail.com"
ALERT_TO_EMAIL:   "your-verified-email@gmail.com"
```

Note: `ALERT_FROM_EMAIL` must be a verified sender in SendGrid. Use Single Sender Verification at `app.sendgrid.com/settings/sender_auth` if you do not own a domain.

4. Install chart and verify:

```bash
helm install alert-engine charts/alert-engine \
  -n alert-naming --create-namespace

kubectl get cronjobs -n alert-naming

kubectl create job --from=cronjob/alert-engine test-alerts -n alert-naming
kubectl logs -n alert-naming -l app=alert-engine -f

# Verify alerts in PostgreSQL
kubectl run pg-check --image=postgres:15 -n alert-naming --rm -it \
  --restart=Never \
  --env="PGPASSWORD=$(terraform -chdir=terraforms output -raw db_password)" \
  -- psql -h $(terraform -chdir=terraforms output -raw db_private_ip) \
     -U phmadmin -d phmdb \
     -c "SELECT alert_type, severity, engine_id, rul, message FROM alerts ORDER BY indexed_at DESC LIMIT 10;"
```

---

### Step 10: Deploy dashboard and ingress-nginx Namespaces

The dashboard namespace runs a Grafana instance connected to PostgreSQL, showing 11 panels covering fleet RUL status, RUL degradation trends, anomaly scores, model metrics, and recent alerts. It is exposed publicly via `ingress-nginx-public`. The monitoring namespace Grafana is exposed via `ingress-nginx-dev` with basic auth.

1. Install both ingress-nginx controllers:

```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

# Public controller for dashboard
helm install ingress-nginx-public ingress-nginx/ingress-nginx \
  --namespace ingress-nginx-public \
  --create-namespace \
  --set controller.ingressClassResource.name=nginx-public \
  --set controller.ingressClassResource.controllerValue="k8s.io/ingress-nginx-public" \
  --set controller.service.type=LoadBalancer \
  --set controller.resources.requests.cpu=100m \
  --set controller.resources.requests.memory=128Mi

# Dev controller for monitoring
helm install ingress-nginx-dev ingress-nginx/ingress-nginx \
  --namespace ingress-nginx-dev \
  --create-namespace \
  --set controller.ingressClassResource.name=nginx-dev \
  --set controller.ingressClassResource.controllerValue="k8s.io/ingress-nginx-dev" \
  --set controller.service.type=LoadBalancer \
  --set controller.resources.requests.cpu=100m \
  --set controller.resources.requests.memory=128Mi

# Wait for public IPs
kubectl get svc -n ingress-nginx-public -w
kubectl get svc -n ingress-nginx-dev -w
```

2. Fill in Terraform outputs into `charts/dashboard/values.yaml` and `charts/dashboard/templates/grafana.yaml`, then deploy:

```bash
kubectl create secret generic dashboard-secrets \
  --from-literal=db-password="$(terraform -chdir=terraforms output -raw db_password)" \
  -n dashboard

helm install dashboard charts/dashboard \
  -n dashboard --create-namespace

kubectl get pods -n dashboard -w
```

3. First login: open `http://PUBLIC_IP/grafana`, username `admin`, password `admin`. Grafana forces a password change on first login. The new password is persisted in the PVC and survives pod restarts.

---

### Step 11: Deploy monitoring Namespace

The monitoring namespace runs Prometheus (scraping model-serving metrics), Loki (aggregating pod logs from all namespaces via Promtail), and a second Grafana instance with 9 panels covering pod restarts, FastAPI request rate, inference latency (p50/p95/p99), memory and CPU usage, and log streams for stream processor, model serving, alert engine, and validation errors.

1. Get the dev ingress IP and fill it into the monitoring values:

```bash
DEV_IP=$(kubectl get svc ingress-nginx-dev-controller \
  -n ingress-nginx-dev \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Dev IP: $DEV_IP"

sed -i "s|DEV_INGRESS_IP|${DEV_IP}|g" charts/monitoring/values.yaml
sed -i "s|DEV_INGRESS_IP|${DEV_IP}|g" charts/monitoring/templates/prometheus.yaml
```

2. Create basic auth secret:

```bash
htpasswd -c auth devadmin
kubectl create namespace monitoring
kubectl create secret generic monitoring-basic-auth \
  --from-file=auth -n monitoring
```

3. Install chart:

```bash
helm install monitoring charts/monitoring \
  -n monitoring --create-namespace

kubectl get pods -n monitoring -w
```

4. Fix Promtail ConfigMap conflict and apply working static log config:

The Helm-managed Promtail ConfigMap uses Kubernetes SD for log path discovery, which does not work reliably on GKE Standard with containerd. Replace it with a static wildcard config that tails all pod logs directly and extracts namespace, pod, and container labels via a regex pipeline stage:

```bash
kubectl delete configmap promtail-config -n monitoring
helm upgrade monitoring charts/monitoring -n monitoring

kubectl create configmap promtail-config \
  --from-literal=promtail.yaml='
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki-svc.monitoring.svc.cluster.local:3100/loki/api/v1/push

scrape_configs:
  - job_name: pod-logs
    pipeline_stages:
      - cri: {}
      - regex:
          expression: /var/log/pods/(?P<namespace>[^_]+)_(?P<pod>[^_]+(?:_[^_]+)*)_[^/]+/(?P<container>[^/]+)/.*
          source: filename
      - labels:
          namespace:
          pod:
          container:
    static_configs:
      - targets:
          - localhost
        labels:
          job: pod-logs
          __path__: /var/log/pods/*/*/*.log
' \
  --dry-run=client -o yaml | kubectl apply -f - -n monitoring

kubectl delete pod -n monitoring -l app=promtail
kubectl rollout status daemonset/promtail -n monitoring
```

5. Verify all components:

```bash
# Prometheus ready
curl -u devadmin:YOUR_PASSWORD \
  http://${DEV_IP}/prometheus/-/ready

# Grafana health
curl -u devadmin:YOUR_PASSWORD \
  http://${DEV_IP}/monitoring/api/health

# Loki labels (should return namespace, pod, container, filename, job, stream)
kubectl run curl-test --image=curlimages/curl -n monitoring --rm -it \
  --restart=Never -- \
  curl -s "http://loki-svc:3100/loki/api/v1/labels" 2>/dev/null | \
  grep -v "BCID\|recorded\|prompt\|warning"
```

6. Access dashboards:

```bash
echo "Monitoring Grafana: http://${DEV_IP}/monitoring"
echo "Prometheus UI:      http://${DEV_IP}/prometheus"
```

First login: username `admin`, password `admin`. Grafana forces a password change on first login. The new password is persisted in the PVC.

---

### Step 12: Deploy Jenkins Locally

Jenkins is deployed locally using Docker. The custom image built from `CICD/Dockerfile` includes the Docker CLI, kubectl, Helm 3, gcloud with the GKE auth plugin, and a Python virtual environment with all packages needed to run the model-serving unit tests.

1. Build the custom Jenkins image:

```bash
cd CICD
docker build -t phm-jenkins .
cd ..
```

2. Run Jenkins:

```bash
docker run -d \
  --name jenkins \
  --restart unless-stopped \
  -p 8080:8080 \
  -p 50000:50000 \
  -v jenkins-data:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  phm-jenkins
```

3. Get the initial admin password and complete setup at `http://localhost:8080`:

```bash
docker exec jenkins \
  cat /var/jenkins_home/secrets/initialAdminPassword
```

4. After setup, add one credential under Manage Jenkins > Credentials > Global:
   - Kind: Secret file
   - ID: `gcp-sa-key`
   - File: your GCP service account JSON key
```bash
gcloud iam service-accounts create jenkins-deployer \
  --display-name "Jenkins GKE Deployer" \
  --project aide2-494008
  
# Grant required IAM roles
# Push images to Artifact Registry
gcloud projects add-iam-policy-binding aide2-494008 \
  --member="serviceAccount:jenkins-deployer@aide2-494008.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

# Deploy to GKE
gcloud projects add-iam-policy-binding aide2-494008 \
  --member="serviceAccount:jenkins-deployer@aide2-494008.iam.gserviceaccount.com" \
  --role="roles/container.developer"

# Read GKE cluster credentials
gcloud projects add-iam-policy-binding aide2-494008 \
  --member="serviceAccount:jenkins-deployer@aide2-494008.iam.gserviceaccount.com" \
  --role="roles/container.clusterViewer"
  
# Generate the key file (do not commit this file)
gcloud iam service-accounts keys create jenkins-gke.json \
  --iam-account jenkins-deployer@aide2-494008.iam.gserviceaccount.com
```
5. Create a Pipeline job pointing at this repository with Script Path set to `Jenkinsfile`. The pipeline runs 7 stages: Checkout, Unit Test (70% coverage gate), Authenticate to GCP, Build Image, Push Image, Helm Dependency Build, and Deploy. The deployment is blocked if any unit test fails or coverage falls below 70%.

---

## Simulating Events

### Simulate a Validation Error

Upload a malformed engine cycle event with `SensorMes2 = 9999` which is outside the valid FD002 range of 530 to 650. The validation service will route it to the quality bucket and log an INVALID line visible in the Grafana Validation Errors panel.

```bash
cat > /tmp/bad_event.json << 'EOF'
{
  "event_type": "engine_cycle",
  "dataset": "FD002",
  "unit_id": 999,
  "cycle": 1,
  "row_index": 99999,
  "emitted_at": "2026-05-03T00:00:00+00:00",
  "sensors": {
    "OperSet1": 35.0, "OperSet2": 0.84, "OperSet3": 100.0,
    "SensorMes1": 449.0, "SensorMes2": 9999.0, "SensorMes3": 1358.0,
    "SensorMes4": 1137.0, "SensorMes5": 5.48, "SensorMes6": 8.0,
    "SensorMes7": 194.0, "SensorMes8": 2222.0, "SensorMes9": 8341.0,
    "SensorMes10": 1.02, "SensorMes11": 42.0, "SensorMes12": 183.0,
    "SensorMes13": 2387.0, "SensorMes14": 8048.0, "SensorMes15": 9.34,
    "SensorMes16": 0.02, "SensorMes17": 334.0, "SensorMes18": 2223.0,
    "SensorMes19": 100.0, "SensorMes20": 14.73, "SensorMes21": 8.8
  }
}
EOF

gsutil cp /tmp/bad_event.json \
  gs://phm-raw-data-aide2-494008/raw_engine_cycles/FD002/unit_999/20260503T000000000000_99999.json

# Trigger validation service manually to process it immediately
kubectl create job --from=cronjob/validation-service test-val -n data-ingestion
kubectl logs -n data-ingestion -l job-name=test-val -f
```

---

### Simulate an Anomaly Detection Event

Inject a window of 30 cycles with extreme out-of-distribution sensor values directly into Redis for engine 999. The LSTM AutoEncoder will produce a reconstruction error far above the 0.117615 threshold, setting `anomaly_flag = True`.

1. Inject the anomalous window into Redis:

```bash
kubectl run redis-inject --image=redis:7 -n model-serving --rm -it \
  --restart=Never -- \
  redis-cli -h 10.72.0.3 SET "engine:999:window" \
  '[{"cycle":1,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":2,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":3,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":4,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":5,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":6,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":7,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":8,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":9,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":10,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":11,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":12,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":13,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":14,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":15,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":16,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":17,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":18,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":19,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":20,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":21,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":22,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":23,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":24,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":25,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":26,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":27,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":28,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":29,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}},{"cycle":30,"features":{"SensorMes2":999.0,"SensorMes3":9999.0,"SensorMes4":9999.0,"SensorMes6":99.0,"SensorMes7":999.0,"SensorMes8":9999.0,"SensorMes9":99999.0,"SensorMes11":99.0,"SensorMes12":999.0,"SensorMes13":9999.0,"SensorMes14":99999.0,"SensorMes15":99.0,"SensorMes16":9.0,"SensorMes17":9999.0,"SensorMes20":99.0,"SensorMes21":99.0}}]'
```

2. Trigger inference on engine 999:

```bash
kubectl run curl-test --image=curlimages/curl -n model-serving --rm -it \
  --restart=Never -- \
  curl -s -X POST http://model-serving-svc:80/infer/999 \
  -H "Content-Type: application/json" \
  -d '{"cycle": 999}'
```

3. Run the alert engine to write the anomaly alert to PostgreSQL and send an email:

```bash
kubectl create job --from=cronjob/alert-engine demo-anomaly -n alert-naming
kubectl logs -n alert-naming -l job-name=demo-anomaly -f
```

The Grafana Anomaly Score panel in the dashboard namespace queries `alert_type = 'anomaly_detected'` and will display engine 999 within the next 30-second refresh.

---

### Simulate an Email Alert

The alert engine sends email for every `critical` severity alert (RUL < 10) when `NOTIFY_ENABLED = true`. To force a fresh run that retriggers all previous alerts including the engine 999 anomaly and RUL alerts:

```bash
# Reset the alert checkpoint so all results are reprocessed
gsutil rm gs://phm-raw-data-aide2-494008/checkpoints/FD002/alert_processed_keys.json

kubectl create job --from=cronjob/alert-engine test-notify -n alert-naming
kubectl logs -n alert-naming -l job-name=test-notify -f
# Expected: Email sent — engine=999 type=rul_threshold status=202
```

Emails arrive in the spam folder for SendGrid trial accounts without domain authentication. Mark them as not spam to train Gmail. In a production deployment, configure Domain Authentication (DMARC/SPF/DKIM) in the SendGrid sender authentication settings to ensure inbox delivery.

---

## Conclusion

This repository demonstrates a complete production-grade MLOps pipeline for predictive maintenance on GCP. The pipeline covers every stage from raw telemetry ingestion and data quality validation through feature engineering, model training, registry promotion, online inference, alerting with email notification, and full observability via metrics and logs. The CI/CD pipeline automates the model-serving deployment gated by unit tests, following the same pattern as a real engineering team would use for a production service.

---

## Reference

This project is inspired by the Hall of Fame community at Full Stack Data Science. Visit their page at https://fullstackdatascience.com/hall-of-fame.

The C-MAPSS dataset and damage propagation model are described in: A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

---

## Citation

If you use this project in your research or work, please cite it as follows:

```
@software{MLOpsEnginePredictiveMaintenance2026,
  author  = {Le, Huy Phuong},
  title   = {MLOps Engine Predictive Maintenance: A Cloud-Native RUL Prediction Pipeline on GCP},
  year    = {2026},
  url     = {https://github.com/lehuyphuong/MLOps_Engine_Predictive_Maintenance},
  note    = {GKE-native MLOps pipeline with XGBoost RUL prediction, LSTM AutoEncoder anomaly detection,
             Prometheus/Loki/Grafana observability, and Jenkins CI/CD}
}
```
