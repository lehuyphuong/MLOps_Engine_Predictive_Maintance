# ---------------------------------------------------------------
# GKE
# ---------------------------------------------------------------

output "cluster_name" {
  description = "Name of the GKE cluster"
  value       = google_container_cluster.phm_gke.name
}

output "cluster_location" {
  description = "Region of the GKE cluster"
  value       = google_container_cluster.phm_gke.location
}

output "cluster_endpoint" {
  description = "Public endpoint of the GKE control plane"
  value       = google_container_cluster.phm_gke.endpoint
}

output "cluster_ca_certificate" {
  description = "Cluster CA certificate (base64)"
  value       = google_container_cluster.phm_gke.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "gcloud_get_credentials_command" {
  description = "Run this to configure kubectl"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.phm_gke.name} --region ${var.region} --project ${var.project_id}"
}

# ---------------------------------------------------------------
# Artifact Registry — image URLs for Helm values.yaml
# ---------------------------------------------------------------

output "artifact_registry_urls" {
  description = "Map of service → Artifact Registry image URL base (append :tag)"
  value = {
    for k, repo in google_artifact_registry_repository.phm_repos :
    k => "${var.region}-docker.pkg.dev/${var.project_id}/${repo.repository_id}"
  }
}

output "docker_auth_command" {
  description = "Run this once to configure docker to push to Artifact Registry"
  value       = "gcloud auth configure-docker ${var.region}-docker.pkg.dev"
}

# ---------------------------------------------------------------
# GCS Buckets
# ---------------------------------------------------------------

output "gcs_raw_data_bucket" {
  description = "GCS bucket for raw C-MAPSS sensor events"
  value       = google_storage_bucket.phm_buckets["raw_data"].name
}

output "gcs_model_artifacts_bucket" {
  description = "GCS bucket for trained model binaries"
  value       = google_storage_bucket.phm_buckets["model_artifacts"].name
}

output "gcs_mlflow_bucket" {
  description = "GCS bucket used as MLflow artifact store (set as gs://... in MLflow URI)"
  value       = google_storage_bucket.phm_buckets["mlflow"].name
}

output "gcs_data_quality_bucket" {
  description = "GCS bucket for invalid / rejected records"
  value       = google_storage_bucket.phm_buckets["data_quality"].name
}

# ---------------------------------------------------------------
# Cloud SQL — PostgreSQL
# ---------------------------------------------------------------

output "db_private_ip" {
  description = "Private IP of the Cloud SQL PostgreSQL instance — use this as DB_HOST in all configmaps"
  value       = google_sql_database_instance.phm_pg.private_ip_address
}

output "db_connection_name" {
  description = "Cloud SQL connection name (project:region:instance) — needed if using Cloud SQL Auth Proxy"
  value       = google_sql_database_instance.phm_pg.connection_name
}

output "db_name" {
  description = "Database name"
  value       = google_sql_database.phmdb.name
}

output "db_username" {
  description = "Database master username"
  value       = google_sql_user.phmadmin.name
}

output "db_password" {
  description = "Auto-generated DB password — retrieve with: terraform output -raw db_password"
  value       = random_password.db.result
  sensitive   = true
}

# ---------------------------------------------------------------
# Memorystore Redis
# ---------------------------------------------------------------

output "redis_host" {
  description = "Redis private IP — use as REDIS_HOST in feature-platform and model-serving configmaps"
  value       = google_redis_instance.phm_redis.host
}

output "redis_port" {
  description = "Redis port"
  value       = google_redis_instance.phm_redis.port
}

# ---------------------------------------------------------------
# Workload Identity — GSA emails for Helm serviceAccount annotations
# ---------------------------------------------------------------

output "workload_identity_annotations" {
  description = "Paste these into each chart's values.yaml under serviceAccount.annotations"
  value = {
    data_ingestion   = "iam.gke.io/gcp-service-account: ${google_service_account.data_ingestion.email}"
    feature_platform = "iam.gke.io/gcp-service-account: ${google_service_account.feature_platform.email}"
    model_training   = "iam.gke.io/gcp-service-account: ${google_service_account.model_training.email}"
    model_serving    = "iam.gke.io/gcp-service-account: ${google_service_account.model_serving.email}"
    alert_engine     = "iam.gke.io/gcp-service-account: ${google_service_account.alert_engine.email}"
  }
}

# ---------------------------------------------------------------
# Networking
# ---------------------------------------------------------------

output "vpc_name" {
  description = "VPC network name"
  value       = google_compute_network.phm_vpc.name
}

output "subnet_name" {
  description = "Subnetwork name"
  value       = google_compute_subnetwork.phm_subnet.name
}
