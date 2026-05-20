terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  # Remote backend — state lives in GCS, not on your local machine.
  # One-time bootstrap before first terraform init:
  #   gcloud storage buckets create gs://phm-tf-state-aide2-494008 \
  #     --location=us-central1 --uniform-bucket-level-access
  #
  # Replace the bucket value below with your actual bucket name.
  backend "gcs" {
    bucket = "phm-tf-state-aide2-494008"
    prefix = "phm/terraform.tfstate"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# DB password auto-generated — no prompt needed on plan/apply
# Retrieve after apply: terraform output -raw db_password
resource "random_password" "db" {
  length           = 16
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

# ---------------------------------------------------------------
# Step 0 — Enable required GCP APIs
#
# This block MUST complete before any other resource is created.
# Every resource group below carries depends_on = [google_project_service.apis]
# to enforce this ordering and prevent the race condition where
# Terraform tries to create a VPC or Artifact Registry repo before
# the Compute / Artifact Registry APIs are fully active.
# ---------------------------------------------------------------

resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",             # VPC, GKE nodes, Cloud NAT
    "container.googleapis.com",           # GKE control plane
    "artifactregistry.googleapis.com",    # Artifact Registry (replaces ECR)
    "storage.googleapis.com",             # GCS (replaces S3)
    "sqladmin.googleapis.com",            # Cloud SQL (replaces RDS)
    "redis.googleapis.com",               # Memorystore (replaces ElastiCache)
    "servicenetworking.googleapis.com",   # Private service access for SQL + Redis
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
  ])

  service            = each.key
  disable_on_destroy = false
}

# ---------------------------------------------------------------
# Step 1 — Networking: VPC + Subnetwork + Cloud NAT + Firewall
# depends_on: APIs must be enabled first (compute.googleapis.com)
# ---------------------------------------------------------------

resource "google_compute_network" "phm_vpc" {
  name                    = "${var.project}-vpc"
  auto_create_subnetworks = false
  description             = "VPC for PHM predictive maintenance pipeline"

  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "phm_subnet" {
  name          = "${var.project}-subnet"
  ip_cidr_range = var.subnet_ip_cidr
  region        = var.region
  network       = google_compute_network.phm_vpc.id

  # Secondary ranges for VPC-native GKE (pods + services)
  secondary_ip_range {
    range_name    = var.pods_secondary_range_name
    ip_cidr_range = var.pods_ip_cidr
  }

  secondary_ip_range {
    range_name    = var.services_secondary_range_name
    ip_cidr_range = var.services_ip_cidr
  }

  # Allow pods to reach Cloud SQL / Memorystore / GCS privately
  private_ip_google_access = true

  description = "Subnetwork for PHM GKE cluster and managed services"

  depends_on = [google_project_service.apis]
}

# Cloud NAT — lets GKE nodes pull images without a public IP on each node
resource "google_compute_router" "phm_router" {
  name    = "${var.project}-router"
  region  = var.region
  network = google_compute_network.phm_vpc.id

  depends_on = [google_project_service.apis]
}

resource "google_compute_router_nat" "phm_nat" {
  name                               = "${var.project}-nat"
  router                             = google_compute_router.phm_router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = false
    filter = "ERRORS_ONLY"
  }

  depends_on = [google_project_service.apis]
}

# Firewall — allow GKE control plane to reach nodes (required for webhooks)
resource "google_compute_firewall" "gke_master_webhook" {
  name    = "${var.project}-gke-master-webhook"
  network = google_compute_network.phm_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["8443", "9443", "15017"]
  }

  source_ranges = [var.master_ipv4_cidr_block]
  target_tags   = ["gke-${var.cluster_name}"]
  description   = "GKE master to node webhook traffic"

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------
# Step 2 — VPC peering for Cloud SQL + Memorystore private IP
# depends_on: VPC must exist, servicenetworking API must be active
# ---------------------------------------------------------------

resource "google_compute_global_address" "phm_sql_peering" {
  name          = "${var.project}-sql-peering-range"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.phm_vpc.id

  depends_on = [google_project_service.apis]
}

resource "google_service_networking_connection" "phm_vpc_peering" {
  network                 = google_compute_network.phm_vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.phm_sql_peering.name]

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------
# Step 3 — GKE Standard Cluster + Node Pool
# depends_on: networking + APIs
# The GKE cluster creation also initialises the Workload Identity
# pool (PROJECT_ID.svc.id.goog) — all WI bindings in Step 6
# depend on this cluster being fully created.
# ---------------------------------------------------------------

resource "google_container_cluster" "phm_gke" {
  name     = var.cluster_name
  location = var.region   # regional cluster — HA control plane

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.phm_vpc.self_link
  subnetwork = google_compute_subnetwork.phm_subnet.self_link

  networking_mode = "VPC_NATIVE"

  ip_allocation_policy {
    cluster_secondary_range_name  = var.pods_secondary_range_name
    services_secondary_range_name = var.services_secondary_range_name
  }

  release_channel {
    channel = var.release_channel
  }

  node_locations = var.node_locations

  # Workload Identity — replaces AWS IRSA
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Private nodes — no public IP on worker nodes, Cloud NAT handles egress
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false   # keep public endpoint for kubectl
    master_ipv4_cidr_block  = var.master_ipv4_cidr_block
  }

  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = "0.0.0.0/0"
      display_name = "all — restrict this in production"
    }
  }

  description = "GKE Standard cluster for PHM C-MAPSS turbofan RUL pipeline"

  depends_on = [
    google_project_service.apis,
    google_compute_subnetwork.phm_subnet,
    google_compute_router_nat.phm_nat,
  ]

  lifecycle {
    ignore_changes = [node_config]
  }
}

resource "google_container_node_pool" "primary" {
  name       = "primary-pool"
  location   = var.region
  cluster    = google_container_cluster.phm_gke.name

  initial_node_count = var.node_min_count

  autoscaling {
    min_node_count = var.node_min_count
    max_node_count = var.node_max_count
  }

  node_config {
    machine_type = var.machine_type   # e2-standard-4: 4 vCPU, 16 GB RAM

    disk_size_gb = var.node_disk_size_gb
    disk_type    = var.node_disk_type

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]

    labels = {
      workload = "general"
      env      = "phm"
    }

    tags = ["gke-${var.cluster_name}"]
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# ---------------------------------------------------------------
# Step 4 — Artifact Registry
# depends_on: artifactregistry API must be active
# ---------------------------------------------------------------

resource "google_artifact_registry_repository" "phm_repos" {
  for_each = toset([
    "data-ingestion",
    "validation-service",
    "stream-processor",
    "model-training",
    "model-registry",
    "model-serving",
    "alert-engine",
  ])

  location      = var.region
  repository_id = "phm-${each.key}"
  format        = "DOCKER"
  description   = "PHM ${each.key} service images"

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------
# Step 4 — GCS Buckets
# depends_on: storage API must be active
# ---------------------------------------------------------------

resource "google_storage_bucket" "phm_buckets" {
  for_each = {
    raw_data        = "phm-raw-data"
    model_artifacts = "phm-model-artifacts"
    mlflow          = "phm-mlflow-artifacts"
    data_quality    = "phm-data-quality"
  }

  name          = "${each.value}-${var.project_id}"
  location      = var.region
  storage_class = "STANDARD"
  force_destroy = var.gcs_force_destroy

  versioning {
    enabled = each.key == "model_artifacts" ? true : false
  }

  public_access_prevention    = "enforced"
  uniform_bucket_level_access = true

  labels = {
    project = var.project
    env     = "phm"
  }

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------
# Step 5 — Cloud SQL + Memorystore Redis
# depends_on: VPC peering must be established first
# ---------------------------------------------------------------

resource "google_sql_database_instance" "phm_pg" {
  name             = "${var.project}-pg"
  database_version = "POSTGRES_15"
  region           = var.region
  deletion_protection = false   # set true in production

  settings {
    tier              = var.db_tier
    availability_type = "ZONAL"

    disk_size       = var.db_disk_size_gb
    disk_type       = "PD_SSD"
    disk_autoresize = true

    backup_configuration {
      enabled    = true
      start_time = "03:00"
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.phm_vpc.id
      require_ssl     = false
    }

    database_flags {
      name  = "max_connections"
      value = "100"
    }
  }

  depends_on = [
    google_project_service.apis,
    google_service_networking_connection.phm_vpc_peering,
  ]
}

resource "google_sql_database" "phmdb" {
  name     = var.db_name
  instance = google_sql_database_instance.phm_pg.name
}

resource "google_sql_user" "phmadmin" {
  name     = var.db_username
  instance = google_sql_database_instance.phm_pg.name
  password = random_password.db.result
}

resource "google_redis_instance" "phm_redis" {
  name           = "${var.project}-redis"
  tier           = "BASIC"
  memory_size_gb = var.redis_memory_size_gb
  region         = var.region

  authorized_network = google_compute_network.phm_vpc.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"

  redis_version = "REDIS_7_0"
  display_name  = "PHM online feature store"

  labels = {
    project = var.project
    env     = "phm"
  }

  depends_on = [
    google_project_service.apis,
    google_service_networking_connection.phm_vpc_peering,
  ]
}

# ---------------------------------------------------------------
# Step 6 — Google Service Accounts + Workload Identity bindings
#
# GSA creation depends on: IAM API active
# WI bindings depend on: GKE cluster fully created
#   (cluster creation initialises the PROJECT_ID.svc.id.goog
#    identity pool — binding before it exists causes the
#    "Identity Pool does not exist" 400 error)
# ---------------------------------------------------------------

# --- data-ingestion ---
resource "google_service_account" "data_ingestion" {
  account_id   = "${var.project}-data-ingestion"
  display_name = "PHM data-ingestion service account"
  depends_on   = [google_project_service.apis]
}

resource "google_storage_bucket_iam_member" "data_ingestion_raw" {
  bucket = google_storage_bucket.phm_buckets["raw_data"].name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.data_ingestion.email}"
}

resource "google_storage_bucket_iam_member" "data_ingestion_quality" {
  bucket = google_storage_bucket.phm_buckets["data_quality"].name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.data_ingestion.email}"
}

resource "google_service_account_iam_member" "data_ingestion_wi" {
  service_account_id = google_service_account.data_ingestion.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[data-ingestion/data-ingestion-sa]"

  # GKE cluster must be fully created before the WI pool exists
  depends_on = [google_container_cluster.phm_gke]
}

# --- feature-platform ---
resource "google_service_account" "feature_platform" {
  account_id   = "${var.project}-feature-platform"
  display_name = "PHM feature-platform service account"
  depends_on   = [google_project_service.apis]
}

resource "google_storage_bucket_iam_member" "feature_platform_raw" {
  bucket = google_storage_bucket.phm_buckets["raw_data"].name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.feature_platform.email}"
}

resource "google_service_account_iam_member" "feature_platform_wi" {
  service_account_id = google_service_account.feature_platform.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[feature-platform/feature-platform-sa]"
  depends_on         = [google_container_cluster.phm_gke]
}

# --- model-training ---
resource "google_service_account" "model_training" {
  account_id   = "${var.project}-model-training"
  display_name = "PHM model-training service account"
  depends_on   = [google_project_service.apis]
}

resource "google_storage_bucket_iam_member" "model_training_raw" {
  bucket = google_storage_bucket.phm_buckets["raw_data"].name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.model_training.email}"
}

resource "google_storage_bucket_iam_member" "model_training_artifacts" {
  bucket = google_storage_bucket.phm_buckets["model_artifacts"].name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.model_training.email}"
}

resource "google_storage_bucket_iam_member" "model_training_mlflow" {
  bucket = google_storage_bucket.phm_buckets["mlflow"].name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.model_training.email}"
}

resource "google_service_account_iam_member" "model_training_wi" {
  service_account_id = google_service_account.model_training.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[model-training/model-training-sa]"
  depends_on         = [google_container_cluster.phm_gke]
}

# --- model-serving ---
resource "google_service_account" "model_serving" {
  account_id   = "${var.project}-model-serving"
  display_name = "PHM model-serving service account"
  depends_on   = [google_project_service.apis]
}

resource "google_storage_bucket_iam_member" "model_serving_artifacts" {
  bucket = google_storage_bucket.phm_buckets["model_artifacts"].name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.model_serving.email}"
}

resource "google_storage_bucket_iam_member" "model_serving_raw" {
  bucket = google_storage_bucket.phm_buckets["raw_data"].name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.model_serving.email}"
}

resource "google_service_account_iam_member" "model_serving_wi" {
  service_account_id = google_service_account.model_serving.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[model-serving/model-serving-sa]"
  depends_on         = [google_container_cluster.phm_gke]
}

# --- alert-engine ---
resource "google_service_account" "alert_engine" {
  account_id   = "${var.project}-alert-engine"
  display_name = "PHM alert-engine service account"
  depends_on   = [google_project_service.apis]
}

resource "google_storage_bucket_iam_member" "alert_engine_raw" {
  bucket = google_storage_bucket.phm_buckets["raw_data"].name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.alert_engine.email}"
}

resource "google_service_account_iam_member" "alert_engine_wi" {
  service_account_id = google_service_account.alert_engine.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[alert-naming/alert-engine-sa]"
  depends_on         = [google_container_cluster.phm_gke]
}
