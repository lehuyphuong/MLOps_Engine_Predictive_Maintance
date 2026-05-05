# ---------------------------------------------------------------
# Project / Location
# ---------------------------------------------------------------

variable "project_id" {
  type        = string
  description = "GCP project ID"
  default     = "aide2-494008"
}

variable "project" {
  type        = string
  description = "Short prefix applied to every resource name"
  default     = "phm"
}

variable "region" {
  type        = string
  description = "GCP region for all resources"
  default     = "us-central1"
}

# ---------------------------------------------------------------
# Networking
# ---------------------------------------------------------------

variable "subnet_ip_cidr" {
  type        = string
  description = "Primary CIDR for GKE nodes"
  default     = "10.10.0.0/20"
}

variable "pods_secondary_range_name" {
  type        = string
  description = "Secondary IP range name for Pods"
  default     = "gke-pods"
}

variable "pods_ip_cidr" {
  type        = string
  description = "CIDR range for Pods"
  default     = "10.20.0.0/16"
}

variable "services_secondary_range_name" {
  type        = string
  description = "Secondary IP range name for Services"
  default     = "gke-services"
}

variable "services_ip_cidr" {
  type        = string
  description = "CIDR range for Services"
  default     = "10.30.0.0/20"
}

variable "master_ipv4_cidr_block" {
  type        = string
  description = "Private CIDR for the GKE control plane (must not overlap node or pod ranges)"
  default     = "172.16.0.0/28"
}

# ---------------------------------------------------------------
# GKE cluster
# ---------------------------------------------------------------

variable "cluster_name" {
  type        = string
  description = "Name of the GKE cluster"
  default     = "gke-phm"
}

variable "release_channel" {
  type        = string
  description = "GKE release channel (RAPID, REGULAR, STABLE)"
  default     = "REGULAR"
}

variable "node_locations" {
  type        = list(string)
  description = "Zones where GKE nodes run (single zone keeps costs low for dev)"
  default     = ["us-central1-a"]
}

# ---------------------------------------------------------------
# Node pool
# ---------------------------------------------------------------

variable "machine_type" {
  type        = string
  description = "GCE machine type for GKE nodes — e2-standard-4: 4 vCPU, 16 GB RAM"
  default     = "e2-standard-4"
}

variable "node_min_count" {
  type        = number
  description = "Minimum nodes in the primary pool"
  default     = 1
}

variable "node_max_count" {
  type        = number
  description = "Maximum nodes in the primary pool"
  default     = 2
}

variable "node_disk_size_gb" {
  type        = number
  description = "Boot disk size per node in GB"
  default     = 50
}

variable "node_disk_type" {
  type        = string
  description = "Boot disk type (pd-standard, pd-balanced, pd-ssd)"
  default     = "pd-balanced"
}

# ---------------------------------------------------------------
# Cloud SQL — PostgreSQL
# ---------------------------------------------------------------

variable "db_tier" {
  type        = string
  description = "Cloud SQL machine tier — db-f1-micro (0.6 GB) for dev, db-g1-small (1.7 GB) for more headroom"
  default     = "db-g1-small"
}

variable "db_disk_size_gb" {
  type        = number
  description = "Cloud SQL disk size in GB"
  default     = 20
}

variable "db_name" {
  type        = string
  description = "Initial database name"
  default     = "phmdb"
}

variable "db_username" {
  type        = string
  description = "PostgreSQL master username"
  default     = "phmadmin"
}

# ---------------------------------------------------------------
# Memorystore Redis
# ---------------------------------------------------------------

variable "redis_memory_size_gb" {
  type        = number
  description = "Redis instance memory in GB — 1 GB is sufficient for the 24-cycle feature window"
  default     = 1
}

# ---------------------------------------------------------------
# GCS
# ---------------------------------------------------------------

variable "gcs_force_destroy" {
  type        = bool
  description = "Allow terraform destroy to delete non-empty GCS buckets (safe for dev)"
  default     = true
}
