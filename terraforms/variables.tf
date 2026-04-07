# -------------------------------
# Project / Location
# -------------------------------

variable "project" {
  type        = string
  description = "Short project prefix applied to every resource name"
  default     = "phm"
}

variable "aws_region" {
  type        = string
  description = "AWS region"
  default     = "ap-southeast-1"
}

# -------------------------------
# Networking
# -------------------------------

variable "vpc_cidr" {
  type        = string
  description = "CIDR block for the VPC"
  default     = "10.10.0.0/16"
}

variable "availability_zone" {
  type        = string
  description = "Primary AZ — keeps EKS and Redis in one zone to avoid cross-AZ transfer costs"
  default     = "ap-southeast-1a"
}

variable "rds_availability_zone" {
  type        = string
  description = "Second AZ used only for the RDS subnet group (AWS requires minimum 2 AZs)"
  default     = "ap-southeast-1b"
}

variable "public_subnet_cidr" {
  type        = string
  description = "CIDR for the public subnet (EKS API LB)"
  default     = "10.10.1.0/24"
}

variable "private_subnet_cidr" {
  type        = string
  description = "CIDR for the private subnet (EKS worker node, RDS, Redis)"
  default     = "10.10.2.0/24"
}

# -------------------------------
# EKS cluster
# -------------------------------

variable "cluster_name" {
  type        = string
  description = "Name of the EKS cluster"
  default     = "eks-phm"
}

variable "kubernetes_version" {
  type        = string
  description = "Kubernetes version for EKS"
  default     = "1.29"
}

# -------------------------------
# Node group sizing / type
# -------------------------------

variable "node_instance_type" {
  type        = string
  description = "EC2 instance type — t3.medium: 2 vCPU, 4 GB RAM, free-tier eligible"
  default     = "t3.small"
}

variable "node_count" {
  type        = number
  description = "Fixed number of worker nodes — no auto-scaling"
  default     = 1
}

variable "node_disk_size_gb" {
  type        = number
  description = "EBS root volume size per node in GiB"
  default     = 20
}

# -------------------------------
# RDS PostgreSQL (feature store + MLflow backend)
# -------------------------------

variable "db_instance_class" {
  type        = string
  description = "RDS instance class — db.t3.micro is free-tier eligible"
  default     = "db.t3.micro"
}

variable "db_allocated_storage" {
  type        = number
  description = "Allocated storage in GiB — 20 GiB is within free-tier limit"
  default     = 20
}

variable "db_name" {
  type        = string
  description = "Initial database name"
  default     = "phmdb"
}

variable "db_username" {
  type        = string
  description = "Master username for RDS"
  default     = "phmadmin"
}

# -------------------------------
# ElastiCache Redis (online feature store)
# -------------------------------

variable "redis_node_type" {
  type        = string
  description = "ElastiCache node type — cache.t3.micro is the smallest available"
  default     = "cache.t3.micro"
}

# -------------------------------
# S3
# -------------------------------

variable "s3_force_destroy" {
  type        = bool
  description = "Allow terraform destroy to delete non-empty buckets (safe for dev)"
  default     = true
}
