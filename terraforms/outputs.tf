# -------------------------------
# EKS
# -------------------------------

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = aws_eks_cluster.main.name
}

output "cluster_endpoint" {
  description = "Public endpoint of the EKS control plane"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_ca_certificate" {
  description = "Cluster CA certificate (base64)"
  value       = aws_eks_cluster.main.certificate_authority[0].data
  sensitive   = true
}

output "aws_get_credentials_command" {
  description = "Run this to configure kubectl"
  value       = "aws eks update-kubeconfig --name ${aws_eks_cluster.main.name} --region ${var.aws_region}"
}

output "eks_oidc_issuer_url" {
  description = "OIDC issuer URL — referenced by IRSA trust policies"
  value       = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

# -------------------------------
# Networking
# -------------------------------

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "private_subnet_id" {
  description = "Private subnet ID — EKS node, RDS, Redis"
  value       = aws_subnet.private.id
}

# -------------------------------
# RDS
# -------------------------------

output "db_endpoint" {
  description = "RDS PostgreSQL endpoint (host:port)"
  value       = aws_db_instance.main.endpoint
}

output "db_name" {
  description = "Database name"
  value       = aws_db_instance.main.db_name
}

output "db_username" {
  description = "Database master username"
  value       = aws_db_instance.main.username
}

output "db_password" {
  description = "Auto-generated DB password — retrieve with: terraform output -raw db_password"
  value       = random_password.db.result
  sensitive   = true
}

# -------------------------------
# ElastiCache Redis
# -------------------------------

output "redis_endpoint" {
  description = "Redis cache endpoint"
  value       = aws_elasticache_cluster.main.cache_nodes[0].address
}

output "redis_port" {
  description = "Redis port"
  value       = aws_elasticache_cluster.main.cache_nodes[0].port
}

# -------------------------------
# S3
# -------------------------------

output "s3_raw_data_bucket" {
  description = "S3 bucket for raw C-MAPSS sensor data"
  value       = aws_s3_bucket.main["raw_data"].id
}

output "s3_model_artifacts_bucket" {
  description = "S3 bucket for trained model binaries"
  value       = aws_s3_bucket.main["model_artifacts"].id
}

output "s3_mlflow_bucket" {
  description = "S3 bucket used as MLflow artifact store"
  value       = aws_s3_bucket.main["mlflow"].id
}

output "s3_data_quality_bucket" {
  description = "S3 bucket for invalid records"
  value       = aws_s3_bucket.main["data_quality"].id
}

# -------------------------------
# ECR
# -------------------------------

output "ecr_repository_urls" {
  description = "Map of service name → ECR repository URL (used in Jenkinsfile)"
  value       = { for k, r in aws_ecr_repository.main : k => r.repository_url }
}

# -------------------------------
# IAM / IRSA
# -------------------------------

output "irsa_role_model_training" {
  description = "IRSA role ARN — annotate model-training ServiceAccount with this"
  value       = aws_iam_role.model_training.arn
}

output "irsa_role_model_serving" {
  description = "IRSA role ARN — annotate model-serving ServiceAccount with this"
  value       = aws_iam_role.model_serving.arn
}

output "irsa_role_data_ingestion" {
  description = "IRSA role ARN — annotate data-ingestion ServiceAccount with this"
  value       = aws_iam_role.data_ingestion.arn
}

output "irsa_role_feature_platform" {
  value = aws_iam_role.feature_platform.arn
}