terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.30"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  # Remote backend — state lives in S3, not on your local machine.
  # Safe to interrupt terraform apply at any time: the next run reads
  # the real state from S3 and picks up exactly where it left off.
  #
  # One-time bootstrap before first terraform init:
  #   aws s3 mb s3://phm-tf-state-YOURACCOUNTID --region ap-southeast-1
  #   aws s3api put-bucket-versioning \
  #     --bucket phm-tf-state-YOURACCOUNTID \
  #     --versioning-configuration Status=Enabled
  #
  # Replace the bucket value below with your actual bucket name.
  backend "s3" {
    bucket  = "phm-tf-state-982215135701"
    key     = "phm/terraform.tfstate"
    region  = "ap-southeast-1"
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "terraform"
    }
  }
}

locals {
  name = var.project
}

data "aws_caller_identity" "current" {}

# DB password auto-generated — no prompt needed on plan/apply
# Retrieve after apply: terraform output -raw db_password
resource "random_password" "db" {
  length           = 16
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

# -------------------------------
# Networking: VPC + Subnets
# -------------------------------

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = { Name = "${local.name}-vpc" }
}

# Public subnet — used by the EKS load balancer endpoint
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidr
  availability_zone       = var.availability_zone
  map_public_ip_on_launch = true

  tags = {
    Name                     = "${local.name}-public-subnet"
    "kubernetes.io/role/elb" = "1"
  }
}

# Private subnet — EKS worker node, RDS, Redis (no public IP)
resource "aws_subnet" "private" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidr
  availability_zone = var.availability_zone

  tags = {
    Name                              = "${local.name}-private-subnet"
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# Spare subnet in a second AZ — required by RDS subnet group (AWS mandates >= 2 AZs)
# No resources are launched here; it exists only to satisfy the RDS constraint.
resource "aws_subnet" "rds_spare" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.10.3.0/24"
  availability_zone = var.rds_availability_zone

  tags = { Name = "${local.name}-rds-spare-subnet" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${local.name}-igw" }
}

# Single NAT gateway — one AZ, one EIP
resource "aws_eip" "nat" {
  domain = "vpc"
  tags   = { Name = "${local.name}-nat-eip" }
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public.id
  depends_on    = [aws_internet_gateway.main]
  tags          = { Name = "${local.name}-nat" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  tags = { Name = "${local.name}-public-rt" }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }
  tags = { Name = "${local.name}-private-rt" }
}

resource "aws_route_table_association" "private" {
  subnet_id      = aws_subnet.private.id
  route_table_id = aws_route_table.private.id
}

# -------------------------------
# Security Groups
# -------------------------------

resource "aws_security_group" "eks_nodes" {
  name        = "${local.name}-eks-nodes-sg"
  description = "EKS worker node - allows all egress, controlled ingress from cluster"
  vpc_id      = aws_vpc.main.id

  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name}-eks-nodes-sg" }
}

resource "aws_security_group" "rds" {
  name        = "${local.name}-rds-sg"
  description = "RDS PostgreSQL - reachable only from EKS worker"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "PostgreSQL from EKS workers"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }

  ingress {
    description = "PostgreSQL from EKS cluster SG"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name}-rds-sg" }
}

resource "aws_security_group" "redis" {
  name        = "${local.name}-redis-sg"
  description = "ElastiCache Redis - reachable only from EKS worker"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Redis from EKS workers"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }

  ingress {
    description = "Redis from EKS cluster SG"
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name}-redis-sg" }
}

# -------------------------------
# EKS Cluster
# -------------------------------

resource "aws_iam_role" "eks_cluster" {
  name = "${local.name}-eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "eks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  role       = aws_iam_role.eks_cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

resource "aws_eks_cluster" "main" {
  name     = var.cluster_name
  role_arn = aws_iam_role.eks_cluster.arn
  version  = var.kubernetes_version

  vpc_config {
    subnet_ids              = [aws_subnet.public.id, aws_subnet.private.id, aws_subnet.rds_spare.id]
    security_group_ids      = [aws_security_group.eks_nodes.id]
    endpoint_private_access = true
    endpoint_public_access  = true
  }

  # api + audit only — minimises CloudWatch cost
  enabled_cluster_log_types = ["api", "audit"]

  tags       = { Name = var.cluster_name }
  depends_on = [aws_iam_role_policy_attachment.eks_cluster_policy]
}

# OIDC provider — required for IRSA (pod-level AWS permissions)
data "tls_certificate" "eks" {
  url = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

# -------------------------------
# EKS Node Group (fixed size, no auto-scaling)
# -------------------------------

resource "aws_iam_role" "eks_node" {
  name = "${local.name}-eks-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action    = "sts:AssumeRole"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_node_worker" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "eks_node_ecr" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "eks_node_cni" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_eks_node_group" "primary" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "primary-pool"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = [aws_subnet.private.id]
  instance_types  = [var.node_instance_type]

  # desired = min = max — completely fixed, no auto-scaling
  scaling_config {
    desired_size = var.node_count
    min_size     = var.node_count
    max_size     = var.node_count
  }

  disk_size = var.node_disk_size_gb

  update_config { max_unavailable = 1 }

  labels = {
    workload = "general"
    env      = var.project
  }

  lifecycle {
    ignore_changes = [scaling_config[0].desired_size]
  }

  tags = { Name = "${local.name}-primary-pool" }

  depends_on = [
    aws_iam_role_policy_attachment.eks_node_worker,
    aws_iam_role_policy_attachment.eks_node_ecr,
    aws_iam_role_policy_attachment.eks_node_cni,
  ]
}

# Essential add-ons only
resource "aws_eks_addon" "vpc_cni" {
  cluster_name                = aws_eks_cluster.main.name
  addon_name                  = "vpc-cni"
  resolve_conflicts_on_update = "OVERWRITE"
}

resource "aws_eks_addon" "coredns" {
  cluster_name                = aws_eks_cluster.main.name
  addon_name                  = "coredns"
  resolve_conflicts_on_update = "OVERWRITE"
  depends_on                  = [aws_eks_node_group.primary]
}

resource "aws_eks_addon" "kube_proxy" {
  cluster_name                = aws_eks_cluster.main.name
  addon_name                  = "kube-proxy"
  resolve_conflicts_on_update = "OVERWRITE"
}

# -------------------------------
# ECR — one repository per service
# -------------------------------

locals {
  ecr_services = toset([
    "data-ingestion",
    "validation-service",
    "stream-processor",
    "model-training",
    "model-registry",
    "model-serving",
    "alert-engine",
    "streamlit-dashboard",
  ])
}

resource "aws_ecr_repository" "main" {
  for_each             = local.ecr_services
  name                 = "${local.name}/${each.value}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration { scan_on_push = true }

  tags = { Service = each.value }
}

resource "aws_ecr_lifecycle_policy" "main" {
  for_each   = local.ecr_services
  repository = aws_ecr_repository.main[each.value].name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only last 5 images to save storage"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

# -------------------------------
# S3 — four buckets (AES256, no KMS cost)
# -------------------------------

locals {
  s3_buckets = {
    model_artifacts = "${local.name}-model-artifacts"
    mlflow          = "${local.name}-mlflow-artifacts"
    raw_data        = "${local.name}-raw-data"
    data_quality    = "${local.name}-data-quality"
  }
}

resource "aws_s3_bucket" "main" {
  for_each      = local.s3_buckets
  bucket        = each.value
  force_destroy = var.s3_force_destroy
  tags          = { Name = each.value, Purpose = each.key }
}

resource "aws_s3_bucket_versioning" "main" {
  for_each = local.s3_buckets
  bucket   = aws_s3_bucket.main[each.key].id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "main" {
  for_each = local.s3_buckets
  bucket   = aws_s3_bucket.main[each.key].id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"   # free — no KMS cost
    }
  }
}

resource "aws_s3_bucket_public_access_block" "main" {
  for_each                = local.s3_buckets
  bucket                  = aws_s3_bucket.main[each.key].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# -------------------------------
# RDS PostgreSQL 15 — free-tier eligible
# db.t3.micro · 20 GiB gp2 · single-AZ · auto-generated password
# -------------------------------

resource "aws_db_subnet_group" "main" {
  name       = "${local.name}-rds-subnet-group"
  subnet_ids = [aws_subnet.private.id, aws_subnet.rds_spare.id]
  tags       = { Name = "${local.name}-rds-subnet-group" }
}

resource "aws_db_instance" "main" {
  identifier        = "${local.name}-db"
  engine            = "postgres"
  engine_version    = "15"
  instance_class    = var.db_instance_class
  allocated_storage = var.db_allocated_storage
  storage_type      = "gp2"          # gp2 is free-tier eligible

  db_name  = var.db_name
  username = var.db_username
  password = random_password.db.result  # auto-generated, no prompt

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  multi_az                = false    # single-AZ — free tier
  publicly_accessible     = false
  deletion_protection     = false
  skip_final_snapshot     = true
  backup_retention_period = 1
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"

  tags = { Name = "${local.name}-db" }
}

# -------------------------------
# ElastiCache Redis — online feature store (rolling 24-cycle window)
# Single node, no replication — free-tier friendly
# -------------------------------

resource "aws_elasticache_subnet_group" "main" {
  name       = "${local.name}-redis-subnet-group"
  subnet_ids = [aws_subnet.private.id]
  tags       = { Name = "${local.name}-redis-subnet-group" }
}

resource "aws_elasticache_cluster" "main" {
  cluster_id           = "${local.name}-redis"
  engine               = "redis"
  engine_version       = "7.0"
  node_type            = var.redis_node_type
  num_cache_nodes      = 1              # single node — no replication
  parameter_group_name = "default.redis7"
  port                 = 6379

  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]

  snapshot_retention_limit = 0          # disable snapshots — saves storage cost
  maintenance_window       = "sun:05:00-sun:06:00"

  tags = { Name = "${local.name}-redis" }
}

# -------------------------------
# IAM / IRSA — pod-level AWS permissions (no node-level credentials)
# -------------------------------

locals {
  oidc_issuer = replace(aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")
}

# model-training — reads raw data, writes artifacts + MLflow
resource "aws_iam_role" "model_training" {
  name = "${local.name}-model-training-irsa"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect  = "Allow"
      Action  = "sts:AssumeRoleWithWebIdentity"
      Principal = { Federated = aws_iam_openid_connect_provider.eks.arn }
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:sub" = "system:serviceaccount:model-training:model-training-sa"
          "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "model_training" {
  name = "${local.name}-model-training-policy"
  role = aws_iam_role.model_training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadRawData"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.main["raw_data"].arn,
          "${aws_s3_bucket.main["raw_data"].arn}/*",
        ]
      },
      {
        Sid    = "WriteArtifacts"
        Effect = "Allow"
        Action = ["s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.main["model_artifacts"].arn,
          "${aws_s3_bucket.main["model_artifacts"].arn}/*",
          aws_s3_bucket.main["mlflow"].arn,
          "${aws_s3_bucket.main["mlflow"].arn}/*",
        ]
      }
    ]
  })
}

# model-serving — reads artifacts only
resource "aws_iam_role" "model_serving" {
  name = "${local.name}-model-serving-irsa"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect  = "Allow"
      Action  = "sts:AssumeRoleWithWebIdentity"
      Principal = { Federated = aws_iam_openid_connect_provider.eks.arn }
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:sub" = "system:serviceaccount:model-serving:model-serving-sa"
          "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "model_serving" {
  name = "${local.name}-model-serving-policy"
  role = aws_iam_role.model_serving.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "ReadArtifacts"
      Effect = "Allow"
      Action = ["s3:GetObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.main["model_artifacts"].arn,
        "${aws_s3_bucket.main["model_artifacts"].arn}/*",
      ]
    }]
  })
}

# data-ingestion role — reads simulated file + writes raw events to S3
resource "aws_iam_role" "data_ingestion" {
  name = "${local.name}-data-ingestion-irsa"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect  = "Allow"
      Action  = "sts:AssumeRoleWithWebIdentity"
      Principal = { Federated = aws_iam_openid_connect_provider.eks.arn }
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:sub" = "system:serviceaccount:data-ingestion:data-ingestion-sa"
          "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "data_ingestion" {
  name = "${local.name}-data-ingestion-policy"
  role = aws_iam_role.data_ingestion.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "KafkaIAMAuth"
        Effect = "Allow"
        Action = ["kafka-cluster:Connect", "kafka-cluster:DescribeCluster",
                  "kafka-cluster:WriteData", "kafka-cluster:ReadData",
                  "kafka-cluster:DescribeTopic", "kafka-cluster:CreateTopic",
                  "kafka-cluster:AlterGroup", "kafka-cluster:DescribeGroup"]
        Resource = [
          "arn:aws:kafka:${var.aws_region}:${data.aws_caller_identity.current.account_id}:cluster/${local.name}-kafka/*",
          "arn:aws:kafka:${var.aws_region}:${data.aws_caller_identity.current.account_id}:topic/${local.name}-kafka/*",
          "arn:aws:kafka:${var.aws_region}:${data.aws_caller_identity.current.account_id}:group/${local.name}-kafka/*",
        ]
      },
      {
        Sid    = "S3FullAccess"
        Effect = "Allow"
        Action = ["s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.main["raw_data"].arn,
          "${aws_s3_bucket.main["raw_data"].arn}/*",
          aws_s3_bucket.main["data_quality"].arn,
          "${aws_s3_bucket.main["data_quality"].arn}/*",
        ]
      }
    ]
  })
}

resource "aws_iam_role" "feature_platform" {
  name = "${local.name}-feature-platform-irsa"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect  = "Allow"
      Action  = "sts:AssumeRoleWithWebIdentity"
      Principal = { Federated = aws_iam_openid_connect_provider.eks.arn }
      Condition = {
        StringEquals = {
          "${local.oidc_issuer}:sub" = "system:serviceaccount:feature-platform:feature-platform-sa"
          "${local.oidc_issuer}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "feature_platform" {
  name = "${local.name}-feature-platform-policy"
  role = aws_iam_role.feature_platform.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "S3ReadValidated"
      Effect = "Allow"
      Action = ["s3:GetObject", "s3:ListBucket", "s3:PutObject"]
      Resource = [
        aws_s3_bucket.main["raw_data"].arn,
        "${aws_s3_bucket.main["raw_data"].arn}/*",
      ]
    }]
  })
}