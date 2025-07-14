# MLOps Infrastructure with Terraform
# Production-ready cloud infrastructure for MLOps pipeline

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }

  backend "s3" {
    bucket = "mlops-terraform-state"
    key    = "mlops-pipeline/terraform.tfstate"
    region = "us-west-2"
  }
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "mlops-pipeline"
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "cluster_version" {
  description = "Kubernetes cluster version"
  type        = string
  default     = "1.27"
}

variable "node_instance_types" {
  description = "EC2 instance types for worker nodes"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge"]
}

variable "min_nodes" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 2
}

variable "max_nodes" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 10
}

variable "desired_nodes" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 3
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# Local values
locals {
  name_prefix = "${var.project_name}-${var.environment}"
  
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Owner       = "mlops-team"
  }
}

# Provider configurations
provider "aws" {
  region = var.region
  
  default_tags {
    tags = local.common_tags
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

# Data source for EKS cluster auth
data "aws_eks_cluster_auth" "cluster" {
  name = module.eks.cluster_name
}

# VPC Module
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.name_prefix}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.region}a", "${var.region}b", "${var.region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true

  # Additional tags for Kubernetes
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }

  tags = merge(local.common_tags, {
    "kubernetes.io/cluster/${local.name_prefix}-eks" = "owned"
  })
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "${local.name_prefix}-eks"
  cluster_version = var.cluster_version

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    main = {
      name = "${local.name_prefix}-nodes"

      instance_types = var.node_instance_types
      capacity_type  = "ON_DEMAND"

      min_size     = var.min_nodes
      max_size     = var.max_nodes
      desired_size = var.desired_nodes

      # Launch template configuration
      create_launch_template = false
      launch_template_name   = ""

      disk_size = 50
      ami_type  = "AL2_x86_64"

      # Remote access
      remote_access = {
        ec2_ssh_key               = aws_key_pair.eks_nodes.key_name
        source_security_group_ids = [aws_security_group.remote_access.id]
      }

      labels = {
        Environment = var.environment
        NodeGroup   = "main"
      }

      taints = []

      tags = local.common_tags
    }

    # GPU nodes for ML workloads
    gpu = {
      name = "${local.name_prefix}-gpu-nodes"

      instance_types = ["g4dn.xlarge"]
      capacity_type  = "SPOT"

      min_size     = 0
      max_size     = 5
      desired_size = 0

      ami_type = "AL2_x86_64_GPU"
      disk_size = 100

      labels = {
        Environment = var.environment
        NodeGroup   = "gpu"
        WorkloadType = "ml-training"
      }

      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]

      tags = merge(local.common_tags, {
        "k8s.io/cluster-autoscaler/node-template/label/WorkloadType" = "ml-training"
      })
    }
  }

  # Cluster access entry
  access_entries = {
    admin = {
      principal_arn     = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
      policy_associations = {
        admin = {
          policy_arn = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
          access_scope = {
            type = "cluster"
          }
        }
      }
    }
  }

  tags = local.common_tags
}

# Key pair for EKS nodes
resource "aws_key_pair" "eks_nodes" {
  key_name   = "${local.name_prefix}-eks-nodes"
  public_key = file("~/.ssh/id_rsa.pub") # You may need to generate this

  tags = local.common_tags
}

# Security group for remote access
resource "aws_security_group" "remote_access" {
  name_prefix = "${local.name_prefix}-remote-access"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port = 22
    to_port   = 22
    protocol  = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }

  tags = local.common_tags
}

# RDS Instance for MLflow
resource "aws_db_subnet_group" "mlflow" {
  name       = "${local.name_prefix}-mlflow"
  subnet_ids = module.vpc.private_subnets

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-mlflow-db-subnet-group"
  })
}

resource "aws_security_group" "rds" {
  name_prefix = "${local.name_prefix}-rds"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

resource "aws_db_instance" "mlflow" {
  identifier = "${local.name_prefix}-mlflow"

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.micro"

  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = "mlflow"
  username = "mlflow"
  password = random_password.db_password.result

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.mlflow.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = true
  deletion_protection = false

  tags = local.common_tags
}

resource "random_password" "db_password" {
  length  = 16
  special = true
}

# ElastiCache for Redis
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${local.name_prefix}-redis"
  subnet_ids = module.vpc.private_subnets

  tags = local.common_tags
}

resource "aws_security_group" "redis" {
  name_prefix = "${local.name_prefix}-redis"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id         = "${local.name_prefix}-redis"
  description                  = "Redis for MLOps pipeline"

  port               = 6379
  parameter_group_name = "default.redis7"
  node_type          = "cache.t3.micro"
  
  num_cache_clusters = 2
  
  subnet_group_name  = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.redis_auth.result

  tags = local.common_tags
}

resource "random_password" "redis_auth" {
  length  = 32
  special = false
}

# S3 Buckets
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${local.name_prefix}-mlflow-artifacts-${random_id.bucket_suffix.hex}"

  tags = local.common_tags
}

resource "aws_s3_bucket" "data_storage" {
  bucket = "${local.name_prefix}-data-storage-${random_id.bucket_suffix.hex}"

  tags = local.common_tags
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "data_storage" {
  bucket = aws_s3_bucket.data_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_storage" {
  bucket = aws_s3_bucket.data_storage.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# IAM Roles and Policies
resource "aws_iam_role" "mlops_service_role" {
  name = "${local.name_prefix}-mlops-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      },
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${module.eks.oidc_provider}:sub" = "system:serviceaccount:mlops:mlops-service"
            "${module.eks.oidc_provider}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_policy" "mlops_service_policy" {
  name = "${local.name_prefix}-mlops-service-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*",
          aws_s3_bucket.data_storage.arn,
          "${aws_s3_bucket.data_storage.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "rds:DescribeDBInstances",
          "rds:DescribeDBClusters"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "elasticache:DescribeReplicationGroups",
          "elasticache:DescribeCacheClusters"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:*"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "mlops_service_policy" {
  role       = aws_iam_role.mlops_service_role.name
  policy_arn = aws_iam_policy.mlops_service_policy.arn
}

# Application Load Balancer
resource "aws_lb" "mlops" {
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets

  enable_deletion_protection = false

  tags = local.common_tags
}

resource "aws_security_group" "alb" {
  name_prefix = "${local.name_prefix}-alb"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = local.common_tags
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "mlops" {
  name              = "/aws/eks/${local.name_prefix}/cluster"
  retention_in_days = 14

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "mlops_applications" {
  name              = "/aws/mlops/${local.name_prefix}/applications"
  retention_in_days = 7

  tags = local.common_tags
}

# Kubernetes Namespaces and RBAC
resource "kubernetes_namespace" "mlops" {
  metadata {
    name = "mlops"
    labels = {
      name        = "mlops"
      environment = var.environment
    }
  }

  depends_on = [module.eks]
}

resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = "monitoring"
    labels = {
      name        = "monitoring"
      environment = var.environment
    }
  }

  depends_on = [module.eks]
}

resource "kubernetes_service_account" "mlops_service" {
  metadata {
    name      = "mlops-service"
    namespace = kubernetes_namespace.mlops.metadata[0].name
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.mlops_service_role.arn
    }
  }

  depends_on = [module.eks]
}

# Helm Charts
resource "helm_release" "aws_load_balancer_controller" {
  name       = "aws-load-balancer-controller"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  namespace  = "kube-system"
  version    = "1.6.2"

  set {
    name  = "clusterName"
    value = module.eks.cluster_name
  }

  set {
    name  = "serviceAccount.create"
    value = "false"
  }

  set {
    name  = "serviceAccount.name"
    value = "aws-load-balancer-controller"
  }

  depends_on = [module.eks]
}

resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = kubernetes_namespace.monitoring.metadata[0].name
  version    = "51.2.0"

  values = [
    file("${path.module}/helm-values/prometheus-values.yaml")
  ]

  depends_on = [module.eks, kubernetes_namespace.monitoring]
}

resource "helm_release" "cluster_autoscaler" {
  name       = "cluster-autoscaler"
  repository = "https://kubernetes.github.io/autoscaler"
  chart      = "cluster-autoscaler"
  namespace  = "kube-system"
  version    = "9.29.0"

  set {
    name  = "autoDiscovery.clusterName"
    value = module.eks.cluster_name
  }

  set {
    name  = "awsRegion"
    value = var.region
  }

  depends_on = [module.eks]
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_name" {
  description = "Kubernetes Cluster Name"
  value       = module.eks.cluster_name
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "vpc_id" {
  description = "ID of the VPC where the cluster is deployed"
  value       = module.vpc.vpc_id
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.mlflow.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
  sensitive   = true
}

output "s3_mlflow_bucket" {
  description = "Name of the S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "s3_data_bucket" {
  description = "Name of the S3 bucket for data storage"
  value       = aws_s3_bucket.data_storage.bucket
}

output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.mlops.dns_name
}