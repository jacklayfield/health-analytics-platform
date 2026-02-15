# =============================================================================
# Health Analytics Platform - Terraform Root Module
# =============================================================================
# This is the main Terraform configuration for deploying the
# Health Analytics Platform infrastructure to AWS.
#
# Usage:
#   terraform init
#   terraform plan
#   terraform apply
# =============================================================================

# -----------------------------------------------------------------------------
# Networking Module (VPC, Subnets, Security Groups)
# -----------------------------------------------------------------------------
module "networking" {
  source = "./modules/networking"

  project_name   = var.project_name
  environment    = var.environment
  vpc_cidr       = var.vpc_cidr
  availability_zones = var.availability_zones
  tags           = var.tags
}

# -----------------------------------------------------------------------------
# Database Module (RDS PostgreSQL)
# -----------------------------------------------------------------------------
module "database" {
  source = "./modules/database"

  project_name       = var.project_name
  environment        = var.environment
  db_instance_class = var.db_instance_class
  db_allocated_storage = var.db_allocated_storage
  db_name            = var.db_name
  db_username        = var.db_username
  db_password        = var.db_password
  
  vpc_id             = module.networking.vpc_id
  subnet_ids         = module.networking.private_subnet_ids
  security_group_id  = module.networking.database_security_group_id
  
  tags               = var.tags
}

# -----------------------------------------------------------------------------
# Storage Module (S3 Buckets)
# -----------------------------------------------------------------------------
module "storage" {
  source = "./modules/storage"

  project_name       = var.project_name
  environment        = var.environment
  s3_bucket_prefix   = var.s3_bucket_prefix
  enable_versioning  = var.enable_versioning
  
  tags               = var.tags
}