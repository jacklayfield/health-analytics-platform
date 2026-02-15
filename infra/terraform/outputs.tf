# =============================================================================
# Common Outputs
# =============================================================================

output "project_name" {
  description = "Project name"
  value       = var.project_name
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

# =============================================================================
# Networking Outputs
# =============================================================================

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.networking.vpc_id
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = module.networking.public_subnet_ids
}

output "private_subnet_ids" {
  description = "IDs of private subnets"
  value       = module.networking.private_subnet_ids
}

output "security_groups" {
  description = "Security groups created"
  value       = module.networking.security_groups
}

# =============================================================================
# Database Outputs
# =============================================================================

output "db_endpoint" {
  description = "RDS database endpoint"
  value       = module.database.db_endpoint
  sensitive   = true
}

output "db_port" {
  description = "RDS database port"
  value       = module.database.db_port
}

output "db_name" {
  description = "Database name"
  value       = module.database.db_name
}

output "db_arn" {
  description = "RDS database ARN"
  value       = module.database.db_arn
}

# =============================================================================
# S3 Outputs
# =============================================================================

output "s3_bucket_names" {
  description = "Names of S3 buckets"
  value       = {
    raw_data     = module.storage.raw_data_bucket_name
    processed    = module.storage.processed_data_bucket_name
    ml_models    = module.storage.ml_models_bucket_name
    analytics    = module.storage.analytics_bucket_name
  }
}

output "s3_bucket_arns" {
  description = "ARNs of S3 buckets"
  value       = {
    raw_data     = module.storage.raw_data_bucket_arn
    processed    = module.storage.processed_data_bucket_arn
    ml_models    = module.storage.ml_models_bucket_arn
    analytics    = module.storage.analytics_bucket_arn
  }
}