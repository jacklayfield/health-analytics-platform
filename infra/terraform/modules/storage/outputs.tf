# =============================================================================
:# Storage Module Outputs
# =============================================================================

output "raw_data_bucket_name" {
  description = "Name of the raw data S3 bucket"
  value       = aws_s3_bucket.raw_data.id
}

output "raw_data_bucket_arn" {
  description = "ARN of the raw data S3 bucket"
  value       = aws_s3_bucket.raw_data.arn
}

output "processed_data_bucket_name" {
  description = "Name of the processed data S3 bucket"
  value       = aws_s3_bucket.processed_data.id
}

output "processed_data_bucket_arn" {
  description = "ARN of the processed data S3 bucket"
  value       = aws_s3_bucket.processed_data.arn
}

output "ml_models_bucket_name" {
  description = "Name of the ML models S3 bucket"
  value       = aws_s3_bucket.ml_models.id
}

output "ml_models_bucket_arn" {
  description = "ARN of the ML models S3 bucket"
  value       = aws_s3_bucket.ml_models.arn
}

output "analytics_bucket_name" {
  description = "Name of the analytics S3 bucket"
  value       = aws_s3_bucket.analytics.id
}

output "analytics_bucket_arn" {
  description = "ARN of the analytics S3 bucket"
  value       = aws_s3_bucket.analytics.arn
}