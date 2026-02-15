# =================================================================.============
# Database Module Outputs
# =============================================================================

output "db_endpoint" {
  description = "RDS database endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}

output "db_port" {
  description = "RDS database port"
  value       = aws_db_instance.main.port
}

output "db_name" {
  description = "Database name"
  value       = aws_db_instance.main.db_name
}

output "db_arn" {
  description = "RDS database ARN"
  value       = aws_db_instance.main.arn
}

output "db_instance_id" {
  description = "RDS instance ID"
  value       = aws_db_instance.main.id
}

output "db_instance_address" {
  description = "RDS instance address"
  value       = aws_db_instance.main.address
  sensitive   = true
}
