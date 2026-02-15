# Database Module

This module creates an RDS PostgreSQL instance for the Health Analytics Platform.

## Resources Created

- **RDS PostgreSQL Instance**: Multi-AZ PostgreSQL 15.4 database
- **DB Subnet Group**: Subnet group for RDS deployment
- **DB Parameter Group**: Custom parameter group for PostgreSQL tuning
- **DB Option Group**: Option group for PostgreSQL features
- **IAM Role**: Enhanced monitoring role for RDS
- **IAM Policy Attachment**: Policy for RDS monitoring

## Features

- **Storage Encryption**: Enabled by default using AWS-managed keys
- **Automated Backups**: Configurable retention period (default: 7 days)
- **Enhanced Monitoring**: CloudWatch integration for performance metrics
- **Multi-AZ**: Deploys to multiple availability zones (when available)
- **Performance Tuning**: Pre-configured PostgreSQL parameters

## Usage

```
hcl
module "database" {
  source = "./modules/database"

  project_name         = "health-analytics"
  environment         = "dev"
  db_instance_class   = "db.t3.medium"
  db_allocated_storage = 20
  db_name             = "healthanalytics"
  db_username         = "dbadmin"
  db_password         = var.db_password  # Use a secure variable!
  
  vpc_id             = module.networking.vpc_id
  subnet_ids         = module.networking.private_subnet_ids
  security_group_id  = module.networking.database_security_group_id
  
  tags = {
    Project     = "health-analytics-platform"
    ManagedBy   = "Terraform"
  }
}
```

## Outputs

| Output | Description |
|--------|-------------|
| `db_endpoint` | RDS database endpoint (sensitive) |
| `db_port` | RDS database port |
| `db_name` | Database name |
| `db_arn` | RDS database ARN |
| `db_instance_id` | RDS instance ID |
| `db_instance_address` | RDS instance address (sensitive) |

## Requirements

- Terraform >= 1.0
- AWS Provider >= 5.0
- VPC and subnets must exist (from networking module)

## Security Considerations

- The `db_password` variable is marked as sensitive
- Use AWS Secrets Manager for production deployments
- Database is deployed in private subnets
- SSL enforcement recommended for production
