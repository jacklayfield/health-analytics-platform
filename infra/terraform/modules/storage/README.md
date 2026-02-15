# Storage Module

This module creates S3 buckets for the Health Analytics Platform data lake and ML artifacts.

## Resources Created

### S3 Buckets

1. **Raw Data Bucket** (`{prefix}-raw-data-{env}`)
   - Purpose: Storing incoming ETL data before processing
   - Versioning: Enabled by default
   - Encryption: AES-256 server-side encryption

2. **Processed Data Bucket** (`{prefix}-processed-data-{env}`)
   - Purpose: Storing cleaned and transformed data
   - Versioning: Enabled by default
   - Encryption: AES-256 server-side encryption

3. **ML Models Bucket** (`{prefix}-ml-models-{env}`)
   - Purpose: Storing ML models and MLflow artifacts
   - Versioning: Enabled by default
   - Encryption: AES-256 server-side encryption

4. **Analytics Bucket** (`{prefix}-analytics-{env}`)
   - Purpose: Storing generated reports and dashboard data
   - Versioning: Enabled by default
   - Encryption: AES-256 server-side encryption

## Features

- **Versioning**: Optional bucket versioning for all buckets
- **Encryption**: Server-side encryption with AWS-managed keys
- **Security Policy**: Deny insecure transport (HTTP) policy
- **Tagging**: Consistent tagging for cost allocation

## Usage

```
hcl
module "storage" {
  source = "./modules/storage"

  project_name      = "health-analytics"
  environment      = "dev"
  s3_bucket_prefix = "health-analytics"
  enable_versioning = true
  
  tags = {
    Project     = "health-analytics-platform"
    ManagedBy   = "Terraform"
  }
}
```

## Outputs

| Output | Description |
|--------|-------------|
| `raw_data_bucket_name` | Name of raw data bucket |
| `raw_data_bucket_arn` | ARN of raw data bucket |
| `processed_data_bucket_name` | Name of processed data bucket |
| `processed_data_bucket_arn` | ARN of processed data bucket |
| `ml_models_bucket_name` | Name of ML models bucket |
| `ml_models_bucket_arn` | ARN of ML models bucket |
| `analytics_bucket_name` | Name of analytics bucket |
| `analytics_bucket_arn` | ARN of analytics bucket |

## Requirements

- Terraform >= 1.0
- AWS Provider >= 5.0

## Data Flow

```
OpenFDA API → Raw Data Bucket → ETL Pipeline → Processed Data Bucket
                                                      ↓
                                              PostgreSQL Database
                                                      ↓
                                        ML Training → ML Models Bucket
                                                      ↓
                                        Dashboard → Analytics Bucket
```

## Cost Considerations

- S3 pricing: GB/month + requests + data transfer
- Enable lifecycle policies for cost optimization in production
- Consider using S3 Intelligent-Tiering for infrequent access
