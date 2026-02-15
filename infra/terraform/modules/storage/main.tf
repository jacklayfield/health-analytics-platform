# =============================================================================
# Storage Module - S3 Buckets for Data Lake and ML Artifacts
# =============================================================================

# -----------------------------------------------------------------------------
# Raw Data Bucket (for incoming ETL data)
# -----------------------------------------------------------------------------
resource "aws_s3_bucket" "raw_data" {
  bucket = "${var.s3_bucket_prefix}-raw-data-${var.environment}"

  tags = merge(var.tags, {
    Name        = "${var.s3_bucket_prefix}-raw-data-${var.environment}"
    Purpose     = "raw-data"
    Environment = var.environment
  })
}

resource "aws_s3_bucket_versioning" "raw_data" {
  count  = var.enable_versioning ? 1 : 0
  bucket = aws_s3_bucket.raw_data.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# -----------------------------------------------------------------------------
# Processed Data Bucket (for cleaned/transformed data)
# -----------------------------------------------------------------------------
resource "aws_s3_bucket" "processed_data" {
  bucket = "${var.s3_bucket_prefix}-processed-data-${var.environment}"

  tags = merge(var.tags, {
    Name        = "${var.s3_bucket_prefix}-processed-data-${var.environment}"
    Purpose     = "processed-data"
    Environment = var.environment
  })
}

resource "aws_s3_bucket_versioning" "processed_data" {
  count  = var.enable_versioning ? 1 : 0
  bucket = aws_s3_bucket.processed_data.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "processed_data" {
  bucket = aws_s3_bucket.processed_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# -----------------------------------------------------------------------------
# ML Models Bucket (for MLflow models and artifacts)
# -----------------------------------------------------------------------------
resource "aws_s3_bucket" "ml_models" {
  bucket = "${var.s3_bucket_prefix}-ml-models-${var.environment}"

  tags = merge(var.tags, {
    Name        = "${var.s3_bucket_prefix}-ml-models-${var.environment}"
    Purpose     = "ml-models"
    Environment = var.environment
  })
}

resource "aws_s3_bucket_versioning" "ml_models" {
  count  = var.enable_versioning ? 1 : 0
  bucket = aws_s3_bucket.ml_models.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ml_models" {
  bucket = aws_s3_bucket.ml_models.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# -----------------------------------------------------------------------------
# Analytics/Reports Bucket (for generated reports and dashboards)
# -----------------------------------------------------------------------------
resource "aws_s3_bucket" "analytics" {
  bucket = "${var.s3_bucket_prefix}-analytics-${var.environment}"

  tags = merge(var.tags, {
    Name        = "${var.s3_bucket_prefix}-analytics-${var.environment}"
    Purpose     = "analytics"
    Environment = var.environment
  })
}

resource "aws_s3_bucket_versioning" "analytics" {
  count  = var.enable_versioning ? 1 : 0
  bucket = aws_s3_bucket.analytics.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "analytics" {
  bucket = aws_s3_bucket.analytics.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# -----------------------------------------------------------------------------
# S3 Bucket Policy (optional - for cross-account access)
# -----------------------------------------------------------------------------
resource "aws_s3_bucket_policy" "raw_data" {
  bucket = aws_s3_bucket.raw_data.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DeneyInsecureTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.raw_data.arn,
          "${aws_s3_bucket.raw_data.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })
}
