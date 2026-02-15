# =============================================================================
# Database Module - RDS PostgreSQL
# =============================================================================

# -----------------------------------------------------------------------------
# DB Subnet Group
# -----------------------------------------------------------------------------
resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-db-subnet-${var.environment}"
  subnet_ids = var.subnet_ids

  tags = merge(var.tags, {
    Name = "${var.project_name}-db-subnet-${var.environment}"
  })
}

# -----------------------------------------------------------------------------
# RDS Instance
# -----------------------------------------------------------------------------
resource "aws_db_instance" "main" {
  identifier     = "${var.project_name}-postgres-${var.environment}"
  engine        = "postgres"
  engine_version = "15.4"
  instance_class = var.db_instance_class

  # Storage
  allocated_storage     = var.db_allocated_storage
  storage_type          = "gp3"
  storage_encrypted    = true

  # Credentials
  username = var.db_username
  password = var.db_password
  db_name  = var.db_name

  # Network
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [var.security_group_id]

  # Backup & Maintenance
  backup_retention_period = var.backup_retention_days
  backup_window           = "03:00-04:00"
  maintenance_window      = "mon:04:00-mon:05:00"
  skip_final_snapshot     = var.skip_final_snapshot
  final_snapshot_identifier = "${var.project_name}-final-snapshot-${var.environment}"

  # Performance & Scaling
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn
  ca_cert_identifier   = "rds-ca-2024"

  # Tags
  tags = merge(var.tags, {
    Name = "${var.project_name}-rds-${var.environment}"
  })
}

# -----------------------------------------------------------------------------
# IAM Role for RDS Monitoring
# -----------------------------------------------------------------------------
resource "aws_iam_role" "rds_monitoring" {
  name = "${var.project_name}-rds-monitoring-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "monitoring.rds.amazonaws.com"
      }
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# -----------------------------------------------------------------------------
# Parameter Group
# -----------------------------------------------------------------------------
resource "aws_db_parameter_group" "main" {
  name   = "${var.project_name}-pg-${var.environment}"
  family = "postgres15"

  parameter {
    name  = "shared_buffers"
    value = "DBInstanceClassMemory*1/4"
  }

  parameter {
    name  = "max_connections"
    value = "LEAST({DBInstanceClassMemory/9539576767},10000)"
  }

  parameter {
    name  = "work_mem"
    value = "4MB"
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-pg-${var.environment}"
  })
}

# -----------------------------------------------------------------------------
# Option Group
# -----------------------------------------------------------------------------
resource "aws_db_option_group" "main" {
  name     = "${var.project_name}-og-${var.environment}"
  engine_name = "postgres"
  major_engine_version = "15"

  option {
    option_name = "NULL"
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-og-${var.environment}"
  })
}
