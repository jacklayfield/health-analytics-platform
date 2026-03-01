# =============================================================================
# pgAdmin Module Variables
# =============================================================================

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
}

variable "environment" {
  description = "Environment name (dev/stage/prod)"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID for load balancer"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for the service (public subnets)"
  type        = list(string)
}

variable "security_group_id" {
  description = "Security group to attach to the service and load balancer"
  type        = string
}

variable "admin_email" {
  description = "pgAdmin default admin email"
  type        = string
}

variable "admin_password" {
  description = "pgAdmin default admin password"
  type        = string
  sensitive   = true
}

variable "image" {
  description = "Container image for pgAdmin"
  type        = string
  default     = "dpage/pgadmin4:latest"
}

variable "cpu" {
  description = "CPU units for the task"
  type        = number
  default     = 256
}

variable "memory" {
  description = "Memory (MiB) for the task"
  type        = number
  default     = 512
}

variable "port" {
  description = "Container port pgAdmin will listen on"
  type        = number
  default     = 80
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}