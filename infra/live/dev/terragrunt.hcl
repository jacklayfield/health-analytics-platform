# Common Terragrunt configuration for all environments

remote_state {
  backend = "s3"
  config = {
    bucket = "health-analytics-terraform-state"
    key    = "${path_relative_to_include()}/terraform.tfstate"
    region = "us-east-1"
  }
}

locals {
  common_tags = {
    Project     = "health-analytics-platform"
    Environment = "${path_relative_to_include()}" # may be overridden in child
    ManagedBy   = "Terragrunt"
  }
}
