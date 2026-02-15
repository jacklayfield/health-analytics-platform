# AWS Provider Configuration
# This file can be customized based on your AWS credentials setup
# Options: shared credentials, environment variables, or IAM role

provider "aws" {
  alias  = "region"
  region = var.aws_region
}