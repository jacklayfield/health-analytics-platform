# Networking Module

This module creates the networking infrastructure for the Health Analytics Platform on AWS.

## Resources Created

- **VPC**: Virtual Private Cloud with configurable CIDR block
- **Public Subnets**: For load balancers and public-facing resources
- **Private Subnets**: For application servers, databases, and internal services
- **Internet Gateway**: For VPC internet access
- **NAT Gateways**: For private subnet outbound internet access
- **Route Tables**: Public and private routing tables
- **Security Groups**:
  - Database security group (PostgreSQL port 5432)
  - Application security group (HTTP/HTTPS)
  - ALB security group (HTTP/HTTPS)

## Usage

```
hcl
module "networking" {
  source = "./modules/networking"

  project_name       = "health-analytics"
  environment        = "dev"
  vpc_cidr          = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  
  tags = {
    Project     = "health-analytics-platform"
    ManagedBy   = "Terraform"
    Environment = "dev"
  }
}
```

## Outputs

| Output | Description |
|--------|-------------|
| `vpc_id` | ID of the VPC |
| `vpc_cidr` | CIDR block of the VPC |
| `public_subnet_ids` | IDs of public subnets |
| `private_subnet_ids` | IDs of private subnets |
| `database_security_group_id` | ID of the database security group |
| `application_security_group_id` | ID of the application security group |
| `alb_security_group_id` | ID of the ALB security group |

## Requirements

- Terraform >= 1.0
- AWS Provider >= 5.0
