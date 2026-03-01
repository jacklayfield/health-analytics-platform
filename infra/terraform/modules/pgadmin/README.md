# pgAdmin Terraform Module

This module provisions pgAdmin as a small web application running on AWS ECS
(Fargate) fronted by an Application Load Balancer.  It is intended for use in
conjunction with the `networking` module which provides the VPC, subnets and
security groups.

## Inputs
*(See `variables.tf` for types and defaults)*

- `project_name`, `environment` – names used for resource naming.
- `vpc_id`, `subnet_ids` – networking information; subnets should be public for
  the ALB.
- `security_group_id` – security group applied to both the service and ALB.
- `admin_email`, `admin_password` – credentials for the initial pgAdmin
  account.
- Optional tuning: `image`, `cpu`, `memory`, `port`.
- `tags` – map of tags to apply to all resources.

## Outputs
- `cluster_id` – ECS cluster created by the module.
- `lb_dns_name` – DNS name of the load balancer where pgAdmin is reachable.
- `service_arn` – ARN of the ECS service.

## Example
```hcl
module "pgadmin" {
  source = "./modules/pgadmin"

  project_name      = "health-analytics"
  environment       = "dev"
  vpc_id            = module.networking.vpc_id
  subnet_ids        = module.networking.public_subnet_ids
  security_group_id = module.networking.application_security_group_id

  admin_email       = "admin@example.com"
  admin_password    = "password"
  tags              = local.common_tags
}
```
