include {
  path = find_in_parent_folders()
}

dependency "networking" {
  config_path = "../networking"
}

terraform {
  source = "../../../terraform/modules/pgadmin"
}

inputs = {
  project_name       = "health-analytics"
  environment        = "dev"
  admin_email        = "admin@example.com"
  admin_password     = "password"
  vpc_id             = dependency.networking.outputs.vpc_id
  subnet_ids         = dependency.networking.outputs.public_subnet_ids
  security_group_id  = dependency.networking.outputs.application_security_group_id
  tags               = local.common_tags
} 