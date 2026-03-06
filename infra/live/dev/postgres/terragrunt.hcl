include {
  path = find_in_parent_folders()
}

dependency "networking" {
  config_path = "../networking"
}

terraform {
  source = "../../../terraform/modules/database"
}

inputs = {
  project_name      = "health-analytics"
  environment       = "dev"
  db_username       = "devuser"
  db_password       = "devpass"
  vpc_id            = dependency.networking.outputs.vpc_id
  subnet_ids        = dependency.networking.outputs.private_subnet_ids
  security_group_id = dependency.networking.outputs.database_security_group_id
  tags              = local.common_tags
} 