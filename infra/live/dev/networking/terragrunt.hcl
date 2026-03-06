include {
  path = find_in_parent_folders()
}

terraform {
  source = "../../../terraform/modules/networking"
}

inputs = {
  project_name       = "health-analytics"
  environment        = "dev"
  vpc_cidr           = "10.0.0.0/16"
  availability_zones = ["us-east-1a","us-east-1b","us-east-1c"]
  tags               = local.common_tags
}