# =============================================================================
# pgAdmin ECS/Fargate Module
# Creates an ECS cluster, task definition, service, and Application Load Balancer
# =============================================================================

resource "aws_ecs_cluster" "this" {
  name = "${var.project_name}-${var.environment}-pgadmin"
  tags = var.tags
}

resource "aws_ecs_task_definition" "pgadmin" {
  family                   = "${var.project_name}-${var.environment}-pgadmin"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory

  container_definitions = jsonencode([
    {
      name      = "pgadmin"
      image     = var.image
      portMappings = [{ containerPort = var.port }]
      environment = [
        { name = "PGADMIN_DEFAULT_EMAIL",    value = var.admin_email },
        { name = "PGADMIN_DEFAULT_PASSWORD", value = var.admin_password }
      ]
    }
  ])
}

resource "aws_lb" "pgadmin" {
  name               = "${var.project_name}-${var.environment}-pgadmin-alb"
  internal           = false
  load_balancer_type = "application"
  subnets            = var.subnet_ids
  security_groups    = [var.security_group_id]
  tags               = var.tags
}

resource "aws_lb_target_group" "pgadmin" {
  name     = "${var.project_name}-${var.environment}-pgadmin-tg"
  port     = var.port
  protocol = "HTTP"
  vpc_id   = var.vpc_id
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.pgadmin.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.pgadmin.arn
  }
}

resource "aws_ecs_service" "pgadmin" {
  name            = aws_ecs_task_definition.pgadmin.family
  cluster         = aws_ecs_cluster.this.id
  launch_type     = "FARGATE"
  task_definition = aws_ecs_task_definition.pgadmin.arn
  desired_count   = 1

  network_configuration {
    subnets         = var.subnet_ids
    security_groups = [var.security_group_id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.pgadmin.arn
    container_name   = "pgadmin"
    container_port   = var.port
  }

  tags = var.tags
}