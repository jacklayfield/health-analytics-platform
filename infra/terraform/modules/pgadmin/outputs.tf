# =============================================================================
# pgAdmin Module Outputs
# =============================================================================

output "cluster_id" {
  description = "ECS cluster ID created for pgAdmin"
  value       = aws_ecs_cluster.this.id
}

output "lb_dns_name" {
  description = "DNS name of pgAdmin load balancer"
  value       = aws_lb.pgadmin.dns_name
}

output "service_arn" {
  description = "ARN of the ECS service"
  value       = aws_ecs_service.pgadmin.arn
}
