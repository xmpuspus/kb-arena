# Amazon ECS and AWS Fargate

Amazon Elastic Container Service (ECS) is a container orchestration service. AWS Fargate is a serverless compute engine for containers that works with ECS.

## Launch Types

### EC2 Launch Type
- You manage the EC2 instances in your cluster
- Full control over instance types, AMIs, and capacity
- Can use Spot Instances for cost savings
- Best for: workloads needing GPUs, specific instance types, or Windows containers

### Fargate Launch Type
- No EC2 instances to manage — AWS manages the infrastructure
- Pay per vCPU and memory per second
- Task-level isolation (each task runs in its own kernel)
- Best for: microservices, batch jobs, serverless containers

## Task Definitions

A task definition is a blueprint for your application. Key settings:

| Setting | Description | Limits |
|---------|-------------|--------|
| CPU | vCPU units (1024 = 1 vCPU) | 256 to 16384 (.25 to 16 vCPU) |
| Memory | Memory in MB | 512 to 122880 (0.5 to 120 GB) |
| Network mode | awsvpc (recommended), bridge, host | Fargate requires awsvpc |
| Volumes | EFS, bind mounts, Docker volumes | 200 GB ephemeral storage (Fargate) |

### Container Definitions

Each task can have multiple containers:

```json
{
  "containerDefinitions": [
    {
      "name": "web",
      "image": "nginx:latest",
      "portMappings": [{"containerPort": 80, "protocol": "tcp"}],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/my-service",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "web"
        }
      }
    }
  ]
}
```

## Services

An ECS service maintains a desired count of tasks and integrates with:

- **Application Load Balancer**: HTTP/HTTPS routing to tasks
- **Network Load Balancer**: TCP/UDP high-performance routing
- **Service Discovery**: AWS Cloud Map for DNS-based service discovery
- **Auto Scaling**: Target tracking, step scaling, or scheduled scaling

### Deployment Strategies

| Strategy | Description |
|----------|-------------|
| Rolling update | Replace tasks gradually (min/max healthy percentages) |
| Blue/Green (CodeDeploy) | Deploy to new task set, shift traffic, rollback if needed |
| External | Third-party deployment controller |

## Networking (awsvpc)

Each task gets its own ENI with a private IP in your VPC subnet:

- Tasks can communicate using private IPs
- Security groups applied at the task level
- For internet access: public subnet + assign public IP, OR private subnet + NAT Gateway

## IAM Roles

### Task Execution Role
Allows ECS agent to pull images from ECR and write logs to CloudWatch:

- `ecr:GetAuthorizationToken`
- `ecr:BatchGetImage`
- `ecr:GetDownloadUrlForLayer`
- `logs:CreateLogStream`
- `logs:PutLogEvents`

### Task Role
Permissions for the application running inside the container (e.g., access S3, DynamoDB, SQS).

## Integration with Other Services

- **ECR**: Private container image registry
- **CloudWatch**: Logs, metrics, Container Insights
- **Secrets Manager / Parameter Store**: Inject secrets as environment variables
- **App Mesh**: Service mesh for microservices
- **X-Ray**: Distributed tracing
