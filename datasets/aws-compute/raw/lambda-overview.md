# AWS Lambda

AWS Lambda is a serverless compute service that runs your code in response to events and automatically manages the underlying compute resources.

## Runtimes

Lambda natively supports the following runtimes:

- Node.js 18.x, 20.x
- Python 3.9, 3.10, 3.11, 3.12
- Java 11, 17, 21
- .NET 6, 8
- Ruby 3.2, 3.3
- Go (provided.al2023)
- Custom runtime (provided.al2023)

## Function Configuration

### Memory and Timeout

| Setting | Minimum | Maximum | Default |
|---------|---------|---------|---------|
| Memory | 128 MB | 10,240 MB | 128 MB |
| Timeout | 1 second | 900 seconds (15 min) | 3 seconds |
| Ephemeral storage | 512 MB | 10,240 MB | 512 MB |
| Environment variables | — | 4 KB total | — |
| Deployment package | — | 50 MB (zipped), 250 MB (unzipped) | — |

### Environment Variables

Lambda functions can access environment variables through the standard runtime APIs. Common system variables:

- `AWS_LAMBDA_FUNCTION_NAME` — the name of the function
- `AWS_LAMBDA_FUNCTION_VERSION` — the version being executed
- `AWS_REGION` — the AWS Region where the function runs
- `AWS_LAMBDA_LOG_GROUP_NAME` — the CloudWatch Logs group
- `_HANDLER` — the handler location configured on the function

## VPC Configuration

By default, Lambda runs in a Lambda-managed VPC with internet access. To access resources in your own VPC:

1. Attach the function to your VPC by specifying SubnetIds and SecurityGroupIds
2. Lambda creates Elastic Network Interfaces (ENIs) in your subnets
3. The function can now access VPC resources (RDS, ElastiCache, etc.)

### VPC and Internet Access

Lambda functions in a VPC do NOT have internet access by default. To restore internet access:

- **NAT Gateway**: Route outbound traffic through a NAT Gateway in a public subnet ($0.045/hr + $0.045/GB)
- **VPC Endpoints**: Access specific AWS services without internet (S3, DynamoDB, SQS, etc.)

### IAM Requirements

The execution role needs `AWSLambdaVPCAccessExecutionRole` managed policy which grants:

- `ec2:CreateNetworkInterface`
- `ec2:DescribeNetworkInterfaces`
- `ec2:DeleteNetworkInterface`

## Execution Role

Every Lambda function has an execution role — an IAM role that grants the function permission to access AWS services and resources.

### Trust Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
```

### Common Managed Policies

| Policy | Grants |
|--------|--------|
| `AWSLambdaBasicExecutionRole` | CloudWatch Logs (create group, put events) |
| `AWSLambdaVPCAccessExecutionRole` | ENI management in VPC |
| `AWSLambdaDynamoDBExecutionRole` | DynamoDB Streams read |
| `AWSLambdaSQSQueueExecutionRole` | SQS receive/delete messages |
| `AWSLambdaKinesisExecutionRole` | Kinesis Streams read |

## Event Sources

Lambda can be triggered by many AWS services:

### Synchronous Invocation
- API Gateway (REST, HTTP, WebSocket)
- Application Load Balancer
- CloudFront (Lambda@Edge)
- Cognito
- Alexa

### Asynchronous Invocation
- S3 (object events)
- SNS (topic messages)
- CloudWatch Events / EventBridge
- CodeCommit
- CloudFormation (custom resources)

### Stream-based (Polling)
- DynamoDB Streams
- Kinesis Data Streams
- Amazon MQ
- SQS

## Layers

Lambda Layers let you package libraries and dependencies separately from your function code. A function can use up to 5 layers. Layers are extracted to `/opt` in the execution environment.

## Provisioned Concurrency

Provisioned Concurrency keeps a requested number of execution environments initialized and ready to respond. Use it for latency-sensitive workloads to avoid cold starts.

### Pricing
- Provisioned Concurrency: $0.0000041667/GB-second
- Duration: $0.0000033334/GB-second (same as on-demand)

## Cold Starts

Cold start occurs when Lambda needs to create a new execution environment. Contributing factors:

1. Runtime initialization (loading runtime, JIT compilation for Java/.NET)
2. Handler initialization (loading libraries, establishing connections)
3. VPC ENI attachment (1-10 seconds additional for VPC-connected functions)

### Mitigation
- Use Provisioned Concurrency
- Keep deployment packages small
- Initialize SDK clients outside the handler
- Use arm64 (Graviton2) for faster init
