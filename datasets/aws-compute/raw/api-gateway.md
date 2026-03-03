# Amazon API Gateway

Amazon API Gateway is a fully managed service for creating, publishing, maintaining, monitoring, and securing APIs at any scale.

## API Types

### REST API
- Full API management features: usage plans, API keys, request/response transformations
- Edge-optimized, regional, or private deployment
- Request validation, WAF integration, caching
- Cost: $3.50 per million requests + data transfer

### HTTP API
- Low-latency, cost-effective alternative to REST APIs
- Simpler feature set — no usage plans, request transformation, or caching
- Native JWT authorizer, OIDC support
- Cost: $1.00 per million requests (71% cheaper than REST)

### WebSocket API
- Real-time two-way communication
- Route messages to Lambda functions based on route selection expression
- Maintain persistent connections (up to 2 hours idle, 24 hours max)

## Lambda Integration

### Lambda Proxy Integration (recommended)
API Gateway passes the entire request to Lambda as an event object:

```json
{
  "httpMethod": "GET",
  "path": "/users/123",
  "queryStringParameters": {"expand": "profile"},
  "headers": {"Authorization": "Bearer token..."},
  "body": null,
  "requestContext": {
    "authorizer": {"claims": {"sub": "user-id"}}
  }
}
```

Lambda must return a specific response format:

```json
{
  "statusCode": 200,
  "headers": {"Content-Type": "application/json"},
  "body": "{\"name\": \"John\"}"
}
```

### Lambda Non-Proxy Integration
- API Gateway transforms requests before sending to Lambda
- Map request parameters to Lambda function input
- Transform Lambda output to HTTP response using mapping templates (VTL)

## Authorization

### IAM Authorization
- Caller signs request with AWS Signature v4
- API Gateway validates the signature against IAM policies
- Best for: service-to-service calls within AWS

### Cognito User Pools
- JWT tokens from Cognito
- API Gateway validates the token directly
- Best for: mobile/web apps with user authentication

### Lambda Authorizer (Custom)
- Lambda function receives the token/request, returns an IAM policy
- Two types: token-based (Authorization header) and request-based (all parameters)
- Results can be cached (TTL 0-3600 seconds)
- Best for: custom auth schemes, third-party tokens

## Throttling and Quotas

| Setting | Default | Maximum |
|---------|---------|---------|
| Requests per second (steady-state) | 10,000 | Account limit increase |
| Burst limit | 5,000 | — |
| Payload size | — | 10 MB |
| Integration timeout | — | 29 seconds |
| WebSocket message size | — | 128 KB |

## CORS

Enable Cross-Origin Resource Sharing for browser clients:

- Configure allowed origins, methods, headers
- For Lambda proxy integration: Lambda must return CORS headers in the response
- For non-proxy: configure in API Gateway method response

## Stages and Deployment

- Each API has stages (e.g., dev, staging, prod)
- Stage variables parameterize integrations (different Lambda aliases per stage)
- Canary deployments: route a percentage of traffic to a new deployment
- Each stage has its own URL: `https://{api-id}.execute-api.{region}.amazonaws.com/{stage}`
