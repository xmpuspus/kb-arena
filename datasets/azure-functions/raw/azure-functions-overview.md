# Azure Functions Overview

Azure Functions is a serverless compute service that lets you run event-triggered code without having to explicitly provision or manage infrastructure.

## Hosting Plans

### Consumption Plan
Pay only for the time your functions run. Automatically scales based on demand. Functions timeout after 5 minutes by default (configurable up to 10 minutes).

### Premium Plan
Pre-warmed instances to avoid cold starts. VNet connectivity built in. Unlimited execution duration. Starts at one pre-warmed instance minimum.

### Dedicated (App Service) Plan
Run on dedicated VMs. Good when you already have existing App Service instances. Supports always-on to avoid cold starts.

## Triggers and Bindings

### Triggers
- **HTTP trigger**: Respond to HTTP requests
- **Timer trigger**: Run on a schedule (CRON expressions)
- **Blob Storage trigger**: React to new or updated blobs
- **Queue Storage trigger**: Process messages from Azure Queue Storage
- **Cosmos DB trigger**: React to changes in Cosmos DB (change feed)
- **Event Grid trigger**: React to Event Grid events
- **Event Hub trigger**: Process events from Event Hubs
- **Service Bus trigger**: Process messages from Service Bus queues/topics

### Bindings
Input and output bindings connect your function to other services declaratively:

```json
{
  "type": "cosmosDB",
  "direction": "in",
  "name": "documents",
  "databaseName": "MyDatabase",
  "collectionName": "MyCollection",
  "connectionStringSetting": "CosmosDBConnection"
}
```

## VNet Integration

Azure Functions Premium and Dedicated plans support VNet integration:

1. **Regional VNet integration**: Connect to resources in the same region VNet
2. **Private endpoints**: Restrict inbound access to your function app
3. **Service endpoints**: Restrict outbound access to specific Azure services

### Connecting to Azure SQL or Cosmos DB via Private Endpoints

1. Create a Private Endpoint for your database
2. Enable VNet integration on your Function App
3. Configure the connection string to use the private endpoint FQDN
4. Set `WEBSITE_VNET_ROUTE_ALL=1` to route all traffic through the VNet

## Authentication

Azure Functions supports built-in authentication through Azure App Service Authentication (Easy Auth):

- Azure Active Directory
- Microsoft Account
- Google
- Facebook
- Twitter
- OpenID Connect providers

Function-level authorization keys:
- **Anonymous**: No key required
- **Function**: Function-specific key required
- **Admin**: Master key required
