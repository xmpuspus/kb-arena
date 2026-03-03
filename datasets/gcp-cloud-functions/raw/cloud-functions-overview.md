# Cloud Functions Overview

Google Cloud Functions is a serverless execution environment for building and connecting cloud services. With Cloud Functions you write simple, single-purpose functions that are attached to events emitted from your cloud infrastructure and services.

## Runtimes

Cloud Functions supports the following runtimes:

- Node.js 18, 20
- Python 3.9, 3.10, 3.11, 3.12
- Go 1.21, 1.22
- Java 11, 17, 21
- .NET 6, 8
- Ruby 3.2, 3.3
- PHP 8.2, 8.3

## Triggers

### HTTP Triggers

HTTP functions are invoked by standard HTTP requests. These HTTP requests wait for the response and support handling of common HTTP request methods like GET, PUT, POST, DELETE and OPTIONS.

### Event-driven Triggers

Cloud Functions can be triggered by events from Google Cloud services:

- **Cloud Storage**: Object creation, deletion, archival, metadata updates
- **Pub/Sub**: Message published to a topic
- **Firestore**: Document creation, update, deletion
- **Cloud Scheduler**: Cron-based scheduled invocations

## VPC Connector

To connect your function to a VPC network, you need a Serverless VPC Access connector. This allows your function to access resources in the VPC such as:

- Cloud SQL instances
- Memorystore (Redis) instances
- Compute Engine VMs on internal IPs
- GKE clusters with internal endpoints

### Configuration

```yaml
vpc_connector: projects/my-project/locations/us-central1/connectors/my-connector
vpc_connector_egress_settings: ALL_TRAFFIC
```

Setting `ALL_TRAFFIC` routes all egress through the VPC connector, not just traffic to internal IPs.

## Memory and Timeout

| Setting | Minimum | Maximum | Default |
|---------|---------|---------|---------|
| Memory | 128 MB | 32 GB | 256 MB |
| Timeout | 1 second | 540 seconds (Gen 1) / 3600 seconds (Gen 2) | 60 seconds |
| Max instances | 1 | 3000 | 100 |

## IAM and Permissions

Cloud Functions uses IAM for access control. Key roles:

- `roles/cloudfunctions.developer`: Deploy and manage functions
- `roles/cloudfunctions.invoker`: Invoke HTTP functions
- `roles/cloudfunctions.viewer`: View function metadata

The function's runtime service account needs permissions for any Google Cloud resources it accesses (e.g., `roles/cloudsql.client` for Cloud SQL).
