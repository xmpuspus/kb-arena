# Databricks MLOps Overview

Databricks provides an integrated platform for MLOps — the practice of deploying and maintaining machine learning models in production reliably and efficiently.

## MLflow on Databricks

MLflow is the core component for ML lifecycle management on Databricks.

### Experiment Tracking

Track parameters, metrics, and artifacts for each training run:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 100)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)
    mlflow.sklearn.log_model(model, "model")
```

### Model Registry

The Model Registry provides a centralized model store with:

- **Model versioning**: Track multiple versions of each model
- **Stage transitions**: Move models through None → Staging → Production → Archived
- **Approval workflows**: Require approvals before promoting to Production
- **Lineage tracking**: Link models to training runs and datasets

## Feature Store

Databricks Feature Store manages features for ML models:

- **Feature tables**: Store features as Delta tables
- **Online serving**: Publish features to online stores for real-time inference
- **Point-in-time lookups**: Prevent data leakage with time-aware joins
- **Feature lineage**: Track which models use which features

```python
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()
fs.create_table(
    name="user_features",
    primary_keys=["user_id"],
    df=feature_df,
    description="User behavioral features"
)
```

## Model Serving

### Serverless Model Serving

Deploy models as REST endpoints with automatic scaling:

- Pay-per-query pricing
- Automatic scale-to-zero
- GPU support for deep learning models
- A/B testing with traffic splitting

### Provisioned Throughput

For latency-sensitive workloads:

- Dedicated compute resources
- Sub-100ms inference latency
- Guaranteed throughput (tokens per minute)

## Unity Catalog for ML

Unity Catalog provides governance for ML artifacts:

- **Models**: Register models in Unity Catalog for cross-workspace access
- **Feature tables**: Govern feature access with fine-grained permissions
- **Volumes**: Store training data and model artifacts with access controls
- **Lineage**: End-to-end lineage from data to model to serving endpoint

### Permissions

| Object | Permission | Description |
|--------|-----------|-------------|
| Model | EXECUTE | Use model for inference |
| Model | APPLY_TAG | Add tags to model versions |
| Feature Table | SELECT | Read features |
| Volume | READ_VOLUME | Read files from volume |
| Serving Endpoint | CAN_QUERY | Send inference requests |

## Workflows and Jobs

Automate ML pipelines with Databricks Workflows:

- **Task orchestration**: Chain notebooks, Python scripts, SQL queries
- **Scheduling**: CRON-based or event-triggered execution
- **Retry policies**: Automatic retry on failure with configurable attempts
- **Alerts**: Email/Slack notifications on job success/failure
- **Multi-task jobs**: DAG-based pipeline with dependencies between tasks
