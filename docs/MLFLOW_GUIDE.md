# MLflow Integration Guide

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Usage](#usage)
- [Model Registry](#model-registry)
- [API Integration](#api-integration)
- [Best Practices](#best-practices)

---

## Overview

This project integrates **MLflow** for comprehensive MLOps capabilities:

- ✅ **Automatic Experiment Tracking**: All training runs logged automatically
- ✅ **Central Model Registry**: Single source of truth for all models
- ✅ **Clean Versioning System**: Automatic version management
- ✅ **Easy Model Comparison**: Compare LSTM/GRU/Transformer side-by-side
- ✅ **Production Promotion**: Promote best model to production
- ✅ **API Integration**: API automatically uses current best model
- ✅ **Full Traceability**: Complete audit trail of all experiments
- ✅ **Real MLOps Pipeline**: Production-ready workflow

---

## Installation

### Install MLflow

```bash
pip install mlflow
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

### Start MLflow UI (Optional)

For local development:

```bash
mlflow ui
# Open http://localhost:5000
```

For team collaboration, start MLflow server:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

---

## Quick Start

### 1. Train Single Model with MLflow

```bash
python -m src.train_with_versioning --model lstm --epochs 20 --register-model
```

This will:
- Train the model
- Log all parameters and metrics to MLflow
- Register model in MLflow Model Registry

### 2. Train All Models and Compare

```bash
python -m src.train_with_mlflow --epochs 20 --auto-promote
```

This will:
- Train LSTM, BiLSTM, GRU, and Transformer
- Track all experiments in MLflow
- Compare models automatically
- Register all models
- Promote best model to Production

### 3. View Results

```bash
mlflow ui
# Open http://localhost:5000
```

In the UI, you can:
- Browse all experiments
- Compare model metrics
- View training curves
- Access registered models

---

## Features

### 1. Automatic Experiment Tracking

Every training run is automatically tracked with:

- **Hyperparameters**: epochs, batch_size, learning_rate, model_type, etc.
- **Metrics**: train_loss, test_rmse, test_mae (logged per epoch)
- **Artifacts**: Model files, scaler, metadata
- **Code Version**: Git commit (if available)
- **Environment**: Python version, package versions

**Example:**

```python
import mlflow
from src.mlflow_utils import setup_mlflow, log_training_params

setup_mlflow()
mlflow.start_run(run_name="LSTM-Training")

log_training_params(
    model_type="lstm",
    epochs=20,
    batch_size=64,
    learning_rate=0.001,
    ...
)
```

### 2. Central Model Registry

All models are registered in MLflow Model Registry with:

- **Automatic Versioning**: v1, v2, v3, etc.
- **Model Stages**: None → Staging → Production
- **Metadata**: Tags, descriptions, metrics
- **Lineage**: Link to training run

**Register Model:**

```python
from src.mlflow_utils import register_model_in_mlflow

model_version = register_model_in_mlflow(
    run_id=run_id,
    tags={"model_type": "lstm"},
    description="LSTM model with RMSE=24.5"
)
```

### 3. Model Comparison

Compare all models easily in MLflow UI:

1. **Filter by model_type**: See only LSTM, GRU, etc.
2. **Sort by metrics**: Find best RMSE or MAE
3. **Compare runs**: Side-by-side comparison
4. **Visualize**: Training curves, metrics over time

**Find Best Model:**

```python
from src.mlflow_utils import get_best_model_by_metric

best = get_best_model_by_metric("test_rmse", ascending=True)
print(f"Best model: {best['model_type']} with RMSE={best['metric_value']}")
```

### 4. Production Promotion

Promote models through stages:

- **None**: Newly registered model
- **Staging**: Testing before production
- **Production**: Live model used by API
- **Archived**: Previous production models

**Promote to Production:**

```python
from src.mlflow_utils import promote_model_to_production

promote_model_to_production("RUL-Prediction-Model", version=3)
```

### 5. API Integration

API automatically uses Production model from MLflow:

```bash
# Enable MLflow in API
export USE_MLFLOW=true
uvicorn src.api.main:app --reload
```

The API will:
- Load Production model from MLflow registry
- Automatically update when new model is promoted
- Fall back to legacy registry if MLflow unavailable

---

## Usage

### Training with MLflow

#### Option 1: Single Model Training

```bash
# Basic training with MLflow tracking
python -m src.train_with_versioning --model lstm --epochs 20

# Register model in registry
python -m src.train_with_versioning --model lstm --epochs 20 --register-model

# Auto-promote if best
python -m src.train_with_versioning --model lstm --epochs 20 --register-model --auto-promote
```

#### Option 2: Train All Models

```bash
# Train all models and compare
python -m src.train_with_mlflow --epochs 20

# Auto-register and promote best
python -m src.train_with_mlflow --epochs 20 --auto-promote
```

#### Option 3: Custom MLflow URI

```bash
# Use remote MLflow server
python -m src.train_with_versioning \
    --model lstm \
    --mlflow-uri http://mlflow-server:5000 \
    --register-model
```

### Disable MLflow

```bash
# Train without MLflow tracking
python -m src.train_with_versioning --model lstm --no-mlflow
```

---

## Model Registry

### Register Model

```python
from src.mlflow_utils import register_model_in_mlflow

# After training
with mlflow.start_run():
    # ... training code ...
    run_id = mlflow.active_run().info.run_id
    
    # Register model
    version = register_model_in_mlflow(
        run_id=run_id,
        tags={"model_type": "lstm", "experiment": "baseline"},
        description="Baseline LSTM model"
    )
```

### List Registered Models

```python
from src.mlflow_utils import list_registered_models

models = list_registered_models()
for model in models:
    print(f"Version {model['version']}: {model['stage']}")
```

### Promote Model

```python
from src.mlflow_utils import (
    promote_model_to_staging,
    promote_model_to_production
)

# Promote to Staging (for testing)
promote_model_to_staging("RUL-Prediction-Model", version=2)

# After validation, promote to Production
promote_model_to_production("RUL-Prediction-Model", version=2)
```

### Load Model from Registry

```python
from src.mlflow_utils import get_production_model, get_staging_model

# Load Production model
model = get_production_model()

# Load Staging model
model = get_staging_model()
```

---

## API Integration

### Enable MLflow in API

Set environment variable:

```bash
export USE_MLFLOW=true
uvicorn src.api.main:app --reload
```

Or in code:

```python
import os
os.environ["USE_MLFLOW"] = "true"
```

### API Behavior

When MLflow is enabled:

1. API loads Production model from MLflow registry at startup
2. All predictions use the Production model
3. When new model is promoted to Production, restart API to load it
4. Falls back to legacy registry if MLflow unavailable

### Check Current Model

```bash
curl http://localhost:8000/models
```

---

## Best Practices

### 1. Experiment Organization

- Use descriptive run names: `"LSTM-Baseline"`, `"GRU-Hyperopt"`, etc.
- Tag models appropriately: `model_type`, `experiment`, `dataset_version`
- Add descriptions to registered models

### 2. Model Promotion Workflow

1. **Train** → Model registered in "None" stage
2. **Validate** → Promote to "Staging"
3. **Test in Staging** → Run validation tests
4. **Approve** → Promote to "Production"
5. **Monitor** → Track production performance

### 3. Comparison Strategy

- Train all models with same hyperparameters for fair comparison
- Use consistent train/test splits
- Log all relevant metrics (RMSE, MAE, training time, etc.)

### 4. Production Deployment

- Always test in Staging before Production
- Monitor production model performance
- Keep previous Production models in Archived stage
- Document model changes in descriptions

### 5. MLflow Server Setup

For production/team use:

```bash
# Start MLflow server with database backend
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
```

Then set tracking URI:

```python
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

---

## MLflow UI Features

### View Experiments

1. Open `http://localhost:5000`
2. Click on experiment name
3. See all runs with metrics and parameters

### Compare Runs

1. Select multiple runs
2. Click "Compare"
3. See side-by-side comparison:
   - Parameters differences
   - Metrics comparison
   - Training curves

### Model Registry

1. Click "Models" tab
2. See all registered models
3. View versions and stages
4. Promote/transition models

### Search and Filter

- Filter by parameters: `params.model_type = 'lstm'`
- Filter by metrics: `metrics.test_rmse < 25`
- Sort by any metric or parameter

---

## Troubleshooting

### MLflow Not Tracking

**Issue**: Metrics not appearing in MLflow UI

**Solutions**:
- Check MLflow is installed: `pip list | grep mlflow`
- Verify tracking URI: `mlflow.get_tracking_uri()`
- Check experiment exists: `mlflow.get_experiment_by_name(...)`

### Model Not Found in Registry

**Issue**: `ModelNotFound` error

**Solutions**:
- Verify model was registered: `list_registered_models()`
- Check model name matches: `MODEL_REGISTRY_NAME`
- Ensure model version exists

### API Can't Load MLflow Model

**Issue**: API falls back to legacy registry

**Solutions**:
- Check `USE_MLFLOW` environment variable
- Verify Production model exists in registry
- Check MLflow tracking URI is accessible
- Review API logs for errors

### MLflow Server Connection

**Issue**: Can't connect to MLflow server

**Solutions**:
- Verify server is running: `curl http://mlflow-server:5000`
- Check firewall/network settings
- Verify tracking URI is correct
- Check authentication if required

---

## Advanced Usage

### Custom Metrics

```python
import mlflow

mlflow.log_metric("custom_metric", value, step=epoch)
mlflow.log_metrics({
    "metric1": value1,
    "metric2": value2
})
```

### Log Artifacts

```python
# Log files
mlflow.log_artifact("path/to/file.txt")

# Log directory
mlflow.log_artifacts("path/to/directory")

# Log as artifact with name
mlflow.log_artifact("scaler.pkl", artifact_path="preprocessing")
```

### Model Tags

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.set_model_version_tag(
    name="RUL-Prediction-Model",
    version=1,
    key="deployed_by",
    value="john_doe"
)
```

### Search Runs

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="metrics.test_rmse < 25",
    order_by=["metrics.test_rmse ASC"]
)
```

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Train and Register Model
  run: |
    python -m src.train_with_mlflow \
      --epochs 20 \
      --auto-register \
      --mlflow-uri ${{ secrets.MLFLOW_TRACKING_URI }}

- name: Promote Best Model
  run: |
    python -m src.mlflow_utils promote_best_to_production
```

---

## Summary

MLflow provides:

- ✅ **Automatic tracking** of all experiments
- ✅ **Centralized registry** for model management
- ✅ **Easy comparison** of model architectures
- ✅ **Production workflow** with staging and promotion
- ✅ **API integration** for automatic model updates
- ✅ **Full traceability** of all training runs

This creates a **production-ready MLOps pipeline** that scales from development to production.

---

**Last Updated**: 2024
**Maintained By**: Development Team

