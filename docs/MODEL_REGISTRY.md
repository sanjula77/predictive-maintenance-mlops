# Model Registry & Versioning Guide

Complete guide to the model registry and versioning system for Predictive Maintenance MLOps.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Model Versioning System](#model-versioning-system)
- [Available Scripts](#available-scripts)
- [Usage Examples](#usage-examples)
- [MLflow Integration](#mlflow-integration)
- [Best Practices](#best-practices)

## Overview

The model registry automatically manages model versions with:

- **Model Weights** (`.pth`) - Trained PyTorch model
- **Preprocessing Scaler** (`.pkl`) - StandardScaler for feature normalization
- **Metadata** (`.json`) - Metrics, configuration, and timestamp

Each version is stored in `models/v{version}/` directory with complete traceability.

## Quick Start

```bash
# Train with automatic versioning (recommended)
python -m src.train_with_versioning --model lstm --epochs 20

# List all versions
python -m src.list_models

# Load a specific version
python -m src.load_model_version --version 1 --model-type lstm
```

## Model Versioning System

### Directory Structure

```
models/
├── v1/
│   ├── model.pth          # Model weights
│   ├── scaler.pkl         # Preprocessing scaler
│   └── metadata.json      # Version metadata
├── v2/
│   └── ...
└── scaler.pkl             # Legacy scaler (deprecated)
```

### Metadata Format

```json
{
  "version": 1,
  "model_type": "lstm",
  "rmse": 24.04,
  "mae": 16.81,
  "timestamp": "2024-01-15T10:30:45.123456",
  "sequence_length": 30,
  "input_features": 24,
  "epochs": 20,
  "batch_size": 64,
  "learning_rate": 0.001,
  "final_loss": 544.53
}
```

## Available Scripts

### 1. Train with Versioning

**Script**: `src/train_with_versioning.py`

Trains a model and automatically registers it as a new version.

```bash
python -m src.train_with_versioning \
  --model lstm \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.001
```

**Features:**
- Automatic version assignment
- Test set evaluation
- Metadata generation
- Model and scaler saving

### 2. Register Existing Model

**Script**: `src/register_model.py`

Register an already-trained model as a new version.

```bash
python -m src.register_model \
  --model-path models/rul_lstm.pth \
  --model-type lstm \
  --rmse 24.04 \
  --mae 16.81
```

**Use Cases:**
- Register models trained outside the pipeline
- Re-register with updated metrics
- Import external models

### 3. List All Versions

**Script**: `src/list_models.py`

View all registered model versions and their metrics.

```bash
python -m src.list_models
```

**Output:**
```
📋 Model Versions:
Version 3: GRU - RMSE: 22.49, MAE: 16.79 (2024-01-15 10:30:45)
Version 2: BiLSTM - RMSE: 30.24, MAE: 20.54 (2024-01-15 09:15:30)
Version 1: LSTM - RMSE: 24.04, MAE: 16.81 (2024-01-15 08:00:00)
```

### 4. Load Model Version

**Script**: `src/load_model_version.py`

Load a specific model version for inference or evaluation.

```bash
python -m src.load_model_version \
  --version 1 \
  --model-type lstm
```

**Options:**
- `--info-only`: Show metadata without loading model
- `--version`: Version number to load
- `--model-type`: Model architecture type

## Usage Examples

### Complete Workflow

```bash
# 1. Train and register model
python -m src.train_with_versioning --model gru --epochs 20

# 2. List all versions
python -m src.list_models

# 3. Evaluate specific version
python -m src.evaluate --model gru --version 3

# 4. Load for prediction
python -m src.load_model_version --version 3 --model-type gru
```

### Comparing Versions

```bash
# Train multiple versions
python -m src.train_with_versioning --model lstm --epochs 20
python -m src.train_with_versioning --model gru --epochs 20

# Compare all versions
python -m src.list_models

# Compare specific models
python -m src.compare_models --version1 1 --version2 2
```

## MLflow Integration

The project supports both legacy registry and MLflow registry:

### Legacy Registry (File-Based)

- Models stored in `models/v{version}/`
- Metadata in JSON files
- Managed by `model_registry.py`

### MLflow Registry (Recommended)

- Models stored in MLflow Model Registry
- Automatic versioning and aliasing
- Production model promotion
- See [MLflow Guide](MLFLOW_GUIDE.md) for details

### Using MLflow

```bash
# Train with MLflow (registers automatically)
python -m src.train_with_mlflow --epochs 20 --auto-promote

# API automatically uses Production model
USE_MLFLOW=true uvicorn src.api.main:app
```

## Best Practices

1. **Always Version Models**: Never overwrite existing models
2. **Include Metadata**: Always provide RMSE/MAE when registering
3. **Use Descriptive Versions**: Track what changed between versions
4. **Regular Evaluation**: Evaluate all versions before promoting
5. **MLflow for Production**: Use MLflow registry for production deployments
6. **Backup Models**: Keep backups of important versions
7. **Document Changes**: Note hyperparameter changes in version notes

## Model Promotion

### Legacy Registry

Manually track which version is in production (no automatic promotion).

### MLflow Registry

```python
from src.mlflow_utils import promote_model_to_production

# Promote best model to Production
promote_model_to_production("RUL-Prediction-Model", version=3)
```

Or via MLflow UI:
1. Open MLflow UI
2. Go to Models → RUL-Prediction-Model
3. Select version
4. Add alias "production"

## Troubleshooting

### Model Not Found

```bash
# Check if version exists
python -m src.list_models

# Verify directory structure
ls models/v{version}/
```

### Metadata Issues

```bash
# View metadata
python -m src.load_model_version --version 1 --model-type lstm --info-only

# Re-register with correct metadata
python -m src.register_model --model-path models/v1/model.pth --model-type lstm --rmse 24.04 --mae 16.81
```

### Version Conflicts

Versions are auto-incremented. If you need to reset:
1. Remove `models/v{version}/` directories
2. Next registration will start from v1

## API Integration

The API can use both registries:

```bash
# Legacy registry (default)
uvicorn src.api.main:app

# MLflow registry
USE_MLFLOW=true uvicorn src.api.main:app
```

See [API Documentation](API.md) for details.
