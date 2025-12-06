# Usage Guide

Comprehensive usage guide for the Predictive Maintenance MLOps project.

## Table of Contents

- [Quick Start](#quick-start)
- [Training Models](#training-models)
- [Model Management](#model-management)
- [Making Predictions](#making-predictions)
- [API Usage](#api-usage)
- [MLflow Workflow](#mlflow-workflow)
- [Common Workflows](#common-workflows)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Your First Model

```bash
# Train with MLflow (recommended)
python -m src.train_with_mlflow --epochs 20 --auto-promote
```

### 3. Start API

```bash
# With MLflow
USE_MLFLOW=true uvicorn src.api.main:app --reload

# Or with Docker
docker-compose up -d
```

### 4. Make Predictions

```bash
curl -X POST "http://localhost:8000/predict/simple" \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

## Training Models

### Single Model Training

```bash
# Basic training (no versioning)
python -m src.train --model lstm --epochs 20

# With automatic versioning (recommended)
python -m src.train_with_versioning --model lstm --epochs 20
```

### Train All Models

```bash
# Train all models with MLflow
python -m src.train_with_mlflow --epochs 20 --auto-promote
```

### Available Models

- `lstm` - Standard LSTM
- `bilstm` - Bidirectional LSTM
- `gru` - GRU (recommended for speed)
- `transformer` - Transformer encoder

See [Training Guide](TRAINING.md) for detailed information.

## Model Management

### List All Models

```bash
# Legacy registry
python -m src.list_models

# MLflow (via API)
curl http://localhost:8000/models
```

### Load Specific Version

```bash
python -m src.load_model_version --version 1 --model-type lstm
```

### Register Existing Model

```bash
python -m src.register_model \
  --model-path models/rul_lstm.pth \
  --model-type lstm \
  --rmse 24.04 \
  --mae 16.81
```

### Compare Models

```bash
python -m src.compare_models
```

See [Model Registry Guide](MODEL_REGISTRY.md) for details.

## Making Predictions

### Command Line

```bash
python -m src.predict \
  --model lstm \
  --version 1 \
  --input data/test_data.csv \
  --output predictions.csv
```

### Python Script

```python
from src.model_registry import load_model_version
from src.models.architectures import RUL_LSTM
import torch

# Load model
model, scaler, metadata = load_model_version(
    version=1,
    model_class=RUL_LSTM
)

# Make prediction
# ... (see src/predict.py for full example)
```

### API

```bash
curl -X POST "http://localhost:8000/predict/simple" \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_data": [
      [0.0, 0.0, 100.0, 518.67, ...]  // 24 features
    ]  // 30 cycles
  }'
```

See [API Documentation](API.md) for complete API reference.

## API Usage

### Start API Server

```bash
# Development
uvicorn src.api.main:app --reload

# Production
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# With MLflow
USE_MLFLOW=true uvicorn src.api.main:app

# With Docker
docker-compose up -d
```

### API Endpoints

- `GET /health` - Health check
- `GET /models` - List all models
- `GET /models/{version}` - Get model info
- `POST /predict` - Predict RUL (full format)
- `POST /predict/simple` - Predict RUL (simple format)

### Interactive Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

See [API Documentation](API.md) for complete reference.

## MLflow Workflow

### 1. Train with MLflow

```bash
python -m src.train_with_mlflow --epochs 20 --auto-promote
```

### 2. View Experiments

```bash
mlflow ui
# Open http://localhost:5000
```

### 3. Manage Models

- View models: MLflow UI → Models
- Promote to Production: Add alias "production"
- Compare models: Select multiple runs

### 4. Use in API

```bash
USE_MLFLOW=true uvicorn src.api.main:app
```

API automatically uses Production model.

See [MLflow Guide](MLFLOW_GUIDE.md) for detailed information.

## Common Workflows

### Development Workflow

```bash
# 1. Train model
python -m src.train_with_versioning --model gru --epochs 20

# 2. Evaluate
python -m src.evaluate --model gru --version 1

# 3. Test API
uvicorn src.api.main:app --reload

# 4. Make test prediction
curl -X POST "http://localhost:8000/predict/simple" ...
```

### Production Workflow

```bash
# 1. Train all models
python -m src.train_with_mlflow --epochs 20 --auto-promote

# 2. Verify in MLflow UI
mlflow ui

# 3. Deploy API
USE_MLFLOW=true uvicorn src.api.main:app

# Or with Docker
docker-compose up -d
```

### Model Comparison Workflow

```bash
# 1. Train multiple models
python -m src.train_with_mlflow --epochs 20

# 2. Compare in MLflow UI
mlflow ui
# Compare runs, metrics, parameters

# 3. Promote best model
# Via UI or code
from src.mlflow_utils import promote_model_to_production
promote_model_to_production("RUL-Prediction-Model", version=3)
```

### Hyperparameter Tuning Workflow

```bash
# Train with different hyperparameters
python -m src.train_with_mlflow --epochs 20 --lr 0.001
python -m src.train_with_mlflow --epochs 20 --lr 0.0005
python -m src.train_with_mlflow --epochs 30 --lr 0.001

# Compare in MLflow UI
mlflow ui

# Select best configuration
# Promote to Production
```

## Docker Usage

### Start with Docker

```bash
# Build and start
docker-compose up -d

# Or use helper script
./docker/start.sh  # Linux/Mac
docker\start.bat   # Windows
```

### View Logs

```bash
docker-compose logs -f
```

### Stop

```bash
docker-compose down
```

See [Docker Guide](DOCKER.md) for complete Docker documentation.

## Troubleshooting

### Model Not Found

```bash
# Check available models
python -m src.list_models

# Verify model files
ls models/v{version}/
```

### API Not Starting

```bash
# Check if port is in use
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Check logs
uvicorn src.api.main:app --log-level debug
```

### MLflow Issues

```bash
# Check MLflow tracking URI
echo $MLFLOW_TRACKING_URI

# Verify models exist
ls mlruns/models/

# Restart MLflow UI
mlflow ui --backend-store-uri file:///path/to/mlruns
```

## Additional Resources

- [Training Guide](TRAINING.md) - Detailed training instructions
- [Model Registry Guide](MODEL_REGISTRY.md) - Model versioning system
- [API Documentation](API.md) - Complete API reference
- [MLflow Guide](MLFLOW_GUIDE.md) - MLflow integration
- [Docker Guide](DOCKER.md) - Docker deployment
- [Deployment Guide](DEPLOYMENT.md) - Production deployment
