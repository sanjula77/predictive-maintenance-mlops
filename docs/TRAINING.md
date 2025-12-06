# Training Guide

Complete guide to training models for Predictive Maintenance RUL prediction.

## Table of Contents

- [Training Options](#training-options)
- [Model Architectures](#model-architectures)
- [Hyperparameters](#hyperparameters)
- [MLflow Training](#mlflow-training)
- [Training Workflows](#training-workflows)
- [Best Practices](#best-practices)

## Training Options

### Option 1: Train Single Model (Basic)

```bash
python -m src.train --model lstm --epochs 20
```

**Features:**
- Trains one model at a time
- Saves to `models/` directory (legacy format)
- No automatic versioning
- Manual registration required

**Use Cases:**
- Quick experiments
- Testing new architectures
- Development and debugging

### Option 2: Train with Automatic Versioning (Recommended)

```bash
python -m src.train_with_versioning --model lstm --epochs 20
```

**Features:**
- Automatic version assignment
- Test set evaluation
- Metadata tracking
- Model and scaler saving
- Ready for production

**What It Does:**
1. Trains the model
2. Evaluates on test set
3. Calculates RMSE and MAE
4. Saves to versioned directory (`models/v{version}/`)
5. Generates metadata JSON

### Option 3: Train All Models with MLflow (Best for Production)

```bash
python -m src.train_with_mlflow --epochs 20 --auto-promote
```

**Features:**
- Trains all 4 models (LSTM, BiLSTM, GRU, Transformer)
- MLflow experiment tracking
- Automatic model registration
- Model comparison
- Auto-promote best model to Production

**What It Does:**
1. Trains all model architectures
2. Logs all experiments to MLflow
3. Registers all models in MLflow registry
4. Compares models automatically
5. Promotes best model (lowest RMSE) to Production

**Output:**
```
Training LSTM model...
Training BiLSTM model...
Training GRU model...
Training Transformer model...

Model Comparison Results:
LSTM          RMSE: 24.03  MAE: 16.80
BiLSTM        RMSE: 30.24  MAE: 20.54
GRU           RMSE: 22.49  MAE: 16.79  ← Best!
Transformer   RMSE: 28.13  MAE: 20.03

🏆 Best model: GRU (RMSE: 22.49)
🚀 Promoted best model (version 3) to Production
```

### Option 4: Train All Models (Legacy)

```bash
python -m src.train_all --epochs 20
```

**Features:**
- Trains all models sequentially
- Saves to legacy format
- No MLflow integration
- Manual comparison required

## Model Architectures

### Available Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `lstm` | Standard LSTM | Baseline, general purpose |
| `bilstm` | Bidirectional LSTM | Better context understanding |
| `gru` | Gated Recurrent Unit | Faster training, similar performance |
| `transformer` | Transformer Encoder | Attention-based, state-of-the-art |

### Model Selection

```bash
# Train specific model
python -m src.train_with_versioning --model gru --epochs 20

# Train all models (MLflow)
python -m src.train_with_mlflow --epochs 20
```

## Hyperparameters

### Command-Line Arguments

```bash
python -m src.train_with_versioning \
  --model lstm \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.001 \
  --seq-length 30
```

### Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | Required | Model type (lstm, bilstm, gru, transformer) |
| `--epochs` | 20 | Number of training epochs |
| `--batch-size` | 64 | Batch size for training |
| `--lr` | 0.001 | Learning rate |
| `--seq-length` | 30 | Sequence length for time series |

### Configuration File

Default values are in `src/config.py`:

```python
EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
SEQ_LENGTH = 30
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
```

## MLflow Training

### Basic MLflow Training

```bash
python -m src.train_with_mlflow --epochs 20
```

### With Auto-Promotion

```bash
python -m src.train_with_mlflow --epochs 20 --auto-promote
```

### Custom Parameters

```bash
python -m src.train_with_mlflow \
  --epochs 50 \
  --batch-size 128 \
  --lr 0.0005 \
  --auto-promote
```

### Options

- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--lr`: Learning rate
- `--seq-length`: Sequence length
- `--auto-promote`: Automatically promote best model to Production
- `--no-register`: Don't register models (only track experiments)

## Training Workflows

### Development Workflow

```bash
# 1. Quick experiment
python -m src.train --model lstm --epochs 5

# 2. Full training with versioning
python -m src.train_with_versioning --model lstm --epochs 20

# 3. Evaluate
python -m src.evaluate --model lstm --version 1

# 4. Compare with other models
python -m src.compare_models
```

### Production Workflow

```bash
# 1. Train all models with MLflow
python -m src.train_with_mlflow --epochs 20 --auto-promote

# 2. View results in MLflow UI
mlflow ui

# 3. Verify Production model
curl http://localhost:8000/models

# 4. Deploy API
USE_MLFLOW=true uvicorn src.api.main:app
```

### Hyperparameter Tuning Workflow

```bash
# Train multiple configurations
python -m src.train_with_mlflow --epochs 20 --lr 0.001
python -m src.train_with_mlflow --epochs 20 --lr 0.0005
python -m src.train_with_mlflow --epochs 30 --lr 0.001

# Compare in MLflow UI
mlflow ui

# Select best configuration
# Promote to Production via MLflow UI or code
```

## Best Practices

### 1. Always Use Versioning

```bash
# ✅ Good
python -m src.train_with_versioning --model lstm --epochs 20

# ❌ Bad (overwrites model)
python -m src.train --model lstm --epochs 20
```

### 2. Use MLflow for Production

```bash
# ✅ Good (production-ready)
python -m src.train_with_mlflow --epochs 20 --auto-promote

# ⚠️ OK (development)
python -m src.train_with_versioning --model lstm --epochs 20
```

### 3. Evaluate Before Promoting

```bash
# Train
python -m src.train_with_mlflow --epochs 20

# Check results in MLflow UI
mlflow ui

# Verify metrics before promoting
# Then promote best model
```

### 4. Track Hyperparameters

- Use MLflow to track all hyperparameter changes
- Document why you changed parameters
- Compare different configurations

### 5. Regular Retraining

- Retrain periodically with new data
- Compare new models with existing ones
- Only promote if performance improves

## Monitoring Training

### During Training

Training progress is shown with:
- Progress bars (tqdm)
- Epoch loss
- Estimated time remaining

### After Training

```bash
# View training results
python -m src.list_models

# Or in MLflow UI
mlflow ui
```

### Metrics Tracked

- **Training Loss**: Per-epoch training loss
- **Test RMSE**: Root Mean Squared Error on test set
- **Test MAE**: Mean Absolute Error on test set
- **Final Loss**: Final training loss

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python -m src.train_with_versioning --model lstm --batch-size 32

# Or use CPU
# Set CUDA_VISIBLE_DEVICES="" to force CPU
```

### Slow Training

- Use GPU if available (automatic)
- Reduce sequence length
- Use GRU instead of LSTM (faster)
- Reduce hidden size

### Poor Performance

- Increase epochs
- Adjust learning rate
- Try different architectures
- Check data preprocessing

## Next Steps

After training:

1. **Evaluate Models**: `python -m src.evaluate --model lstm`
2. **Compare Models**: `python -m src.compare_models`
3. **Deploy Best Model**: Use MLflow Production model
4. **Monitor Performance**: Track predictions in production

See also:
- [Model Registry Guide](MODEL_REGISTRY.md)
- [MLflow Guide](MLFLOW_GUIDE.md)
- [API Documentation](API.md)
