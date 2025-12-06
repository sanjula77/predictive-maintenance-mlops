# Model Registry & Versioning - Scripts Usage Guide

## 📋 Available Scripts

### 1. `train_with_versioning.py` - Train with Automatic Versioning
Trains a model and automatically registers it as a new version.

### 2. `register_model.py` - Register Existing Model
Register an already-trained model as a new version.

### 3. `list_models.py` - List All Versions
View all registered model versions and their metrics.

### 4. `load_model_version.py` - Load Model Version
Load a specific model version for inference.

---

## 🚀 Usage Examples

### Option 1: Train and Auto-Register (Recommended)

```bash
# Train LSTM model with automatic versioning
python -m src.train_with_versioning --model lstm --epochs 20

# Train with custom hyperparameters
python -m src.train_with_versioning --model bilstm --epochs 30 --batch-size 128 --lr 0.0005
```

**What it does:**
- Trains the model
- Evaluates on test set
- Automatically saves to `models/v1/`, `models/v2/`, etc.
- Includes model, scaler, and metadata.json

---

### Option 2: Register Existing Model

If you already have a trained model and want to register it:

```bash
# Register a model with metrics
python -m src.register_model \
    --model-path models/rul_lstm.pth \
    --model-type lstm \
    --rmse 24.04 \
    --mae 16.81

# With additional metadata
python -m src.register_model \
    --model-path models/rul_lstm.pth \
    --scaler-path models/scaler.pkl \
    --model-type lstm \
    --rmse 24.04 \
    --mae 16.81 \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 0.001
```

**What it does:**
- Loads existing model and scaler
- Registers as new version (auto-increments)
- Saves to versioned directory with metadata

---

### Option 3: List All Model Versions

```bash
# List all versions in table format
python -m src.list_models

# Show details for specific version
python -m src.list_models --version 1

# Export as JSON
python -m src.list_models --format json

# Export as CSV
python -m src.list_models --format csv
```

**Output example:**
```
📚 Found 3 model version(s):

┌─────────┬─────────────┬────────┬────────┬─────────────────────┐
│ Version │ Model Type  │ RMSE   │ MAE    │ Timestamp           │
├─────────┼─────────────┼────────┼────────┼─────────────────────┤
│ v1      │ LSTM        │ 24.04  │ 16.81  │ 2024-01-15T10:30:45 │
│ v2      │ BILSTM      │ 22.15  │ 15.23  │ 2024-01-15T11:45:12 │
│ v3      │ GRU         │ 23.50  │ 16.10  │ 2024-01-15T13:20:30 │
└─────────┴─────────────┴────────┴────────┴─────────────────────┘

🏆 Best RMSE: v2 (BILSTM) - 22.15
🏆 Best MAE:  v2 (BILSTM) - 15.23
```

---

### Option 4: Load Model Version

```bash
# Load version 1 for inference
python -m src.load_model_version --version 1 --model-type lstm

# Show info only (don't load model)
python -m src.load_model_version --version 1 --model-type lstm --info-only

# Load and save to custom directory
python -m src.load_model_version \
    --version 1 \
    --model-type lstm \
    --output-dir ./deploy_models
```

---

## 📁 Directory Structure

After using these scripts, your `models/` directory will look like:

```
models/
├── v1/
│   ├── model.pth          # Trained model
│   ├── scaler.pkl         # Fitted scaler
│   └── metadata.json      # Model metadata
├── v2/
│   ├── model.pth
│   ├── scaler.pkl
│   └── metadata.json
├── v3/
│   └── ...
└── scaler.pkl             # Legacy scaler (if exists)
```

---

## 🔄 Complete Workflow Example

### Step 1: Train All Models with Versioning

```bash
# Train LSTM
python -m src.train_with_versioning --model lstm --epochs 20

# Train BiLSTM
python -m src.train_with_versioning --model bilstm --epochs 20

# Train GRU
python -m src.train_with_versioning --model gru --epochs 20

# Train Transformer
python -m src.train_with_versioning --model transformer --epochs 20
```

### Step 2: List All Versions

```bash
python -m src.list_models
```

### Step 3: Compare Versions

```bash
# See details for each version
python -m src.list_models --version 1
python -m src.list_models --version 2
python -m src.list_models --version 3
```

### Step 4: Load Best Model for Deployment

```bash
# Load the best performing version (e.g., v2)
python -m src.load_model_version --version 2 --model-type bilstm --output-dir ./production_model
```

---

## 📊 Metadata Structure

Each `metadata.json` contains:

```json
{
  "version": 1,
  "model_type": "lstm",
  "rmse": 24.04,
  "mae": 16.81,
  "timestamp": "2024-01-15T10:30:45.123456",
  "sequence_length": 30,
  "input_features": 24,
  "feature_columns": ["op_setting_1", "sensor_1", ...],
  "model_architecture": {
    "type": "RUL_LSTM"
  },
  "epochs": 20,
  "batch_size": 64,
  "learning_rate": 0.001
}
```

---

## 💡 Tips

1. **Use `train_with_versioning.py`** for new training runs - it's the easiest way
2. **Use `register_model.py`** if you have existing models from old training
3. **Use `list_models.py`** regularly to track your model experiments
4. **Use `load_model_version.py`** when deploying models to production

---

## 🔧 Integration with Existing Scripts

Your existing scripts still work:

```bash
# Old way (saves to models/ directly)
python -m src.train --model lstm

# New way (saves to models/v1/, models/v2/, etc.)
python -m src.train_with_versioning --model lstm
```

You can register models trained with the old script:

```bash
python -m src.register_model \
    --model-path models/rul_lstm.pth \
    --model-type lstm \
    --rmse 24.04 \
    --mae 16.81
```

