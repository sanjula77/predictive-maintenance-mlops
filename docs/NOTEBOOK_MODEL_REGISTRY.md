# Model Registry & Versioning - Notebook Cells

Copy and paste these cells directly into your Jupyter notebook.

---

## Cell 1: Import Model Registry Functions

```python
# =========================================
# Cell: Import Model Registry Functions
# =========================================

from src.model_registry import (
    save_model_version,
    load_model_version,
    list_model_versions,
    get_version_info,
    get_latest_version,
    get_next_version
)
from src.models.architectures import RUL_LSTM, RUL_BiLSTM, RUL_GRU, RUL_Transformer
from src.utils import get_device
import torch

print("✅ Model registry functions imported successfully!")
```

---

## Cell 2: Save Model Version (After Training)

```python
# =========================================
# Cell: Save Model Version
# =========================================
# Use this cell AFTER training and evaluating your model

# Example: Save LSTM model version
# Replace these with your actual trained model, scaler, and metrics

# Assuming you have:
# - model: trained PyTorch model
# - scaler: fitted StandardScaler
# - rmse: test RMSE score
# - mae: test MAE score
# - model_type: 'lstm', 'bilstm', 'gru', or 'transformer'

# Save model version (auto-increments: v1, v2, v3...)
version_dir = save_model_version(
    model=model,                    # Your trained model
    scaler=scaler,                  # Your fitted scaler
    rmse=rmse,                      # Test RMSE (e.g., 24.04)
    mae=mae,                        # Test MAE (e.g., 16.81)
    model_type="lstm",              # 'lstm', 'bilstm', 'gru', 'transformer'
    sequence_length=30,             # Sequence length used
    input_features=24,              # Number of input features (optional, auto-detected)
    additional_metadata={           # Optional: add custom metadata
        "epochs": 20,
        "batch_size": 64,
        "learning_rate": 0.001,
        "hidden_size": 64,
    }
)

print(f"\n📦 Model saved to: {version_dir}")
print(f"   Next version will be: v{get_next_version()}")
```

---

## Cell 3: Load Model Version

```python
# =========================================
# Cell: Load Model Version
# =========================================

# Load a specific version (e.g., v1)
version_to_load = 1  # Change this to load different versions

# Choose the model class based on what you saved
# Options: RUL_LSTM, RUL_BiLSTM, RUL_GRU, RUL_Transformer
model_class = RUL_LSTM  # Change based on your model type

# Load model, scaler, and metadata
device = get_device()
loaded_model, loaded_scaler, metadata = load_model_version(
    version=version_to_load,
    model_class=model_class,
    device=device
)

# Display metadata
print("\n📋 Model Metadata:")
print(f"   Version: v{metadata['version']}")
print(f"   Model Type: {metadata['model_type']}")
print(f"   RMSE: {metadata['rmse']:.2f}")
print(f"   MAE: {metadata['mae']:.2f}")
print(f"   Sequence Length: {metadata['sequence_length']}")
print(f"   Input Features: {metadata['input_features']}")
print(f"   Trained: {metadata['timestamp']}")

# Now you can use loaded_model and loaded_scaler for inference!
```

---

## Cell 4: List All Model Versions

```python
# =========================================
# Cell: List All Model Versions
# =========================================

# Get all saved model versions
all_versions = list_model_versions()

if not all_versions:
    print("📭 No model versions found.")
else:
    print(f"📚 Found {len(all_versions)} model version(s):\n")
    print("=" * 80)
    print(f"{'Version':<10} {'Model Type':<15} {'RMSE':<10} {'MAE':<10} {'Timestamp':<20}")
    print("-" * 80)
    
    for meta in all_versions:
        version = meta.get('version', 'N/A')
        model_type = meta.get('model_type', 'N/A')
        rmse = meta.get('rmse', 0)
        mae = meta.get('mae', 0)
        timestamp = meta.get('timestamp', 'N/A')[:19]  # Truncate to date+time
        
        print(f"v{version:<9} {model_type:<15} {rmse:<10.2f} {mae:<10.2f} {timestamp:<20}")
    
    print("=" * 80)
```

---

## Cell 5: Compare Model Versions

```python
# =========================================
# Cell: Compare Model Versions
# =========================================

import pandas as pd

# Get all versions
all_versions = list_model_versions()

if all_versions:
    # Create comparison DataFrame
    comparison_data = []
    for meta in all_versions:
        comparison_data.append({
            'Version': f"v{meta['version']}",
            'Model Type': meta['model_type'],
            'RMSE': meta['rmse'],
            'MAE': meta['mae'],
            'Sequence Length': meta['sequence_length'],
            'Input Features': meta['input_features'],
            'Timestamp': meta['timestamp']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    display(df_comparison)
    
    # Find best model by RMSE
    best_idx = df_comparison['RMSE'].idxmin()
    best_model = df_comparison.loc[best_idx]
    
    print(f"\n🏆 Best Model (Lowest RMSE):")
    print(f"   Version: {best_model['Version']}")
    print(f"   Model Type: {best_model['Model Type']}")
    print(f"   RMSE: {best_model['RMSE']:.2f}")
    print(f"   MAE: {best_model['MAE']:.2f}")
else:
    print("📭 No model versions to compare.")
```

---

## Cell 6: Complete Example - Train, Evaluate, and Save

```python
# =========================================
# Cell: Complete Example - Train, Evaluate, and Save Model Version
# =========================================

from src.data.load_data import load_train_data, load_test_data, load_rul_data
from src.data.preprocessing import (
    calculate_rul,
    fit_scaler,
    scale_features,
    generate_sequences,
    generate_test_sequences,
    load_scaler
)
from src.models.architectures import get_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.config import FEATURE_COLS, SEQ_LENGTH
from src.utils import set_seed, get_device
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm

# Set seed
set_seed(42)
device = get_device()

# 1. Load and preprocess data
print("📂 Loading data...")
train_df = load_train_data()
test_df = load_test_data()
rul_df = load_rul_data()

train_df = calculate_rul(train_df)
scaler = fit_scaler(train_df)
train_scaled = scale_features(train_df, scaler)
test_scaled = scale_features(test_df, scaler)

# 2. Generate sequences
print("⏳ Generating sequences...")
X_train, y_train = generate_sequences(train_scaled, SEQ_LENGTH, FEATURE_COLS)
X_test, y_test = generate_test_sequences(test_scaled, SEQ_LENGTH, rul_df, FEATURE_COLS)

# 3. Create and train model
print("🏗️  Building model...")
model_type = "lstm"  # Change to 'bilstm', 'gru', or 'transformer'
input_size = X_train.shape[2]
model = get_model(model_type, input_size=input_size, seq_len=SEQ_LENGTH)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

print("🚀 Training...")
model, loss_history = train_model(
    model, X_train_tensor, y_train_tensor, device, epochs=20
)

# 4. Evaluate model
print("📊 Evaluating...")
y_pred, rmse, mae = evaluate_model(model, X_test, y_test, device)

print(f"   RMSE: {rmse:.2f}")
print(f"   MAE: {mae:.2f}")

# 5. Save model version
print("\n💾 Saving model version...")
version_dir = save_model_version(
    model=model,
    scaler=scaler,
    rmse=rmse,
    mae=mae,
    model_type=model_type,
    sequence_length=SEQ_LENGTH,
    input_features=input_size,
    additional_metadata={
        "epochs": 20,
        "final_loss": float(loss_history[-1]),
    }
)

print(f"\n✅ Complete! Model saved to: {version_dir}")
```

---

## Cell 7: Load and Use Model for Prediction

```python
# =========================================
# Cell: Load Model Version and Make Predictions
# =========================================

# Load a specific version
version = 1  # Change to load different version
model_class = RUL_LSTM  # Match the model type you saved

loaded_model, loaded_scaler, metadata = load_model_version(
    version=version,
    model_class=model_class,
    device=get_device()
)

# Example: Make prediction on test data
# (Assuming you have X_test from previous cells)

loaded_model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(get_device())
    predictions = loaded_model(X_test_tensor).cpu().numpy().flatten()

print(f"✅ Predictions made using model v{version}")
print(f"   Predicted RUL for {len(predictions)} engines")
print(f"   Mean predicted RUL: {predictions.mean():.2f}")
```

---

## Usage Tips

1. **After Training**: Use Cell 2 to save your model version
2. **To Load**: Use Cell 3 to load any saved version
3. **To Compare**: Use Cell 4 or Cell 5 to see all versions
4. **Complete Workflow**: Use Cell 6 for end-to-end training and saving

## Directory Structure Created

```
models/
├── v1/
│   ├── model.pth
│   ├── scaler.pkl
│   └── metadata.json
├── v2/
│   ├── model.pth
│   ├── scaler.pkl
│   └── metadata.json
└── ...
```

## Metadata JSON Structure

```json
{
  "version": 1,
  "model_type": "lstm",
  "rmse": 24.04,
  "mae": 16.81,
  "timestamp": "2024-01-15T10:30:45.123456",
  "sequence_length": 30,
  "input_features": 24,
  "feature_columns": [...],
  "model_architecture": {
    "type": "RUL_LSTM"
  },
  "epochs": 20,
  "batch_size": 64,
  "learning_rate": 0.001
}
```

