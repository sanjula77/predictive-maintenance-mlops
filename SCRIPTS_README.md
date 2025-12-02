# Python Scripts Conversion Summary

## ✅ What Was Created

The notebook `01_dataset_exploration.ipynb` has been successfully converted into a modular Python script structure:

### Core Modules

1. **`src/config.py`** - Centralized configuration
   - All hyperparameters (SEED, SEQ_LENGTH, BATCH_SIZE, EPOCHS, LR)
   - Paths (DATA_DIR, MODEL_DIR, file paths)
   - Model names and feature columns

2. **`src/utils.py`** - Utility functions
   - `set_seed()` - Reproducibility
   - `get_device()` - Device detection (CPU/CUDA)

3. **`src/data/load_data.py`** - Data loading
   - `load_train_data()` - Load training data
   - `load_test_data()` - Load test data
   - `load_rul_data()` - Load RUL labels

4. **`src/data/preprocessing.py`** - Data preprocessing
   - `calculate_rul()` - Compute RUL from cycles
   - `fit_scaler()` / `load_scaler()` - Scaler management
   - `scale_features()` - Feature scaling
   - `generate_sequences()` - Training sequence generation
   - `generate_test_sequences()` - Test sequence generation

5. **`src/models/architectures.py`** - Model definitions
   - `RUL_LSTM` - LSTM model
   - `RUL_BiLSTM` - Bidirectional LSTM
   - `RUL_GRU` - GRU model
   - `RUL_Transformer` - Transformer encoder
   - `get_model()` - Factory function

### Main Scripts

6. **`src/train.py`** - Training script
   ```bash
   python -m src.train --model lstm --epochs 20
   ```

7. **`src/evaluate.py`** - Evaluation script
   ```bash
   python -m src.evaluate --model lstm
   ```

8. **`src/predict.py`** - Prediction script
   ```bash
   python -m src.predict --model lstm --input data.csv
   ```

9. **`src/compare_models.py`** - Model comparison script
   ```bash
   python -m src.compare_models
   ```

## 📁 Project Structure

```
predictive-maintenance-mlops/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Main evaluation script
│   ├── predict.py            # Main prediction script
│   ├── compare_models.py     # Compare all models
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── preprocessing.py
│   └── models/
│       ├── __init__.py
│       └── architectures.py
├── notebooks/
│   └── 01_dataset_exploration.ipynb  # Keep for visualization
├── data/
│   └── raw/                  # Data files
├── models/                   # Saved models & scalers
├── USAGE.md                  # Detailed usage guide
└── SCRIPTS_README.md         # This file
```

## 🎯 Key Benefits

### ✅ Clean Separation of Concerns
- **Data loading** → `src/data/load_data.py`
- **Preprocessing** → `src/data/preprocessing.py`
- **Model definitions** → `src/models/architectures.py`
- **Training logic** → `src/train.py`
- **Evaluation logic** → `src/evaluate.py`
- **Inference logic** → `src/predict.py`

### ✅ Reusable Code
- Functions can be imported and reused
- Easy to extend with new models or preprocessing steps
- Configuration centralized in one place

### ✅ Production Ready
- Command-line interfaces for all scripts
- Proper error handling
- Reproducible (seeding, consistent paths)
- No data leakage (same logic as notebook)

### ✅ CI/CD Friendly
- Can be run in automated pipelines
- Easy to integrate with MLflow, DVC, etc.
- Scripts can be containerized

### ✅ Deployment Ready
- `predict.py` can be integrated into FastAPI/Flask
- Models and scalers are saved separately
- Clear input/output interfaces

## 📝 Usage Examples

### Quick Start

```bash
# 1. Train a model
python -m src.train --model lstm

# 2. Evaluate the model
python -m src.evaluate --model lstm

# 3. Compare all models
python -m src.compare_models

# 4. Predict on new data
python -m src.predict --model lstm --input new_data.csv
```

### Advanced Usage

```bash
# Train with custom hyperparameters
python -m src.train --model lstm --epochs 30 --batch-size 128 --lr 0.0005

# Evaluate with custom model path
python -m src.evaluate --model lstm --model-path models/custom.pth

# Predict with custom output
python -m src.predict --model lstm --input data.csv --output results.csv
```

## 🔄 Notebook vs Scripts

### Keep Notebook For:
- ✅ Data exploration and visualization
- ✅ Model comparison plots
- ✅ Experimentation and debugging
- ✅ Interactive analysis

### Use Scripts For:
- ✅ Production training pipelines
- ✅ Automated evaluation
- ✅ CI/CD integration
- ✅ Inference on new data
- ✅ Model deployment

## ✨ Features

- **No Data Leakage**: Same preprocessing logic as notebook
- **Reproducible**: Seeding and consistent paths
- **Modular**: Easy to extend and modify
- **Well Documented**: Docstrings and usage guides
- **Type Hints**: Better IDE support and error detection
- **Error Handling**: Proper file existence checks

## 🚀 Next Steps

1. **Train all models**:
   ```bash
   python -m src.train --model lstm
   python -m src.train --model bilstm
   python -m src.train --model gru
   python -m src.train --model transformer
   ```

2. **Compare models**:
   ```bash
   python -m src.compare_models
   ```

3. **Use for inference**:
   ```bash
   python -m src.predict --model lstm --input your_data.csv
   ```

4. **Integrate with MLflow** (optional):
   - Add MLflow tracking to `train.py`
   - Log metrics, parameters, and artifacts

5. **Create API** (optional):
   - Use `predict.py` logic in FastAPI endpoint
   - Serve models via REST API

## 📚 Documentation

- See `USAGE.md` for detailed usage instructions
- See docstrings in each module for function documentation
- See `src/config.py` for all configuration options

