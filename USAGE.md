# Usage Guide: Python Scripts for RUL Prediction

This guide explains how to use the Python scripts extracted from the notebook for training, evaluation, and prediction.

## Project Structure

```
predictive-maintenance-mlops/
├── src/
│   ├── config.py              # Configuration constants
│   ├── utils.py                # Utility functions
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── predict.py              # Prediction script
│   ├── data/
│   │   ├── load_data.py        # Data loading functions
│   │   └── preprocessing.py    # Preprocessing functions
│   └── models/
│       └── architectures.py   # Model definitions
├── notebooks/
│   └── 01_dataset_exploration.ipynb  # Keep for visualization/experiments
├── data/
│   └── raw/                    # Raw data files
├── models/                      # Saved models and scalers
└── requirements.txt
```

## Prerequisites

Make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

## 1. Training (`train.py`)

Train a model on the training dataset.

### Basic Usage

```bash
# Train LSTM model with default parameters
python -m src.train --model lstm

# Train BiLSTM model
python -m src.train --model bilstm

# Train GRU model
python -m src.train --model gru

# Train Transformer model
python -m src.train --model transformer
```

### Advanced Options

```bash
# Custom epochs, batch size, and learning rate
python -m src.train --model lstm --epochs 30 --batch-size 128 --lr 0.0005

# Custom sequence length
python -m src.train --model lstm --seq-length 50
```

### Arguments

- `--model`: Model architecture (`lstm`, `bilstm`, `gru`, `transformer`) - **Required**
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--seq-length`: Sequence length (default: 30)

### Output

- Trained model saved to `models/{model_name}.pth`
- Scaler saved to `models/scaler.pkl` (created during first training run)

## 2. Evaluation (`evaluate.py`)

Evaluate a trained model on the test dataset.

### Basic Usage

```bash
# Evaluate LSTM model
python -m src.evaluate --model lstm

# Evaluate BiLSTM model
python -m src.evaluate --model bilstm

# Evaluate GRU model
python -m src.evaluate --model gru

# Evaluate Transformer model
python -m src.evaluate --model transformer
```

### Advanced Options

```bash
# Use custom model path
python -m src.evaluate --model lstm --model-path models/custom_lstm.pth

# Custom sequence length
python -m src.evaluate --model lstm --seq-length 50
```

### Arguments

- `--model`: Model architecture (`lstm`, `bilstm`, `gru`, `transformer`) - **Required**
- `--model-path`: Path to model checkpoint (default: uses `models/{model_name}.pth`)
- `--seq-length`: Sequence length (default: 30)

### Output

- Prints RMSE and MAE metrics to console

## 3. Prediction (`predict.py`)

Predict RUL for new data (inference).

### Basic Usage

```bash
# Predict RUL for new data
python -m src.predict --model lstm --input data/new_engines.csv

# Specify output file
python -m src.predict --model lstm --input data/new_engines.csv --output results.csv
```

### Input Format

The input CSV file must have the same columns as the training data:
- `engine_id`: Engine identifier
- `cycle`: Cycle number
- `op_setting_1`, `op_setting_2`, `op_setting_3`: Operating settings
- `sensor_1` through `sensor_21`: Sensor readings

**Note**: The `RUL` column is NOT required (it will be predicted).

### Arguments

- `--model`: Model architecture (`lstm`, `bilstm`, `gru`, `transformer`) - **Required**
- `--input`: Path to input CSV file - **Required**
- `--output`: Path to save predictions CSV (default: `predictions.csv`)
- `--model-path`: Path to model checkpoint (default: uses `models/{model_name}.pth`)
- `--seq-length`: Sequence length (default: 30)

### Output

- CSV file with columns: `engine_id`, `predicted_rul`
- Prints summary statistics to console

## Example Workflow

### Complete Training and Evaluation Pipeline

```bash
# 1. Train LSTM model
python -m src.train --model lstm --epochs 20

# 2. Evaluate the trained model
python -m src.evaluate --model lstm

# 3. Train and evaluate other models
python -m src.train --model bilstm --epochs 20
python -m src.evaluate --model bilstm

python -m src.train --model gru --epochs 20
python -m src.evaluate --model gru

python -m src.train --model transformer --epochs 20
python -m src.evaluate --model transformer
```

### Prediction on New Data

```bash
# Prepare your new data in CSV format (same columns as training data)
# Then predict RUL
python -m src.predict --model lstm --input data/new_engines.csv --output predictions.csv
```

## Model Comparison

To compare all models, run evaluation for each:

```bash
python -m src.evaluate --model lstm
python -m src.evaluate --model bilstm
python -m src.evaluate --model gru
python -m src.evaluate --model transformer
```

Or use the notebook (`notebooks/01_dataset_exploration.ipynb`) for visualization and comparison plots.

## Notes

- **Keep the notebook** (`01_dataset_exploration.ipynb`) for:
  - Data exploration and visualization
  - Model comparison plots
  - Experimentation and debugging
  
- **Use scripts** for:
  - Production training pipelines
  - CI/CD integration
  - Automated evaluation
  - Inference on new data

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the project root:

```bash
# From project root
python -m src.train --model lstm

# NOT from src/ directory
```

### File Not Found Errors

- Ensure data files are in `data/raw/` directory
- Ensure models directory exists: `models/`
- Check that scaler exists: `models/scaler.pkl` (created during training)

### Model Not Found

- Train the model first using `train.py`
- Or specify custom path with `--model-path` argument

