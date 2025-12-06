"""
Configuration file for Predictive Maintenance MLOps project.
Contains all hyperparameters, paths, and constants.
"""

from pathlib import Path

# ===============================
# Project Paths
# ===============================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
SCALER_DIR = PROJECT_ROOT / "models"  # Store scaler alongside models

# Data file paths
TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"

# ===============================
# Hyperparameters
# ===============================
SEED = 42
SEQ_LENGTH = 30  # Length of sequence window
BATCH_SIZE = 64  # Batch size for all models
EPOCHS = 20  # Default number of epochs
LR = 1e-3  # Default learning rate

# Model architecture hyperparameters
HIDDEN_SIZE = 64  # Number of LSTM/GRU units
NUM_LAYERS = 2  # Number of stacked layers
DROPOUT = 0.2  # Dropout rate

# Transformer-specific hyperparameters
TRANSFORMER_DIM_MODEL = 128
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DIM_FEEDFORWARD = 128

# ===============================
# Data Configuration
# ===============================
# CMAPSS FD001 has 26 columns
COLUMN_NAMES = [
    "engine_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]

# Feature columns (exclude engine_id, cycle, RUL)
FEATURE_COLS = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]

# ===============================
# Model Names
# ===============================
MODEL_NAMES = {
    "lstm": "rul_lstm.pth",
    "bilstm": "bi_lstm.pth",
    "gru": "gru.pth",
    "transformer": "transformer_rul.pth",
}

SCALER_NAME = "scaler.pkl"

# ===============================
# Ensure Directories Exist
# ===============================
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SCALER_DIR.mkdir(parents=True, exist_ok=True)
