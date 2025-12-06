"""
Data loading functions for CMAPSS dataset.
"""
import pandas as pd
from pathlib import Path

from src.config import COLUMN_NAMES, TRAIN_FILE, TEST_FILE, RUL_FILE


def load_train_data() -> pd.DataFrame:
    """Load training data from CMAPSS FD001 dataset.
    
    Returns
    -------
    pd.DataFrame
        Training data with columns defined in COLUMN_NAMES.
    """
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    
    df = pd.read_csv(TRAIN_FILE, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def load_test_data() -> pd.DataFrame:
    """Load test data from CMAPSS FD001 dataset.
    
    Returns
    -------
    pd.DataFrame
        Test data with columns defined in COLUMN_NAMES.
    """
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    
    df = pd.read_csv(TEST_FILE, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def load_rul_data() -> pd.DataFrame:
    """Load RUL (Remaining Useful Life) labels for test engines.
    
    Returns
    -------
    pd.DataFrame
        RUL values for each test engine.
    """
    if not RUL_FILE.exists():
        raise FileNotFoundError(f"RUL file not found: {RUL_FILE}")
    
    df = pd.read_csv(RUL_FILE, sep=r"\s+", header=None, names=["RUL"])
    return df

