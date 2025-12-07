"""
Data preprocessing functions: RUL calculation, scaling, and sequence generation.
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_COLS, SCALER_DIR, SCALER_NAME


def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Remaining Useful Life (RUL) for each engine.

    RUL is computed as: max_cycle_per_engine - current_cycle

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'engine_id' and 'cycle' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'RUL' column added.
    """
    df = df.copy()

    # Get max cycle per engine
    max_cycles = df.groupby("engine_id")["cycle"].max()

    # Map max cycle to each row
    df["max_cycle"] = df["engine_id"].map(max_cycles)

    # Calculate RUL
    df["RUL"] = df["max_cycle"] - df["cycle"]

    # Drop helper column
    df.drop(columns=["max_cycle"], inplace=True)

    return df


def fit_scaler(train_df: pd.DataFrame, save_path: Optional[Path] = None) -> StandardScaler:
    """Fit StandardScaler on training data features.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training DataFrame with feature columns.
    save_path : Path, optional
        Path to save the scaler. If None, uses default SCALER_DIR/SCALER_NAME.

    Returns
    -------
    StandardScaler
        Fitted scaler.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS])

    if save_path is None:
        save_path = SCALER_DIR / SCALER_NAME

    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, save_path)

    return scaler


def load_scaler(scaler_path: Optional[Path] = None) -> StandardScaler:
    """Load a saved StandardScaler.

    Parameters
    ----------
    scaler_path : Path, optional
        Path to the saved scaler. If None, uses default SCALER_DIR/SCALER_NAME.

    Returns
    -------
    StandardScaler
        Loaded scaler.
    """
    if scaler_path is None:
        scaler_path = SCALER_DIR / SCALER_NAME

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    return joblib.load(scaler_path)


def scale_features(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Scale feature columns using a fitted scaler.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to scale.
    scaler : StandardScaler
        Fitted StandardScaler.

    Returns
    -------
    pd.DataFrame
        DataFrame with scaled features.
    """
    df_scaled = df.copy()
    df_scaled[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    return df_scaled


def generate_sequences(
    df: pd.DataFrame, seq_len: int, feature_cols: Optional[list] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate sequences for training/validation.

    Uses only past `seq_len` cycles to predict RUL at the next time step
    to avoid any future information leakage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'engine_id', feature columns, and 'RUL'.
    seq_len : int
        Length of sequence window.
    feature_cols : list, optional
        List of feature column names. If None, uses FEATURE_COLS from config.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X (sequences) and y (targets) arrays.
        X shape: (n_samples, seq_len, n_features)
        y shape: (n_samples,)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X, y = [], []

    for engine_id in df["engine_id"].unique():
        engine_data = df[df["engine_id"] == engine_id].sort_values("cycle")
        engine_array = engine_data[feature_cols].values
        rul_array = engine_data["RUL"].values

        # Sliding window over time for this engine
        for i in range(len(engine_data) - seq_len):
            X.append(engine_array[i : i + seq_len])
            y.append(rul_array[i + seq_len])

    return np.array(X), np.array(y)


def generate_test_sequences(
    df: pd.DataFrame, seq_len: int, rul_df: pd.DataFrame, feature_cols: Optional[list] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate sequences for test set evaluation.

    For each engine, uses only the last `seq_len` observed cycles
    to predict the final RUL (from RUL_FD001.txt).

    Parameters
    ----------
    df : pd.DataFrame
        Test DataFrame with 'engine_id' and feature columns.
    seq_len : int
        Length of sequence window.
    rul_df : pd.DataFrame
        DataFrame with actual RUL values for each test engine.
    feature_cols : list, optional
        List of feature column names. If None, uses FEATURE_COLS from config.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X_test (sequences) and y_test (actual RUL) arrays.
        X_test shape: (n_engines, seq_len, n_features)
        y_test shape: (n_engines,)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X_test, y_test = [], []
    engine_ids = sorted(df["engine_id"].unique())

    for i, engine_id in enumerate(engine_ids):
        engine_data = df[df["engine_id"] == engine_id].sort_values("cycle")
        engine_array = engine_data[feature_cols].values

        # Only last seq_len cycles to predict final RUL
        if len(engine_array) >= seq_len:
            X_test.append(engine_array[-seq_len:])
        else:  # pad if not enough cycles
            pad = np.zeros((seq_len - len(engine_array), len(feature_cols)))
            X_test.append(np.vstack([pad, engine_array]))

        # Actual RUL from RUL_FD001.txt
        y_test.append(rul_df.iloc[i, 0])

    return np.array(X_test), np.array(y_test)
