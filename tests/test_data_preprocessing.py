"""
Tests for data preprocessing module.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import calculate_rul, generate_sequences, scale_features


class TestRULCalculation:
    """Test RUL calculation functions."""

    def test_calculate_rul_basic(self):
        """Test basic RUL calculation."""
        # Create sample dataframe with engine_id and cycle
        data = {"engine_id": [1, 1, 1, 2, 2, 2], "cycle": [1, 2, 3, 1, 2, 4]}
        df = pd.DataFrame(data)

        result_df = calculate_rul(df)

        assert "RUL" in result_df.columns
        assert len(result_df) == len(df)
        # Engine 1: max cycle is 3, so RUL should be 2, 1, 0
        engine_1_rul = result_df[result_df["engine_id"] == 1]["RUL"].values
        assert all(r >= 0 for r in engine_1_rul)
        assert engine_1_rul[0] == 2  # 3 - 1
        assert engine_1_rul[-1] == 0  # 3 - 3


class TestFeatureScaling:
    """Test feature scaling functions."""

    def test_scale_features(self):
        """Test feature scaling with DataFrame."""
        # Create sample dataframe with all required FEATURE_COLS
        from src.config import FEATURE_COLS

        np.random.seed(42)
        data = {col: np.random.randn(100) * 10 + 5 for col in FEATURE_COLS}
        df = pd.DataFrame(data)

        scaler = StandardScaler()
        scaler.fit(df[FEATURE_COLS])

        df_scaled = scale_features(df, scaler)

        assert df_scaled.shape == df.shape
        # Check that scaling was applied (mean should be ~0, std ~1)
        scaled_values = df_scaled[FEATURE_COLS].mean(axis=0).values
        assert np.allclose(scaled_values, 0, atol=0.1)
        scaled_std = df_scaled[FEATURE_COLS].std(axis=0).values
        assert np.allclose(scaled_std, 1, atol=0.1)


class TestSequenceGeneration:
    """Test sequence generation functions."""

    def test_generate_sequences_basic(self):
        """Test basic sequence generation."""
        # Create sample dataframe with required columns
        from src.config import FEATURE_COLS

        np.random.seed(42)
        n_samples = 100
        n_engines = 5
        seq_length = 10

        # Create dataframe with engine_id, cycle, RUL, and ALL feature columns
        data = {
            "engine_id": np.repeat(range(1, n_engines + 1), n_samples // n_engines),
            "cycle": np.tile(range(1, (n_samples // n_engines) + 1), n_engines),
            "RUL": np.random.randn(n_samples),
        }
        # Add ALL feature columns (required by generate_sequences)
        for col in FEATURE_COLS:
            data[col] = np.random.randn(n_samples)

        df = pd.DataFrame(data)

        X_seq, y_seq = generate_sequences(df, seq_len=seq_length)

        assert len(X_seq) == len(y_seq)
        assert X_seq.shape[1] == seq_length  # sequence length
        assert X_seq.shape[2] == len(FEATURE_COLS)  # number of features
        # Number of sequences should be valid
        assert len(X_seq) > 0
