"""
Tests for configuration module.
"""

from pathlib import Path

import pytest

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    DROPOUT,
    EPOCHS,
    FEATURE_COLS,
    HIDDEN_SIZE,
    LR,
    MODEL_DIR,
    MODEL_NAMES,
    NUM_LAYERS,
    PROJECT_ROOT,
    SEED,
    SEQ_LENGTH,
)


class TestConfigPaths:
    """Test configuration paths."""

    def test_project_root_exists(self):
        """Test that PROJECT_ROOT is a valid path."""
        assert isinstance(PROJECT_ROOT, Path)
        assert PROJECT_ROOT.exists()

    def test_data_dir_path(self):
        """Test DATA_DIR path structure."""
        assert isinstance(DATA_DIR, Path)
        # Handle both Windows and Unix path separators
        data_dir_str = str(DATA_DIR).replace("\\", "/")
        assert data_dir_str.endswith("data/raw")

    def test_model_dir_path(self):
        """Test MODEL_DIR path structure."""
        assert isinstance(MODEL_DIR, Path)
        assert str(MODEL_DIR).endswith("models")


class TestConfigHyperparameters:
    """Test hyperparameters configuration."""

    def test_seed_is_int(self):
        """Test SEED is an integer."""
        assert isinstance(SEED, int)
        assert SEED == 42

    def test_sequence_length(self):
        """Test SEQ_LENGTH is positive."""
        assert isinstance(SEQ_LENGTH, int)
        assert SEQ_LENGTH > 0

    def test_batch_size(self):
        """Test BATCH_SIZE is positive."""
        assert isinstance(BATCH_SIZE, int)
        assert BATCH_SIZE > 0

    def test_epochs(self):
        """Test EPOCHS is positive."""
        assert isinstance(EPOCHS, int)
        assert EPOCHS > 0

    def test_learning_rate(self):
        """Test LR is positive float."""
        assert isinstance(LR, float)
        assert 0 < LR < 1

    def test_hidden_size(self):
        """Test HIDDEN_SIZE is positive."""
        assert isinstance(HIDDEN_SIZE, int)
        assert HIDDEN_SIZE > 0

    def test_num_layers(self):
        """Test NUM_LAYERS is positive."""
        assert isinstance(NUM_LAYERS, int)
        assert NUM_LAYERS > 0

    def test_dropout(self):
        """Test DROPOUT is between 0 and 1."""
        assert isinstance(DROPOUT, float)
        assert 0 <= DROPOUT < 1


class TestConfigData:
    """Test data-related configuration."""

    def test_feature_cols_not_empty(self):
        """Test FEATURE_COLS is not empty."""
        assert isinstance(FEATURE_COLS, list)
        assert len(FEATURE_COLS) > 0

    def test_model_names_dict(self):
        """Test MODEL_NAMES is a dictionary."""
        assert isinstance(MODEL_NAMES, dict)
        assert len(MODEL_NAMES) > 0
