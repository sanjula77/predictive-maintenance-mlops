"""
Tests for model architectures.
"""

import numpy as np
import pytest
import torch

from src.config import FEATURE_COLS, SEQ_LENGTH
from src.models.architectures import RUL_GRU, RUL_LSTM, RUL_BiLSTM, RUL_Transformer, get_model


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    batch_size = 4
    seq_length = SEQ_LENGTH
    input_size = len(FEATURE_COLS)
    return torch.randn(batch_size, seq_length, input_size)


class TestModelArchitectures:
    """Test model architecture classes."""

    def test_lstm_creation(self):
        """Test LSTM model can be created."""
        model = RUL_LSTM(input_size=len(FEATURE_COLS))
        assert model is not None
        assert isinstance(model, RUL_LSTM)

    def test_lstm_forward(self, sample_input):
        """Test LSTM forward pass."""
        model = RUL_LSTM(input_size=sample_input.shape[2])
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        assert output.shape == (sample_input.shape[0], 1)

    def test_bilstm_creation(self):
        """Test BiLSTM model can be created."""
        model = RUL_BiLSTM(input_size=len(FEATURE_COLS))
        assert model is not None
        assert isinstance(model, RUL_BiLSTM)

    def test_bilstm_forward(self, sample_input):
        """Test BiLSTM forward pass."""
        model = RUL_BiLSTM(input_size=sample_input.shape[2])
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        assert output.shape == (sample_input.shape[0], 1)

    def test_gru_creation(self):
        """Test GRU model can be created."""
        model = RUL_GRU(input_size=len(FEATURE_COLS))
        assert model is not None
        assert isinstance(model, RUL_GRU)

    def test_gru_forward(self, sample_input):
        """Test GRU forward pass."""
        model = RUL_GRU(input_size=sample_input.shape[2])
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        assert output.shape == (sample_input.shape[0], 1)

    def test_transformer_creation(self):
        """Test Transformer model can be created."""
        model = RUL_Transformer(input_size=len(FEATURE_COLS))
        assert model is not None
        assert isinstance(model, RUL_Transformer)

    def test_transformer_forward(self, sample_input):
        """Test Transformer forward pass."""
        model = RUL_Transformer(input_size=sample_input.shape[2])
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        assert output.shape == (sample_input.shape[0], 1)


class TestModelFactory:
    """Test model factory function."""

    def test_get_model_lstm(self):
        """Test get_model for LSTM."""
        model = get_model("lstm", input_size=len(FEATURE_COLS))
        assert isinstance(model, RUL_LSTM)

    def test_get_model_bilstm(self):
        """Test get_model for BiLSTM."""
        model = get_model("bilstm", input_size=len(FEATURE_COLS))
        assert isinstance(model, RUL_BiLSTM)

    def test_get_model_gru(self):
        """Test get_model for GRU."""
        model = get_model("gru", input_size=len(FEATURE_COLS))
        assert isinstance(model, RUL_GRU)

    def test_get_model_transformer(self):
        """Test get_model for Transformer."""
        model = get_model("transformer", input_size=len(FEATURE_COLS))
        assert isinstance(model, RUL_Transformer)

    def test_get_model_invalid(self):
        """Test get_model with invalid model type."""
        with pytest.raises(ValueError):
            get_model("invalid", input_size=len(FEATURE_COLS))
