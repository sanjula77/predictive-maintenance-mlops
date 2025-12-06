"""
Neural network architectures for RUL prediction.
"""
import torch
import torch.nn as nn

from src.config import (
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    TRANSFORMER_DIM_MODEL,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_DIM_FEEDFORWARD,
)


class RUL_LSTM(nn.Module):
    """LSTM-based regressor for Remaining Useful Life (RUL)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        output_size: int = 1,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out


class RUL_BiLSTM(nn.Module):
    """Bi-directional LSTM regressor for Remaining Useful Life (RUL)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        output_size: int = 1,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,  # BiLSTM
        )

        # *2 because bidirectional (forward + backward)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out


class RUL_GRU(nn.Module):
    """GRU-based regressor for Remaining Useful Life (RUL)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        output_size: int = 1,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class RUL_Transformer(nn.Module):
    """Transformer Encoder-based regressor for Remaining Useful Life (RUL)."""

    def __init__(
        self,
        input_size: int,
        seq_len: int = 30,
        dim_model: int = TRANSFORMER_DIM_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.seq_len = seq_len

        # Project input features to model dimension
        self.input_proj = nn.Linear(input_size, dim_model)

        # Learnable positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, dim_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Regression head
        self.fc = nn.Linear(dim_model, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.input_proj(x) + self.pos_encoder  # Add positional encoding
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Take the last time step
        return self.fc(x)


def get_model(
    model_name: str,
    input_size: int,
    seq_len: int = 30,
) -> nn.Module:
    """Factory function to create a model by name.
    
    Parameters
    ----------
    model_name : str
        Model name: 'lstm', 'bilstm', 'gru', or 'transformer'.
    input_size : int
        Number of input features.
    seq_len : int, optional
        Sequence length (required for transformer), by default 30.
        
    Returns
    -------
    nn.Module
        Model instance.
    """
    model_name = model_name.lower()
    
    if model_name == "lstm":
        return RUL_LSTM(input_size=input_size)
    elif model_name == "bilstm":
        return RUL_BiLSTM(input_size=input_size)
    elif model_name == "gru":
        return RUL_GRU(input_size=input_size)
    elif model_name == "transformer":
        return RUL_Transformer(input_size=input_size, seq_len=seq_len)
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Choose from: 'lstm', 'bilstm', 'gru', 'transformer'"
        )

