"""
Training script for RUL prediction models.
Usage: python -m src.train --model lstm --epochs 20
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    FEATURE_COLS,
    LR,
    MODEL_DIR,
    MODEL_NAMES,
    SEED,
    SEQ_LENGTH,
)
from src.data.load_data import load_train_data
from src.data.preprocessing import calculate_rul, fit_scaler, generate_sequences, scale_features
from src.models.architectures import get_model
from src.utils import get_device, set_seed


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> tuple[nn.Module, list]:
    """Train a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    X_train : torch.Tensor
        Training inputs of shape (n_samples, seq_len, n_features).
    y_train : torch.Tensor
        Training targets of shape (n_samples,).
    device : torch.device
        CPU or GPU device.
    epochs : int, optional
        Number of training epochs, by default EPOCHS.
    batch_size : int, optional
        Batch size, by default BATCH_SIZE.
    lr : float, optional
        Learning rate, by default LR.

    Returns
    -------
    tuple[nn.Module, list]
        Trained model and loss history.
    """
    # Move model to device
    model = model.to(device)

    # DataLoader
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    loss_history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {epoch_loss:.4f}")

    return model, loss_history


def main():
    parser = argparse.ArgumentParser(description="Train RUL prediction model")
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["lstm", "bilstm", "gru", "transformer"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Learning rate (default: {LR})",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=SEQ_LENGTH,
        help=f"Sequence length (default: {SEQ_LENGTH})",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(SEED)

    # Get device
    device = get_device()
    print(f"🔥 Using device: {device}")

    # Load and preprocess data
    print("📂 Loading training data...")
    train_df = load_train_data()
    print(f"   Train dataset shape: {train_df.shape}")

    print("📊 Calculating RUL...")
    train_df = calculate_rul(train_df)

    print("🔧 Fitting scaler...")
    scaler = fit_scaler(train_df)
    print(f"   Scaler saved to: {MODEL_DIR / 'scaler.pkl'}")

    print("📏 Scaling features...")
    train_scaled = scale_features(train_df, scaler)

    print(f"⏳ Generating sequences (seq_len={args.seq_length})...")
    X_train, y_train = generate_sequences(train_scaled, args.seq_length, FEATURE_COLS)
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")

    # Create model
    input_size = X_train.shape[2]
    print(f"🏗️  Building {args.model.upper()} model (input_size={input_size})...")
    model = get_model(args.model, input_size=input_size, seq_len=args.seq_length)
    print(model)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    # Train model
    print(f"🚀 Training {args.model.upper()} model...")
    model, loss_history = train_model(
        model,
        X_train_tensor,
        y_train_tensor,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Save model (legacy: save to models/ directly)
    model_name = MODEL_NAMES[args.model]
    model_path = MODEL_DIR / model_name
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to: {model_path}")

    print("\n💡 Tip: Use 'python -m src.register_model' to register this model as a version")
    print(
        f"   Example: python -m src.register_model --model-path {model_path} --model-type {args.model} --rmse <rmse> --mae <mae>"
    )

    print(f"\n✅ Training complete! Final loss: {loss_history[-1]:.4f}")


if __name__ == "__main__":
    main()
