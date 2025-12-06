"""
Training script with automatic model versioning.
This script trains a model and automatically registers it as a new version.

Usage: python -m src.train_with_versioning --model lstm --epochs 20

The model will be saved to models/v1/, models/v2/, etc. with:
- model.pth: Trained model
- scaler.pkl: Fitted scaler  
- metadata.json: Model metadata (metrics, config, timestamp)
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import (
    SEED,
    SEQ_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LR,
    FEATURE_COLS,
)
from src.data.load_data import load_train_data, load_test_data, load_rul_data
from src.data.preprocessing import (
    calculate_rul,
    fit_scaler,
    scale_features,
    generate_sequences,
    generate_test_sequences,
)
from src.models.architectures import get_model
from src.model_registry import save_model_version
from src.utils import set_seed, get_device


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> tuple[nn.Module, list]:
    """Train a PyTorch model."""
    model = model.to(device)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}")

    return model, loss_history


def evaluate_model(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model and return RMSE and MAE."""
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy().flatten()
    
    y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
    rmse = np.sqrt(mean_squared_error(y_test_np, y_pred))
    mae = mean_absolute_error(y_test_np, y_pred)
    
    return rmse, mae


def main():
    parser = argparse.ArgumentParser(description="Train RUL prediction model with versioning")
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
    parser.add_argument(
        "--no-versioning",
        action="store_true",
        help="Disable automatic versioning (save to models/ directly)",
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
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_df = load_test_data()
    rul_df = load_rul_data()
    test_scaled = scale_features(test_df, scaler)
    X_test, y_test = generate_test_sequences(
        test_scaled, args.seq_length, rul_df, FEATURE_COLS
    )
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    rmse, mae = evaluate_model(model, X_test_tensor, y_test_tensor, device)
    print(f"   Test RMSE: {rmse:.2f}")
    print(f"   Test MAE: {mae:.2f}")
    
    # Save model with versioning
    if not args.no_versioning:
        print("\n💾 Saving model version...")
        version_dir = save_model_version(
            model=model,
            scaler=scaler,
            rmse=rmse,
            mae=mae,
            model_type=args.model,
            sequence_length=args.seq_length,
            input_features=input_size,
            additional_metadata={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "final_loss": float(loss_history[-1]),
            }
        )
        print(f"✅ Training complete! Model saved to: {version_dir}")
    else:
        # Old behavior: save to models/ directly
        from src.config import MODEL_DIR, MODEL_NAMES
        model_path = MODEL_DIR / MODEL_NAMES[args.model]
        torch.save(model.state_dict(), model_path)
        print(f"✅ Training complete! Model saved to: {model_path}")
    
    print(f"   Final training loss: {loss_history[-1]:.4f}")


if __name__ == "__main__":
    main()

