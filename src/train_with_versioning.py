"""
Training script with automatic model versioning and MLflow tracking.
This script trains a model and automatically registers it in MLflow Model Registry.

Usage: python -m src.train_with_versioning --model lstm --epochs 20

Features:
- Automatic experiment tracking with MLflow
- Model registration in MLflow Model Registry
- Comparison of LSTM/GRU/Transformer models
- Full traceability of training runs
"""

import argparse

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    DROPOUT,
    EPOCHS,
    FEATURE_COLS,
    HIDDEN_SIZE,
    LR,
    NUM_LAYERS,
    SEED,
    SEQ_LENGTH,
)
from src.data.load_data import load_rul_data, load_test_data, load_train_data
from src.data.preprocessing import (
    calculate_rul,
    fit_scaler,
    generate_sequences,
    generate_test_sequences,
    scale_features,
)
from src.mlflow_utils import (
    get_best_model_by_metric,
    log_model_artifacts,
    log_training_metrics,
    log_training_params,
    register_model_in_mlflow,
    setup_mlflow,
)
from src.model_registry import save_model_version
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
    use_mlflow: bool = True,
) -> tuple[nn.Module, list]:
    """Train a PyTorch model with optional MLflow tracking.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    X_train : torch.Tensor
        Training input data.
    y_train : torch.Tensor
        Training target data.
    device : torch.device
        Device to train on.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    use_mlflow : bool
        Whether to log metrics to MLflow.

    Returns
    -------
    tuple[nn.Module, list]
        Trained model and loss history.
    """
    model = model.to(device)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

        # Log to MLflow
        if use_mlflow:
            log_training_metrics(train_loss=epoch_loss, epoch=epoch)

    return model, loss_history


def evaluate_model(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model and return RMSE and MAE."""
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: local file system)",
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register model in MLflow Model Registry",
    )
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Automatically promote best model to Production (based on RMSE)",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(SEED)

    # Setup MLflow
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        setup_mlflow(tracking_uri=args.mlflow_uri)
        print("📊 MLflow tracking enabled")

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

    # Start MLflow run
    run_name = f"{args.model.upper()}-Training"
    if use_mlflow:
        mlflow.start_run(run_name=run_name)

        # Log hyperparameters
        log_training_params(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            sequence_length=args.seq_length,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            seed=SEED,
        )

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
        use_mlflow=use_mlflow,
    )

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_df = load_test_data()
    rul_df = load_rul_data()
    test_scaled = scale_features(test_df, scaler)
    X_test, y_test = generate_test_sequences(test_scaled, args.seq_length, rul_df, FEATURE_COLS)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    rmse, mae = evaluate_model(model, X_test_tensor, y_test_tensor, device)
    print(f"   Test RMSE: {rmse:.2f}")
    print(f"   Test MAE: {mae:.2f}")

    # Log final metrics to MLflow
    if use_mlflow:
        log_training_metrics(
            train_loss=loss_history[-1],
            test_rmse=rmse,
            test_mae=mae,
        )

        # Log model and scaler artifacts
        log_model_artifacts(
            model=model,
            scaler=scaler,
            model_type=args.model,
        )

        # Register model if requested
        if args.register_model or args.auto_promote:
            run_id = mlflow.active_run().info.run_id
            model_version = register_model_in_mlflow(
                run_id=run_id,
                tags={"model_type": args.model, "auto_trained": "true"},
                description=f"{args.model.upper()} model trained with RMSE={rmse:.2f}, MAE={mae:.2f}",
            )
            print(f"📝 Registered model version {model_version} in MLflow")

            # Auto-promote if best model
            if args.auto_promote:
                best_model = get_best_model_by_metric("test_rmse", ascending=True)
                if best_model and best_model["run_id"] == run_id:
                    from src.mlflow_utils import promote_model_to_production

                    promote_model_to_production("RUL-Prediction-Model", int(model_version))
                    print(f"🚀 Auto-promoted model version {model_version} to Production")

        mlflow.end_run()

    # Save model with versioning (legacy system)
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
            },
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
