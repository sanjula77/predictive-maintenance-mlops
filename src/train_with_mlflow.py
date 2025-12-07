"""
Training script with full MLflow integration.
Trains all models (LSTM, BiLSTM, GRU, Transformer) and compares them in MLflow.

Usage: python -m src.train_with_mlflow --epochs 20

This script:
- Trains all model architectures
- Tracks all experiments in MLflow
- Compares models automatically
- Registers best model in Model Registry
- Promotes best model to Production
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
    promote_model_to_production,
    register_model_in_mlflow,
    setup_mlflow,
)
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
    """Evaluate model and return RMSE and MAE.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    X_test : torch.Tensor
        Test input data.
    y_test : torch.Tensor
        Test target data.
    device : torch.device
        Device to evaluate on.

    Returns
    -------
    tuple[float, float]
        RMSE and MAE metrics.
    """
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


def train_all_models_with_mlflow(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    seq_length: int = SEQ_LENGTH,
    auto_register: bool = True,
    auto_promote: bool = False,
):
    """Train all model architectures and compare in MLflow.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    seq_length : int
        Sequence length.
    auto_register : bool
        Automatically register models in MLflow registry.
    auto_promote : bool
        Automatically promote best model to Production.
    """
    # Setup MLflow
    setup_mlflow()
    print("📊 MLflow tracking enabled for all models")

    # Set seed
    set_seed(SEED)
    device = get_device()
    print(f"🔥 Using device: {device}")

    # Load and preprocess data (once for all models)
    print("📂 Loading and preprocessing data...")
    train_df = load_train_data()
    train_df = calculate_rul(train_df)
    scaler = fit_scaler(train_df)
    train_scaled = scale_features(train_df, scaler)

    test_df = load_test_data()
    rul_df = load_rul_data()
    test_scaled = scale_features(test_df, scaler)

    # Generate sequences
    X_train, y_train = generate_sequences(train_scaled, seq_length, FEATURE_COLS)
    X_test, y_test = generate_test_sequences(test_scaled, seq_length, rul_df, FEATURE_COLS)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    input_size = X_train.shape[2]

    # Train each model
    model_types = ["lstm", "bilstm", "gru", "transformer"]
    results = {}

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*70}")

        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_type.upper()}-Training"):
            # Log parameters
            log_training_params(
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                sequence_length=seq_length,
                hidden_size=64,  # From config
                num_layers=2,  # From config
                dropout=0.2,  # From config
                seed=SEED,
            )

            # Create and train model
            model = get_model(model_type, input_size=input_size, seq_len=seq_length)
            model, loss_history = train_model(
                model,
                X_train_tensor,
                y_train_tensor,
                device,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                use_mlflow=True,
            )

            # Evaluate
            rmse, mae = evaluate_model(model, X_test_tensor, y_test_tensor, device)
            print(f"   Test RMSE: {rmse:.2f}")
            print(f"   Test MAE: {mae:.2f}")

            # Log final metrics
            log_training_metrics(
                train_loss=loss_history[-1],
                test_rmse=rmse,
                test_mae=mae,
            )

            # Log artifacts
            log_model_artifacts(
                model=model,
                scaler=scaler,
                model_type=model_type,
            )

            # Store results
            results[model_type] = {
                "rmse": rmse,
                "mae": mae,
                "run_id": mlflow.active_run().info.run_id,
            }

            # Register model if requested
            if auto_register:
                model_version = register_model_in_mlflow(
                    run_id=mlflow.active_run().info.run_id,
                    tags={"model_type": model_type, "batch_training": "true"},
                    description=f"{model_type.upper()} - RMSE: {rmse:.2f}, MAE: {mae:.2f}",
                )
                results[model_type]["version"] = model_version
                print(f"📝 Registered {model_type.upper()} as version {model_version}")

    # Find and promote best model from current training session
    print(f"\n{'='*70}")
    print("Model Comparison Results")
    print(f"{'='*70}")
    for model_type, result in results.items():
        print(f"{model_type.upper():<15} RMSE: {result['rmse']:.2f}  MAE: {result['mae']:.2f}")

    # Find best model from current training session (lowest RMSE)
    if results:
        best_model_type = min(results.keys(), key=lambda k: results[k]["rmse"])
        best_result = results[best_model_type]
        
        print(
            f"\n🏆 Best model: {best_model_type.upper()} (RMSE: {best_result['rmse']:.2f}, MAE: {best_result['mae']:.2f})"
        )

        if auto_promote and "version" in best_result:
            version = best_result["version"]
            promote_model_to_production("RUL-Prediction-Model", int(version))
            print(f"🚀 Promoted best model ({best_model_type.upper()} version {version}) to Production")
        elif auto_promote:
            print("⚠️  Warning: Could not auto-promote - model version not found in results")

    print("\n✅ All models trained and tracked in MLflow!")
    print("📊 View results: mlflow ui")


def main():
    parser = argparse.ArgumentParser(description="Train all models with MLflow tracking")
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
        "--no-register",
        action="store_true",
        help="Don't register models in MLflow registry",
    )
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Automatically promote best model to Production",
    )

    args = parser.parse_args()

    train_all_models_with_mlflow(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_length=args.seq_length,
        auto_register=not args.no_register,
        auto_promote=args.auto_promote,
    )


if __name__ == "__main__":
    main()
