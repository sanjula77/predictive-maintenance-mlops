"""
Evaluation script for RUL prediction models.
Usage: python -m src.evaluate --model lstm
"""
import argparse
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.config import (
    SEED,
    SEQ_LENGTH,
    MODEL_DIR,
    MODEL_NAMES,
    FEATURE_COLS,
)
from src.data.load_data import load_test_data, load_rul_data
from src.data.preprocessing import (
    load_scaler,
    scale_features,
    generate_test_sequences,
)
from src.models.architectures import get_model
from src.utils import set_seed, get_device


def evaluate_model(
    model: torch.nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float, float]:
    """Evaluate a model on test data.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    X_test : np.ndarray
        Test sequences of shape (n_engines, seq_len, n_features).
    y_test : np.ndarray
        Actual RUL values of shape (n_engines,).
    device : torch.device
        CPU or GPU device.
        
    Returns
    -------
    tuple[np.ndarray, float, float]
        Predictions, RMSE, and MAE.
    """
    model.eval()
    model = model.to(device)
    
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return y_pred, rmse, mae


def main():
    parser = argparse.ArgumentParser(description="Evaluate RUL prediction model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "bilstm", "gru", "transformer"],
        help="Model architecture to evaluate",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=SEQ_LENGTH,
        help=f"Sequence length (default: {SEQ_LENGTH})",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint (default: uses MODEL_DIR/MODEL_NAMES[model])",
    )
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Get device
    device = get_device()
    print(f"🔥 Using device: {device}")
    
    # Load test data
    print("📂 Loading test data...")
    test_df = load_test_data()
    rul_df = load_rul_data()
    print(f"   Test dataset shape: {test_df.shape}")
    print(f"   RUL dataset shape: {rul_df.shape}")
    
    # Load scaler
    print("🔧 Loading scaler...")
    scaler = load_scaler()
    
    # Scale features
    print("📏 Scaling features...")
    test_scaled = scale_features(test_df, scaler)
    
    # Generate test sequences
    print(f"⏳ Generating test sequences (seq_len={args.seq_length})...")
    X_test, y_test = generate_test_sequences(
        test_scaled, args.seq_length, rul_df, FEATURE_COLS
    )
    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_test shape: {y_test.shape}")
    
    # Load model
    if args.model_path is None:
        model_name = MODEL_NAMES[args.model]
        model_path = MODEL_DIR / model_name
    else:
        model_path = Path(args.model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"🏗️  Loading {args.model.upper()} model from: {model_path}")
    input_size = X_test.shape[2]
    model = get_model(args.model, input_size=input_size, seq_len=args.seq_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Evaluate model
    print("📊 Evaluating model...")
    y_pred, rmse, mae = evaluate_model(model, X_test, y_test, device)
    
    # Print results
    print("\n" + "=" * 50)
    print(f"📊 Evaluation Results ({args.model.upper()})")
    print("=" * 50)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print("=" * 50)
    
    print(f"\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

