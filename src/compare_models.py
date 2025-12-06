"""
Compare all trained models on the test set.
Usage: python -m src.compare_models
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
    """Evaluate a model on test data."""
    model.eval()
    model = model.to(device)
    
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return y_pred, rmse, mae


def main():
    parser = argparse.ArgumentParser(description="Compare all trained models")
    parser.add_argument(
        "--seq-length",
        type=int,
        default=SEQ_LENGTH,
        help=f"Sequence length (default: {SEQ_LENGTH})",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["lstm", "bilstm", "gru", "transformer"],
        choices=["lstm", "bilstm", "gru", "transformer"],
        help="Models to compare (default: all)",
    )
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Get device
    device = get_device()
    print(f"🔥 Using device: {device}\n")
    
    # Load test data
    print("📂 Loading test data...")
    test_df = load_test_data()
    rul_df = load_rul_data()
    
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
    print(f"   y_test shape: {y_test.shape}\n")
    
    # Evaluate each model
    results = {}
    input_size = X_test.shape[2]
    
    print("=" * 60)
    print("📊 Model Comparison Results")
    print("=" * 60)
    
    for model_name in args.models:
        model_path = MODEL_DIR / MODEL_NAMES[model_name]
        
        if not model_path.exists():
            print(f"⚠️  {model_name.upper()}: Model not found at {model_path}")
            continue
        
        print(f"\n🏗️  Loading {model_name.upper()} model...")
        model = get_model(model_name, input_size=input_size, seq_len=args.seq_length)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        print(f"📊 Evaluating {model_name.upper()}...")
        y_pred, rmse, mae = evaluate_model(model, X_test, y_test, device)
        
        results[model_name] = {
            "rmse": rmse,
            "mae": mae,
            "predictions": y_pred,
        }
        
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAE:  {mae:.2f}")
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("📊 Final Comparison Table")
    print("=" * 60)
    print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10}")
    print("-" * 60)
    
    for model_name in args.models:
        if model_name in results:
            print(
                f"{model_name.upper():<15} "
                f"{results[model_name]['rmse']:<10.2f} "
                f"{results[model_name]['mae']:<10.2f}"
            )
    
    print("=" * 60)
    
    # Find best model
    if results:
        best_rmse = min(results.items(), key=lambda x: x[1]["rmse"])
        best_mae = min(results.items(), key=lambda x: x[1]["mae"])
        
        print(f"\n🏆 Best RMSE: {best_rmse[0].upper()} ({best_rmse[1]['rmse']:.2f})")
        print(f"🏆 Best MAE:  {best_mae[0].upper()} ({best_mae[1]['mae']:.2f})")
    
    print(f"\n✅ Comparison complete!")


if __name__ == "__main__":
    main()

