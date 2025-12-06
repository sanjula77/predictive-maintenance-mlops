"""
Prediction script for RUL inference on new data.
Usage: python -m src.predict --model lstm --input data.csv
"""
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.config import (
    SEED,
    SEQ_LENGTH,
    MODEL_DIR,
    MODEL_NAMES,
    FEATURE_COLS,
    COLUMN_NAMES,
)
from src.data.preprocessing import load_scaler, scale_features, generate_test_sequences
from src.models.architectures import get_model
from src.utils import set_seed, get_device


def predict_rul(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Predict RUL for given sequences.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    X : np.ndarray
        Input sequences of shape (n_samples, seq_len, n_features).
    device : torch.device
        CPU or GPU device.
        
    Returns
    -------
    np.ndarray
        Predicted RUL values of shape (n_samples,).
    """
    model.eval()
    model = model.to(device)
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_pred = model(X_tensor).cpu().numpy().flatten()
    
    return y_pred


def main():
    parser = argparse.ArgumentParser(description="Predict RUL for new data")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "bilstm", "gru", "transformer"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file (must have same columns as training data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions CSV (default: predictions.csv)",
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
    
    # Load input data
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"📂 Loading input data from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Validate columns
    required_cols = set(COLUMN_NAMES) - {"RUL"}  # RUL not required for prediction
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    print(f"   Input dataset shape: {df.shape}")
    
    # Load scaler
    print("🔧 Loading scaler...")
    scaler = load_scaler()
    
    # Scale features
    print("📏 Scaling features...")
    df_scaled = scale_features(df, scaler)
    
    # Generate sequences (for each engine, use last seq_len cycles)
    print(f"⏳ Generating sequences (seq_len={args.seq_length})...")
    
    # For prediction, we need to generate sequences similar to test set
    # (use last seq_len cycles per engine)
    X_pred = []
    engine_ids = sorted(df["engine_id"].unique())
    
    for engine_id in engine_ids:
        engine_data = df_scaled[df_scaled["engine_id"] == engine_id].sort_values("cycle")
        engine_array = engine_data[FEATURE_COLS].values
        
        if len(engine_array) >= args.seq_length:
            X_pred.append(engine_array[-args.seq_length:])
        else:  # pad if not enough cycles
            pad = np.zeros((args.seq_length - len(engine_array), len(FEATURE_COLS)))
            X_pred.append(np.vstack([pad, engine_array]))
    
    X_pred = np.array(X_pred)
    print(f"   X_pred shape: {X_pred.shape}")
    
    # Load model
    if args.model_path is None:
        model_name = MODEL_NAMES[args.model]
        model_path = MODEL_DIR / model_name
    else:
        model_path = Path(args.model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"🏗️  Loading {args.model.upper()} model from: {model_path}")
    input_size = X_pred.shape[2]
    model = get_model(args.model, input_size=input_size, seq_len=args.seq_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Predict
    print("🔮 Predicting RUL...")
    predictions = predict_rul(model, X_pred, device)
    
    # Create output DataFrame
    engine_ids = sorted(df["engine_id"].unique())
    results_df = pd.DataFrame({
        "engine_id": engine_ids,
        "predicted_rul": predictions,
    })
    
    # Save predictions
    if args.output is None:
        output_path = Path("predictions.csv")
    else:
        output_path = Path(args.output)
    
    results_df.to_csv(output_path, index=False)
    print(f"💾 Predictions saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Prediction Summary")
    print("=" * 50)
    print(f"Number of engines: {len(results_df)}")
    print(f"Mean predicted RUL: {predictions.mean():.2f}")
    print(f"Min predicted RUL:  {predictions.min():.2f}")
    print(f"Max predicted RUL:  {predictions.max():.2f}")
    print("=" * 50)
    
    print(f"\n✅ Prediction complete!")


if __name__ == "__main__":
    main()

