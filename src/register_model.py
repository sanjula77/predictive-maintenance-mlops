"""
Register/Save a trained model version.
Usage: python -m src.register_model --model-path models/rul_lstm.pth --model-type lstm --rmse 24.04 --mae 16.81
"""

import argparse
from pathlib import Path

import torch

from src.config import FEATURE_COLS, MODEL_DIR, SEQ_LENGTH
from src.data.preprocessing import load_scaler
from src.model_registry import get_next_version, save_model_version
from src.models.architectures import get_model
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Register a trained model as a new version")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model .pth file",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=None,
        help="Path to scaler .pkl file (default: models/scaler.pkl)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["lstm", "bilstm", "gru", "transformer"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--rmse",
        type=float,
        required=True,
        help="Test RMSE score",
    )
    parser.add_argument(
        "--mae",
        type=float,
        required=True,
        help="Test MAE score",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=SEQ_LENGTH,
        help=f"Sequence length used (default: {SEQ_LENGTH})",
    )
    parser.add_argument(
        "--input-features",
        type=int,
        default=None,
        help="Number of input features (auto-detected if not provided)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Specific version number (auto-increments if not provided)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (for metadata)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size used (for metadata)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate used (for metadata)",
    )

    args = parser.parse_args()

    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"📂 Loading model from: {model_path}")

    # Determine input features
    input_features = args.input_features
    if input_features is None:
        input_features = len(FEATURE_COLS)
        print(f"   Using default input_features: {input_features}")

    # Create model instance
    device = get_device()
    model = get_model(args.model_type, input_size=input_features, seq_len=args.seq_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Load scaler
    if args.scaler_path:
        scaler_path = Path(args.scaler_path)
    else:
        scaler_path = MODEL_DIR / "scaler.pkl"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    print(f"📂 Loading scaler from: {scaler_path}")
    scaler = load_scaler(scaler_path)

    # Prepare additional metadata
    additional_metadata = {}
    if args.epochs:
        additional_metadata["epochs"] = args.epochs
    if args.batch_size:
        additional_metadata["batch_size"] = args.batch_size
    if args.learning_rate:
        additional_metadata["learning_rate"] = args.learning_rate

    # Save model version
    print("\n💾 Registering model version...")
    version_dir = save_model_version(
        model=model,
        scaler=scaler,
        rmse=args.rmse,
        mae=args.mae,
        model_type=args.model_type,
        sequence_length=args.seq_length,
        input_features=input_features,
        additional_metadata=additional_metadata if additional_metadata else None,
        version=args.version,
    )

    print("\n✅ Model registered successfully!")
    print(f"   Version directory: {version_dir}")
    if args.version is None:
        print(f"   Next version will be: v{get_next_version()}")


if __name__ == "__main__":
    main()
