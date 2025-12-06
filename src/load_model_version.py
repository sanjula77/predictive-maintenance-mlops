"""
Load a specific model version for inference.
Usage: python -m src.load_model_version --version 1 --model-type lstm
"""
import argparse
import torch
import numpy as np
from pathlib import Path

from src.model_registry import load_model_version, get_version_info
from src.models.architectures import RUL_LSTM, RUL_BiLSTM, RUL_GRU, RUL_Transformer
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Load a model version")
    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Version number to load (e.g., 1 for v1)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["lstm", "bilstm", "gru", "transformer"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save loaded model files (optional)",
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only show model info, don't load model",
    )
    
    args = parser.parse_args()
    
    # Map model type to class
    model_classes = {
        "lstm": RUL_LSTM,
        "bilstm": RUL_BiLSTM,
        "gru": RUL_GRU,
        "transformer": RUL_Transformer,
    }
    
    model_class = model_classes[args.model_type]
    
    # Get version info
    try:
        metadata = get_version_info(args.version)
        print(f"\n📋 Model Version v{args.version} Information:")
        print("=" * 70)
        print(f"Model Type:      {metadata.get('model_type', 'N/A')}")
        print(f"RMSE:            {metadata.get('rmse', 0):.2f}")
        print(f"MAE:             {metadata.get('mae', 0):.2f}")
        print(f"Sequence Length: {metadata.get('sequence_length', 'N/A')}")
        print(f"Input Features:  {metadata.get('input_features', 'N/A')}")
        print(f"Timestamp:       {metadata.get('timestamp', 'N/A')}")
        print("=" * 70)
        
        if args.info_only:
            return
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Load model
    print(f"\n📂 Loading model version v{args.version}...")
    device = get_device()
    
    try:
        model, scaler, metadata = load_model_version(
            version=args.version,
            model_class=model_class,
            device=device
        )
        
        print(f"\n✅ Model loaded successfully!")
        print(f"   Model is on device: {device}")
        print(f"   Model is in eval mode: {not model.training}")
        
        # Optionally save to output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = output_dir / "model.pth"
            torch.save(model.state_dict(), model_path)
            
            # Save scaler
            import joblib
            scaler_path = output_dir / "scaler.pkl"
            joblib.dump(scaler, scaler_path)
            
            print(f"\n💾 Model files saved to: {output_dir}")
            print(f"   - Model: {model_path}")
            print(f"   - Scaler: {scaler_path}")
        
        print(f"\n💡 Model is ready for inference!")
        print(f"   Use model.eval() and model(input_tensor) for predictions")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


if __name__ == "__main__":
    main()

