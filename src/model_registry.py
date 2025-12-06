"""
Model Registry and Versioning System for Predictive Maintenance MLOps.

Provides functions to save and load model versions with metadata.
Structure: models/v1/, models/v2/, etc.
"""
import json
import joblib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from src.config import MODEL_DIR, FEATURE_COLS, SEQ_LENGTH


def get_latest_version() -> int:
    """Get the latest version number from existing model directories.
    
    Returns
    -------
    int
        Latest version number. Returns 0 if no versions exist.
    """
    if not MODEL_DIR.exists():
        return 0
    
    versions = []
    for item in MODEL_DIR.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            try:
                version_num = int(item.name[1:])  # Extract number after 'v'
                versions.append(version_num)
            except ValueError:
                continue
    
    return max(versions) if versions else 0


def get_next_version() -> int:
    """Get the next version number (latest + 1).
    
    Returns
    -------
    int
        Next version number.
    """
    return get_latest_version() + 1


def save_model_version(
    model: nn.Module,
    scaler: Any,  # StandardScaler from sklearn
    rmse: float,
    mae: float,
    model_type: str,
    sequence_length: int = SEQ_LENGTH,
    input_features: int = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    version: Optional[int] = None,
) -> Path:
    """Save a model version with scaler and metadata.
    
    Creates a new version directory (v1, v2, v3...) and saves:
    - model.pth: Trained model state dict
    - scaler.pkl: Fitted scaler
    - metadata.json: Model metadata (metrics, config, timestamp)
    
    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    scaler : StandardScaler
        Fitted scaler from sklearn.
    rmse : float
        Root Mean Squared Error on test set.
    mae : float
        Mean Absolute Error on test set.
    model_type : str
        Model type: 'lstm', 'bilstm', 'gru', or 'transformer'.
    sequence_length : int, optional
        Sequence length used for training, by default SEQ_LENGTH.
    input_features : int, optional
        Number of input features. If None, inferred from model.
    additional_metadata : dict, optional
        Additional metadata to include, by default None.
    version : int, optional
        Specific version number. If None, auto-increments, by default None.
        
    Returns
    -------
    Path
        Path to the version directory (e.g., models/v1/).
    """
    # Determine version
    if version is None:
        version = get_next_version()
    
    # Create version directory
    version_dir = MODEL_DIR / f"v{version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # Infer input_features from model if not provided
    if input_features is None:
        # Try to infer from model architecture
        if hasattr(model, "lstm"):
            if hasattr(model.lstm, "input_size"):
                input_features = model.lstm.input_size
        elif hasattr(model, "gru"):
            if hasattr(model.gru, "input_size"):
                input_features = model.gru.input_size
        elif hasattr(model, "input_proj"):
            # Transformer
            input_features = model.input_proj.in_features
        else:
            input_features = len(FEATURE_COLS)  # Fallback
    
    # Prepare metadata
    metadata = {
        "version": version,
        "model_type": model_type.lower(),
        "rmse": float(rmse),
        "mae": float(mae),
        "timestamp": datetime.now().isoformat(),
        "sequence_length": sequence_length,
        "input_features": input_features,
        "feature_columns": FEATURE_COLS,
        "model_architecture": {
            "type": model.__class__.__name__,
        },
    }
    
    # Add additional metadata if provided
    if additional_metadata:
        metadata.update(additional_metadata)
    
    # Save model
    model_path = version_dir / "model.pth"
    torch.save(model.state_dict(), model_path)
    
    # Save scaler
    scaler_path = version_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata_path = version_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Model version v{version} saved to: {version_dir}")
    print(f"   - Model: {model_path}")
    print(f"   - Scaler: {scaler_path}")
    print(f"   - Metadata: {metadata_path}")
    
    return version_dir


def load_model_version(
    version: int,
    model_class: nn.Module,
    device: Optional[torch.device] = None,
) -> tuple[nn.Module, Any, Dict[str, Any]]:
    """Load a model version with scaler and metadata.
    
    Parameters
    ----------
    version : int
        Version number to load (e.g., 1 for v1).
    model_class : nn.Module
        Model class to instantiate (e.g., RUL_LSTM).
    device : torch.device, optional
        Device to load model on. If None, uses CPU, by default None.
        
    Returns
    -------
    tuple[nn.Module, StandardScaler, dict]
        Loaded model, scaler, and metadata dictionary.
    """
    version_dir = MODEL_DIR / f"v{version}"
    
    if not version_dir.exists():
        raise FileNotFoundError(f"Version v{version} not found at {version_dir}")
    
    # Load metadata
    metadata_path = version_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found for version v{version}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Load scaler
    scaler_path = version_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found for version v{version}")
    
    scaler = joblib.load(scaler_path)
    
    # Load model
    model_path = version_dir / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found for version v{version}")
    
    # Get model parameters from metadata
    input_size = metadata.get("input_features", len(FEATURE_COLS))
    seq_len = metadata.get("sequence_length", SEQ_LENGTH)
    
    # Instantiate model
    if device is None:
        device = torch.device("cpu")
    
    # Create model instance (assuming model_class needs input_size)
    if metadata["model_type"] == "transformer":
        model = model_class(input_size=input_size, seq_len=seq_len)
    else:
        model = model_class(input_size=input_size)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded model version v{version} from: {version_dir}")
    print(f"   Model type: {metadata['model_type']}")
    print(f"   RMSE: {metadata['rmse']:.2f}, MAE: {metadata['mae']:.2f}")
    print(f"   Trained: {metadata['timestamp']}")
    
    return model, scaler, metadata


def list_model_versions() -> list[Dict[str, Any]]:
    """List all available model versions with their metadata.
    
    Returns
    -------
    list[dict]
        List of metadata dictionaries for each version.
    """
    versions = []
    
    if not MODEL_DIR.exists():
        return versions
    
    for item in sorted(MODEL_DIR.iterdir()):
        if item.is_dir() and item.name.startswith("v"):
            try:
                version_num = int(item.name[1:])
                metadata_path = item / "metadata.json"
                
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    versions.append(metadata)
            except (ValueError, json.JSONDecodeError):
                continue
    
    return sorted(versions, key=lambda x: x.get("version", 0))


def get_version_info(version: int) -> Dict[str, Any]:
    """Get metadata for a specific version.
    
    Parameters
    ----------
    version : int
        Version number.
        
    Returns
    -------
    dict
        Metadata dictionary.
    """
    version_dir = MODEL_DIR / f"v{version}"
    metadata_path = version_dir / "metadata.json"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Version v{version} not found")
    
    with open(metadata_path, "r") as f:
        return json.load(f)

