"""
Utility functions for the Predictive Maintenance MLOps project.
"""
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments.
    
    Parameters
    ----------
    seed : int, optional
        Random seed value, by default 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, else CPU).
    
    Returns
    -------
    torch.device
        Device to use for training/inference.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

