"""
Pytest configuration and shared fixtures.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_data_path():
    """Return path to test data directory."""
    return Path(__file__).parent.parent / "data" / "raw"


@pytest.fixture
def model_dir():
    """Return path to models directory."""
    return Path(__file__).parent.parent / "models"
