"""
Tests for utility functions.
"""

import pytest
import torch

# Try importing utils - might not exist yet
try:
    from src.utils import get_device, set_seed

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


@pytest.mark.skipif(not UTILS_AVAILABLE, reason="utils module not available")
class TestUtils:
    """Test utility functions."""

    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device is not None
        assert isinstance(device, torch.device)

    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        set_seed(42)
        # After setting seed, should get same random number
        torch.manual_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)
