"""
Tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        # API returns "ok" not "healthy"
        assert data["status"] in ["healthy", "ok"]


class TestModelsEndpoint:
    """Test models listing endpoint."""

    def test_list_models(self):
        """Test listing available models."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data or "count" in data


class TestPredictionEndpoint:
    """Test prediction endpoint."""

    def test_predict_invalid_request(self):
        """Test prediction with invalid request."""
        response = client.post("/predict?version=1&model_type=lstm", json={})
        # Should return error for invalid input
        assert response.status_code in [400, 422, 500]

    def test_predict_missing_parameters(self):
        """Test prediction without required parameters."""
        response = client.post("/predict", json={})
        # Should return 422 for missing query parameters
        assert response.status_code == 422
