"""
Simple test script for the API.
Usage: python test_api.py
"""
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint."""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")


def test_list_models():
    """Test list models endpoint."""
    print("Testing list models...")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Found {data.get('count', 0)} models\n")


def test_predict():
    """Test prediction endpoint."""
    print("Testing prediction...")
    
    # Load example request
    example_file = Path("example_request.json")
    if not example_file.exists():
        print("❌ example_request.json not found")
        return
    
    with open(example_file) as f:
        data = json.load(f)
    
    # Make request
    response = requests.post(
        f"{BASE_URL}/predict?version=1&model_type=lstm",
        json=data
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Predicted RUL: {result['predicted_rul']:.2f}")
        print(f"   Model: {result['model_type']} v{result['model_version']}\n")
    else:
        print(f"❌ Error: {response.json()}\n")


if __name__ == "__main__":
    print("=" * 50)
    print("API Test Suite")
    print("=" * 50)
    print(f"Testing API at: {BASE_URL}\n")
    
    try:
        test_health()
        test_list_models()
        test_predict()
        print("✅ All tests completed!")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API.")
        print("   Make sure the API is running: uvicorn src.api.main:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")

