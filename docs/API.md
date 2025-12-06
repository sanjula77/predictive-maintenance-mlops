# API Documentation

Complete API reference for the Predictive Maintenance RUL Prediction API.

## Table of Contents

- [Base URL](#base-url)
- [Interactive Documentation](#interactive-documentation)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [Request Examples](#request-examples)
- [Error Handling](#error-handling)
- [MLflow Integration](#mlflow-integration)
- [Testing with Postman](#testing-with-postman)

## Base URL

- **Local**: `http://localhost:8000`
- **Docker**: `http://localhost:8000`
- **Production**: `https://your-api-url.com`

## Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs` - Interactive API explorer
- **ReDoc**: `http://localhost:8000/redoc` - Alternative documentation view

## Authentication

Currently, the API does not require authentication. For production deployments, consider adding API keys or OAuth2.

## Endpoints

### Health Check

```http
GET /
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Predictive Maintenance API is running",
  "version": "1.0.0"
}
```

**Use Cases:**
- Load balancer health checks
- Monitoring systems
- Deployment verification

### List Available Models

```http
GET /models
```

**Response:**
```json
{
  "count": 4,
  "source": "mlflow",
  "models": [
    {
      "version": 3,
      "model_type": "gru",
      "rmse": 22.49,
      "mae": 16.79,
      "stage": "Production",
      "aliases": ["production"],
      "timestamp": 1765028250020,
      "sequence_length": 30,
      "input_features": 24,
      "epochs": 20,
      "batch_size": 64,
      "learning_rate": 0.001,
      "source": "mlflow_registry"
    }
  ]
}
```

**Notes:**
- When MLflow is enabled, shows all registered models from MLflow
- When MLflow is disabled, shows legacy registry models
- `source` field indicates which registry is being used

### Get Model Information

```http
GET /models/{version}
```

**Parameters:**
- `version` (path, required): Model version number

**Response:**
```json
{
  "version": 1,
  "model_type": "lstm",
  "rmse": 24.04,
  "mae": 16.81,
  "timestamp": "2024-01-15T10:30:45",
  "sequence_length": 30,
  "input_features": 24,
  "epochs": 20,
  "batch_size": 64,
  "learning_rate": 0.001
}
```

**Error Responses:**
- `404`: Model version not found

### Predict RUL (Full Format)

```http
POST /predict?version=1&model_type=lstm
```

**Query Parameters:**
- `version` (optional, default: 1): Model version (ignored if MLflow enabled)
- `model_type` (optional, default: lstm): Model type (ignored if MLflow enabled)

**Request Body:**
```json
{
  "sensor_readings": [
    {
      "op_setting_1": 0.0,
      "op_setting_2": 0.0,
      "op_setting_3": 100.0,
      "sensor_1": 518.67,
      "sensor_2": 641.82,
      "sensor_3": 1589.70,
      "sensor_4": 1400.60,
      "sensor_5": 14.62,
      "sensor_6": 21.61,
      "sensor_7": 554.36,
      "sensor_8": 2388.06,
      "sensor_9": 9046.19,
      "sensor_10": 1.30,
      "sensor_11": 47.47,
      "sensor_12": 521.66,
      "sensor_13": 2388.02,
      "sensor_14": 8138.62,
      "sensor_15": 8.4195,
      "sensor_16": 0.03,
      "sensor_17": 392,
      "sensor_18": 2388,
      "sensor_19": 100.0,
      "sensor_20": 39.06,
      "sensor_21": 23.4190
    }
    // ... repeat for 30 cycles (recommended)
  ]
}
```

**Response:**
```json
{
  "predicted_rul": 125.5,
  "model_version": 3,
  "model_type": "gru",
  "confidence_interval": null
}
```

### Predict RUL (Simple Format)

```http
POST /predict/simple?version=1&model_type=lstm
```

**Request Body:**
```json
{
  "sensor_data": [
    [0.0, 0.0, 100.0, 518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36, 2388.06, 9046.19, 1.30, 47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100.0, 39.06, 23.4190],
    [0.0, 0.0, 100.0, 518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36, 2388.06, 9046.19, 1.30, 47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100.0, 39.06, 23.4190]
    // ... 30 cycles total (24 features each)
  ]
}
```

**Notes:**
- Simpler format for programmatic access
- Each inner array must have exactly 24 features
- API automatically pads/truncates to 30 cycles

## Request Examples

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Predict RUL (simple format)
curl -X POST "http://localhost:8000/predict/simple" \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

### Using Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# List models
models = requests.get("http://localhost:8000/models").json()
print(f"Available models: {models['count']}")

# Predict RUL
data = {
    "sensor_data": [
        [0.0, 0.0, 100.0, 518.67, 641.82, ...] * 24,  # 24 features
    ] * 30  # 30 cycles
}

response = requests.post(
    "http://localhost:8000/predict/simple",
    json=data
)
result = response.json()
print(f"Predicted RUL: {result['predicted_rul']}")
print(f"Model: {result['model_type']} v{result['model_version']}")
```

### Using JavaScript/Node.js

```javascript
const axios = require('axios');

// Health check
const health = await axios.get('http://localhost:8000/health');
console.log(health.data);

// Predict RUL
const prediction = await axios.post(
  'http://localhost:8000/predict/simple',
  {
    sensor_data: Array(30).fill(
      Array(24).fill(0).map((_, i) => Math.random() * 100)
    )
  }
);
console.log(`Predicted RUL: ${prediction.data.predicted_rul}`);
```

## Testing with Postman

### Import Collection

1. Open Postman
2. Click **Import**
3. Select `postman_collection.json` from project root
4. Collection will be imported with all endpoints pre-configured

### Manual Setup

**Create New Request:**

1. **Health Check**
   - Method: `GET`
   - URL: `http://localhost:8000/health`

2. **List Models**
   - Method: `GET`
   - URL: `http://localhost:8000/models`

3. **Predict RUL**
   - Method: `POST`
   - URL: `http://localhost:8000/predict/simple`
   - Headers: `Content-Type: application/json`
   - Body (raw JSON):
   ```json
   {
     "sensor_data": [
       [0.0, 0.0, 100.0, 518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36, 2388.06, 9046.19, 1.30, 47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100.0, 39.06, 23.4190],
       [0.0, 0.0, 100.0, 518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36, 2388.06, 9046.19, 1.30, 47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100.0, 39.06, 23.4190]
       // Add 28 more cycles (30 total)
     ]
   }
   ```

### Environment Variables

Create a Postman environment:

```json
{
  "base_url": "http://localhost:8000",
  "api_version": "1.0.0"
}
```

Then use: `{{base_url}}/health`

## MLflow Integration

### Automatic Model Selection

When `USE_MLFLOW=true`:
- API automatically loads the **Production** model from MLflow
- Query parameters (`version`, `model_type`) are ignored
- Response shows actual model version and type from MLflow

### Response Format with MLflow

```json
{
  "predicted_rul": 125.5,
  "model_version": 3,        // Actual MLflow version
  "model_type": "gru",       // Actual model type
  "confidence_interval": null
}
```

### Switching Models

To change the Production model:
1. Use MLflow UI: Models → Select version → Add alias "production"
2. Or use code: `promote_model_to_production("RUL-Prediction-Model", version=3)`
3. Restart API or wait for next request (model is cached)

## Error Handling

### 400 Bad Request

**Invalid Input Format:**
```json
{
  "detail": "Expected 24 features per reading, got 23"
}
```

**Missing Required Fields:**
```json
{
  "detail": "sensor_readings field is required"
}
```

### 404 Not Found

**Model Not Found:**
```json
{
  "detail": "Model version 999 not found"
}
```

### 500 Internal Server Error

**Server Error:**
```json
{
  "detail": "Internal server error during prediction: ..."
}
```

**Common Causes:**
- Model file corrupted
- MLflow connection failed
- Memory issues
- CUDA/GPU errors

## Best Practices

1. **Use Simple Format**: `/predict/simple` is easier for programmatic access
2. **30 Cycles Recommended**: Provide 30 cycles for best accuracy
3. **Error Handling**: Always check response status codes
4. **Model Caching**: First request may be slower (model loading)
5. **Health Checks**: Monitor `/health` endpoint for availability
6. **MLflow Mode**: Use MLflow in production for automatic model management

## Rate Limiting

Currently, no rate limiting is implemented. For production:
- Consider adding rate limiting middleware
- Use API gateway for throttling
- Implement request queuing for high load

## Performance

- **Model Loading**: ~2-5 seconds (first request)
- **Prediction**: ~50-200ms (after model loaded)
- **Concurrent Requests**: Supported (FastAPI async)
- **Memory Usage**: ~500MB-2GB (depending on model size)

## Notes

- **Sequence Length**: API uses last 30 cycles (SEQ_LENGTH). Fewer cycles are padded, more are truncated.
- **Model Caching**: Models cached in memory after first load
- **Input Validation**: All inputs validated using Pydantic schemas
- **Preprocessing**: Automatic scaling and reshaping handled by API
- **MLflow Models**: When enabled, uses Production model automatically
