# API Documentation

Complete API reference for the Predictive Maintenance RUL Prediction API.

## Base URL

- **Local**: `http://localhost:8000`
- **Production**: `https://your-api-url.com`

## Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Endpoints

### Health Check

```http
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Predictive Maintenance API is running",
  "version": "1.0.0"
}
```

### Simple Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "ok"
}
```

### List Available Models

```http
GET /models
```

**Response:**
```json
{
  "count": 3,
  "models": [
    {
      "version": 1,
      "model_type": "lstm",
      "rmse": 24.04,
      "mae": 16.81,
      "timestamp": "2024-01-15T10:30:45"
    },
    ...
  ]
}
```

### Get Model Info

```http
GET /models/{version}
```

**Parameters:**
- `version` (path): Model version number

**Response:**
```json
{
  "version": 1,
  "model_type": "lstm",
  "rmse": 24.04,
  "mae": 16.81,
  "timestamp": "2024-01-15T10:30:45",
  "sequence_length": 30,
  "input_features": 24
}
```

### Predict RUL

```http
POST /predict?version=1&model_type=lstm
```

**Query Parameters:**
- `version` (optional, default: 1): Model version to use
- `model_type` (optional, default: lstm): Model type (lstm, bilstm, gru, transformer)

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
      "sensor_10": 1.3,
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
  "model_version": 1,
  "model_type": "lstm",
  "confidence_interval": null
}
```

## Example Requests

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Predict RUL
curl -X POST "http://localhost:8000/predict?version=1&model_type=lstm" \
  -H "Content-Type: application/json" \
  -d '{
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
        "sensor_10": 1.3,
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
    ]
  }'
```

### Using Python

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict RUL
data = {
    "sensor_readings": [
        {
            "op_setting_1": 0.0,
            "op_setting_2": 0.0,
            "op_setting_3": 100.0,
            "sensor_1": 518.67,
            # ... all 21 sensors
        }
    ] * 30  # 30 cycles
}

response = requests.post(
    "http://localhost:8000/predict?version=1&model_type=lstm",
    json=data
)
result = response.json()
print(f"Predicted RUL: {result['predicted_rul']}")
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Expected 24 features per reading, got 23"
}
```

### 404 Not Found
```json
{
  "detail": "Model version 999 not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error during prediction: ..."
}
```

## Notes

- **Sequence Length**: The API uses the last 30 cycles (SEQ_LENGTH). If you provide fewer, it will pad with zeros. If more, it will take the last 30.
- **Model Caching**: Models are cached in memory after first load for faster subsequent requests.
- **Input Validation**: All inputs are validated using Pydantic schemas.
- **Preprocessing**: Automatic scaling and reshaping is handled by the API.
