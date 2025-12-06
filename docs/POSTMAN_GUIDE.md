# Postman API Testing Guide

Complete guide for testing the Predictive Maintenance API with Postman.

## 🚀 Setup

1. **Start the API server**
   ```bash
   uvicorn src.api.main:app --reload
   ```

2. **Base URL**: `http://localhost:8000`

## 📋 Endpoints

### 1. Health Check

**Request:**
- Method: `GET`
- URL: `http://localhost:8000/health`

**Response:**
```json
{
    "status": "ok"
}
```

### 2. List Models

**Request:**
- Method: `GET`
- URL: `http://localhost:8000/models`

### 3. Predict RUL (Structured Format)

**Request:**
- Method: `POST`
- URL: `http://localhost:8000/predict?version=1&model_type=lstm`
- Headers:
  - `Content-Type: application/json`
- Body (raw JSON):

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
    ]
}
```

**Note:** You can use `sensor_data` instead of `sensor_readings` - both work!

**Alternative (using sensor_data):**
```json
{
    "sensor_data": [
        {
            "op_setting_1": 0.0,
            "op_setting_2": 0.0,
            "op_setting_3": 100.0,
            "sensor_1": 518.67,
            "sensor_2": 641.82,
            ...
            "sensor_21": 23.4190
        }
    ]
}
```

### 4. Predict RUL (Simple Array Format)

**Request:**
- Method: `POST`
- URL: `http://localhost:8000/predict/simple?version=1&model_type=lstm`
- Headers:
  - `Content-Type: application/json`
- Body (raw JSON):

```json
{
    "sensor_data": [
        [0.0, 0.0, 100.0, 518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36, 2388.06, 9046.19, 1.3, 47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100.0, 39.06, 23.4190]
    ]
}
```

**Feature Order:** 
`op_setting_1, op_setting_2, op_setting_3, sensor_1, sensor_2, ..., sensor_21` (24 features total)

## ✅ Common Issues & Solutions

### Issue 1: "Field required: sensor_readings"

**Problem:** You're using `sensor_data` but the API expects `sensor_readings`

**Solution:** 
- Use `/predict/simple` endpoint for array format, OR
- Rename `sensor_data` to `sensor_readings`, OR
- The API now accepts both field names!

### Issue 2: Missing operating settings

**Problem:** You only sent sensors, missing `op_setting_1`, `op_setting_2`, `op_setting_3`

**Solution:** 
- Operating settings now have defaults (0.0, 0.0, 100.0)
- You can omit them, but it's better to include them

### Issue 3: Wrong number of features

**Problem:** You have 24 sensors but API expects 24 features (3 op_settings + 21 sensors)

**Solution:**
- Make sure you have exactly 24 features
- Order: `op_setting_1, op_setting_2, op_setting_3, sensor_1, ..., sensor_21`
- Remove any extra sensors (sensor_22, sensor_23, etc.)

### Issue 4: Only one reading sent

**Problem:** API works with one reading, but for best results use 30 cycles

**Solution:**
- Send multiple readings (recommended: 30 cycles)
- API will pad with zeros if fewer, or take last 30 if more

## 📝 Postman Collection Example

### Request 1: Health Check
```
GET http://localhost:8000/health
```

### Request 2: Predict (Structured)
```
POST http://localhost:8000/predict?version=1&model_type=lstm
Content-Type: application/json

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
    ]
}
```

### Request 3: Predict (Simple Array)
```
POST http://localhost:8000/predict/simple?version=1&model_type=lstm
Content-Type: application/json

{
    "sensor_data": [
        [0.0, 0.0, 100.0, 518.67, 641.82, 1589.70, 1400.60, 14.62, 21.61, 554.36, 2388.06, 9046.19, 1.3, 47.47, 521.66, 2388.02, 8138.62, 8.4195, 0.03, 392, 2388, 100.0, 39.06, 23.4190]
    ]
}
```

## 🎯 Quick Fix for Your Error

Based on your error, here's the corrected request:

**Use this format:**
```json
{
    "sensor_readings": [
        {
            "op_setting_1": 0.0,
            "op_setting_2": 0.0,
            "op_setting_3": 100.0,
            "sensor_1": 10.5,
            "sensor_2": 3.41,
            "sensor_3": 500.1,
            "sensor_4": 0.88,
            "sensor_5": 30.1,
            "sensor_6": 12.4,
            "sensor_7": 0.02,
            "sensor_8": 90.2,
            "sensor_9": 0.12,
            "sensor_10": 300.1,
            "sensor_11": 0.22,
            "sensor_12": 14.2,
            "sensor_13": 0.55,
            "sensor_14": 20.1,
            "sensor_15": 0.32,
            "sensor_16": 0.18,
            "sensor_17": 400.5,
            "sensor_18": 1.23,
            "sensor_19": 0.01,
            "sensor_20": 600.3,
            "sensor_21": 0.4
        }
    ]
}
```

**Changes made:**
1. ✅ Changed `sensor_data` → `sensor_readings`
2. ✅ Added `op_setting_1`, `op_setting_2`, `op_setting_3`
3. ✅ Removed `sensor_22`, `sensor_23`, `sensor_24` (don't exist)
4. ✅ Wrapped in array (even if just one reading)

Now it should work! 🎉

