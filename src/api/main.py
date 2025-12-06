"""
Production-ready FastAPI inference service for Predictive Maintenance RUL prediction.

This API provides endpoints to:
- Predict RUL from sensor data
- Load models dynamically by version
- List available models
- Health checks

Usage:
    uvicorn src.api.main:app --reload
"""
import logging
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import torch
import pandas as pd

from src.model_registry import load_model_version, get_version_info, list_model_versions
from src.models.architectures import RUL_LSTM, RUL_BiLSTM, RUL_GRU, RUL_Transformer
from src.data.preprocessing import scale_features
from src.config import FEATURE_COLS, SEQ_LENGTH
from src.utils import get_device
from src.api.schemas import (
    PredictionRequest,
    SimplePredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model and scaler cache
_model_cache = {}
_scaler_cache = {}
_metadata_cache = {}
_device = get_device()

# Model class mapping
MODEL_CLASSES = {
    "lstm": RUL_LSTM,
    "bilstm": RUL_BiLSTM,
    "gru": RUL_GRU,
    "transformer": RUL_Transformer,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    logger.info("Starting Predictive Maintenance API...")
    logger.info(f"Using device: {_device}")
    yield
    logger.info("Shutting down Predictive Maintenance API...")


app = FastAPI(
    title="Predictive Maintenance API",
    description="""
    Production-ready API for Remaining Useful Life (RUL) prediction.
    
    ## Features
    
    - Predict RUL from engine sensor data
    - Load models dynamically by version
    - Automatic preprocessing (scaling, reshaping)
    - Input validation with Pydantic
    - Model caching for performance
    
    ## Usage
    
    Send POST request to `/predict` with sensor readings from last 30 cycles.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model_if_needed(version: int, model_type: str):
    """
    Load model and scaler if not already cached.
    
    Args:
        version: Model version number
        model_type: Model type (lstm, bilstm, gru, transformer)
        
    Returns:
        Tuple of (model, scaler, metadata)
    """
    cache_key = f"{model_type.lower()}_v{version}"
    
    if cache_key not in _model_cache:
        logger.info(f"Loading model: {model_type} v{version}")
        start_time = time.time()
        
        model_class = MODEL_CLASSES.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}. Choose from: {list(MODEL_CLASSES.keys())}")
        
        try:
            model, scaler, metadata = load_model_version(
                version=version,
                model_class=model_class,
                device=_device
            )
            _model_cache[cache_key] = model
            _scaler_cache[cache_key] = scaler
            _metadata_cache[cache_key] = metadata
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
        except FileNotFoundError as e:
            logger.error(f"Model not found: {e}")
            raise HTTPException(
                status_code=404,
                detail=f"Model version {version} of type {model_type} not found"
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error loading model: {str(e)}"
            )
    
    return _model_cache[cache_key], _scaler_cache[cache_key], _metadata_cache[cache_key]


@app.get("/", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status of the API
    """
    return HealthResponse(
        status="healthy",
        message="Predictive Maintenance API is running",
        version="1.0.0"
    )


@app.get("/health")
def health():
    """Simple health check for load balancers."""
    return {"status": "ok"}


@app.get("/models", response_model=dict)
def list_available_models():
    """
    List all available model versions.
    
    Returns:
        Dictionary with list of available models and their metadata
    """
    try:
        versions = list_model_versions()
        logger.info(f"Listing {len(versions)} model versions")
        return {
            "count": len(versions),
            "models": versions
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{version}", response_model=ModelInfoResponse)
def get_model_info(version: int):
    """
    Get detailed information about a specific model version.
    
    Args:
        version: Model version number
        
    Returns:
        Model metadata including metrics and configuration
    """
    try:
        metadata = get_version_info(version)
        logger.info(f"Retrieved info for model version {version}")
        return ModelInfoResponse(**metadata)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model version {version} not found"
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict_rul(
    request: PredictionRequest,
    version: int = Query(1, description="Model version to use", ge=1),
    model_type: ModelType = Query(ModelType.LSTM, description="Model type to use")
):
    """
    Predict Remaining Useful Life (RUL) from engine sensor data.
    
    This endpoint:
    1. Validates input sensor readings
    2. Preprocesses data (scaling, reshaping)
    3. Runs model inference
    4. Returns predicted RUL
    
    Args:
        request: PredictionRequest with sensor readings
        version: Model version number (query parameter, default: 1)
        model_type: Model type - lstm, bilstm, gru, or transformer (query parameter)
        
    Returns:
        PredictionResponse with predicted RUL and model info
        
    Example:
        POST /predict?version=1&model_type=lstm
        {
            "sensor_readings": [
                {"op_setting_1": 0.0, "op_setting_2": 0.0, ..., "sensor_21": 23.4},
                ... (30 cycles)
            ]
        }
    """
    start_time = time.time()
    
    try:
        # Load model and scaler
        model, scaler, metadata = load_model_if_needed(version, model_type.value)
        
        # Get sensor readings (handle both field names)
        try:
            readings = request.get_readings()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Convert sensor readings to numpy array
        sensor_data = np.array([reading.to_list() for reading in readings])
        
        logger.info(
            f"Received prediction request: {len(readings)} cycles, "
            f"model v{version} ({model_type.value})"
        )
        
        # Validate input shape
        if len(sensor_data.shape) != 2:
            raise HTTPException(
                status_code=400,
                detail="Invalid input format: sensor_readings must be a 2D array"
            )
        
        if sensor_data.shape[1] != len(FEATURE_COLS):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(FEATURE_COLS)} features per reading, got {sensor_data.shape[1]}"
            )
        
        # Handle sequence length
        n_cycles = sensor_data.shape[0]
        if n_cycles < SEQ_LENGTH:
            # Pad with zeros at the beginning
            pad = np.zeros((SEQ_LENGTH - n_cycles, len(FEATURE_COLS)))
            sensor_data = np.vstack([pad, sensor_data])
            logger.info(f"Padded input from {n_cycles} to {SEQ_LENGTH} cycles")
        elif n_cycles > SEQ_LENGTH:
            # Take last SEQ_LENGTH cycles
            sensor_data = sensor_data[-SEQ_LENGTH:]
            logger.info(f"Truncated input from {n_cycles} to {SEQ_LENGTH} cycles")
        
        # Scale features
        df = pd.DataFrame(sensor_data, columns=FEATURE_COLS)
        df_scaled = scale_features(df, scaler)
        sensor_data_scaled = df_scaled[FEATURE_COLS].values
        
        # Reshape for model: (batch_size=1, seq_len, n_features)
        sensor_tensor = torch.tensor(
            sensor_data_scaled.reshape(1, SEQ_LENGTH, len(FEATURE_COLS)),
            dtype=torch.float32
        ).to(_device)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            prediction = model(sensor_tensor).cpu().numpy()[0, 0]
        
        # Ensure non-negative RUL
        predicted_rul = max(0.0, float(prediction))
        
        inference_time = time.time() - start_time
        logger.info(f"Prediction completed in {inference_time:.3f}s: RUL={predicted_rul:.2f}")
        
        return PredictionResponse(
            predicted_rul=predicted_rul,
            model_version=version,
            model_type=model_type.value,
            confidence_interval=None  # Can be added later with ensemble models
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during prediction: {str(e)}"
        )


@app.post("/predict/simple", response_model=PredictionResponse)
def predict_rul_simple(
    request: SimplePredictionRequest,
    version: int = Query(1, description="Model version to use", ge=1),
    model_type: ModelType = Query(ModelType.LSTM, description="Model type to use")
):
    """
    Simplified prediction endpoint that accepts raw arrays.
    
    This endpoint accepts a 2D array directly without requiring named fields.
    Useful for programmatic access or when you have data in array format.
    
    Args:
        request: SimplePredictionRequest with sensor_data as 2D array
        version: Model version number (query parameter, default: 1)
        model_type: Model type (query parameter)
        
    Returns:
        PredictionResponse with predicted RUL
        
    Example:
        POST /predict/simple?version=1&model_type=lstm
        {
            "sensor_data": [
                [0.0, 0.0, 100.0, 518.67, 641.82, ...],  // 24 features
                [0.0, 0.0, 100.0, 520.1, 642.5, ...],    // 24 features
                ... // 30 cycles recommended
            ]
        }
    """
    start_time = time.time()
    
    try:
        # Load model and scaler
        model, scaler, metadata = load_model_if_needed(version, model_type.value)
        
        # Convert to numpy array
        sensor_data = np.array(request.sensor_data)
        
        logger.info(
            f"Received simple prediction request: {len(request.sensor_data)} cycles, "
            f"model v{version} ({model_type.value})"
        )
        
        # Validate input shape
        if len(sensor_data.shape) != 2:
            raise HTTPException(
                status_code=400,
                detail="Invalid input format: sensor_data must be a 2D array"
            )
        
        if sensor_data.shape[1] != len(FEATURE_COLS):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(FEATURE_COLS)} features per reading, got {sensor_data.shape[1]}"
            )
        
        # Handle sequence length
        n_cycles = sensor_data.shape[0]
        if n_cycles < SEQ_LENGTH:
            pad = np.zeros((SEQ_LENGTH - n_cycles, len(FEATURE_COLS)))
            sensor_data = np.vstack([pad, sensor_data])
            logger.info(f"Padded input from {n_cycles} to {SEQ_LENGTH} cycles")
        elif n_cycles > SEQ_LENGTH:
            sensor_data = sensor_data[-SEQ_LENGTH:]
            logger.info(f"Truncated input from {n_cycles} to {SEQ_LENGTH} cycles")
        
        # Scale features
        df = pd.DataFrame(sensor_data, columns=FEATURE_COLS)
        df_scaled = scale_features(df, scaler)
        sensor_data_scaled = df_scaled[FEATURE_COLS].values
        
        # Reshape for model
        sensor_tensor = torch.tensor(
            sensor_data_scaled.reshape(1, SEQ_LENGTH, len(FEATURE_COLS)),
            dtype=torch.float32
        ).to(_device)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            prediction = model(sensor_tensor).cpu().numpy()[0, 0]
        
        predicted_rul = max(0.0, float(prediction))
        
        inference_time = time.time() - start_time
        logger.info(f"Prediction completed in {inference_time:.3f}s: RUL={predicted_rul:.2f}")
        
        return PredictionResponse(
            predicted_rul=predicted_rul,
            model_version=version,
            model_type=model_type.value,
            confidence_interval=None
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during prediction: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
