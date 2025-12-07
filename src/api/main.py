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
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    ModelType,
    PredictionRequest,
    PredictionResponse,
    SimplePredictionRequest,
)
from src.config import FEATURE_COLS, SEQ_LENGTH
from src.data.preprocessing import scale_features
from src.model_registry import get_version_info, list_model_versions, load_model_version
from src.models.architectures import RUL_GRU, RUL_LSTM, RUL_BiLSTM, RUL_Transformer
from src.utils import get_device

# MLflow support (optional)
try:
    from src.mlflow_utils import get_production_model, setup_mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install mlflow to use MLflow model registry.")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model and scaler cache
_model_cache = {}
_scaler_cache = {}
_metadata_cache = {}
_device = get_device()

# MLflow model cache
_mlflow_model = None
_mlflow_scaler = None
_use_mlflow = False

# Initialize MLflow if available and enabled
if MLFLOW_AVAILABLE:
    import os
    if os.getenv("USE_MLFLOW", "false").lower() == "true":
        setup_mlflow()
        _use_mlflow = True
        logger.info("MLflow enabled - API will use Production model from MLflow registry")

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
    Uses MLflow Production model if MLflow is enabled, otherwise uses legacy registry.

    Args:
        version: Model version number (ignored if MLflow enabled)
        model_type: Model type (lstm, bilstm, gru, transformer) (ignored if MLflow enabled)

    Returns:
        Tuple of (model, scaler, metadata)
    """
    global _use_mlflow, _mlflow_model, _mlflow_scaler
    
    # Use MLflow Production model if enabled
    if _use_mlflow and MLFLOW_AVAILABLE:
        if _mlflow_model is None:
            try:
                _mlflow_model = get_production_model()
                # Note: MLflow models should include scaler in artifacts
                # For now, we'll still use the legacy scaler loading
                from src.data.preprocessing import load_scaler
                _mlflow_scaler = load_scaler()
                logger.info("Loaded Production model from MLflow registry")
            except Exception as e:
                logger.error(f"Failed to load MLflow model: {e}")
                logger.info("Falling back to legacy model registry")
                _use_mlflow = False

        if _mlflow_model is not None:
            # Get actual model info from MLflow Production model (using alias)
            try:
                from mlflow.tracking import MlflowClient
                from src.mlflow_utils import MODEL_REGISTRY_NAME
                
                client = MlflowClient()
                # Get Production model version using alias
                try:
                    # Get model info to find production alias
                    model_info = client.get_registered_model(name=MODEL_REGISTRY_NAME)
                    aliases_dict = getattr(model_info, 'aliases', {}) or {}
                    
                    # Find version with production alias
                    prod_version_str = aliases_dict.get("production")
                    if prod_version_str:
                        # Get the version object
                        versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
                        prod_version = None
                        for v in versions:
                            if str(v.version) == str(prod_version_str):
                                prod_version = v
                                break
                        
                        if prod_version:
                            run_id = prod_version.run_id
                            run = client.get_run(run_id)
                            params = run.data.params
                            actual_model_type = params.get("model_type", "unknown")
                            version_num = int(prod_version.version)
                            
                            metadata = {
                                "version": version_num,
                                "model_type": actual_model_type,
                                "stage": "Production",
                                "source": "mlflow_registry",
                                "run_id": run_id,
                            }
                        else:
                            raise ValueError("Production alias version not found")
                    else:
                        # Fallback if no Production alias found
                        metadata = {
                            "version": "production",
                            "model_type": "unknown",
                            "source": "mlflow_registry",
                        }
                except Exception as e:
                    logger.warning(f"Could not get Production model info: {e}")
                    metadata = {
                        "version": "production",
                        "model_type": "unknown",
                        "source": "mlflow_registry",
                    }
            except Exception as e:
                logger.warning(f"Could not get MLflow model metadata: {e}")
                metadata = {
                    "version": "production",
                    "model_type": "mlflow",
                    "source": "mlflow_registry",
                }
            return _mlflow_model, _mlflow_scaler, metadata

    # Legacy model loading
    cache_key = f"{model_type.lower()}_v{version}"

    if cache_key not in _model_cache:
        logger.info(f"Loading model: {model_type} v{version}")
        start_time = time.time()

        model_class = MODEL_CLASSES.get(model_type.lower())
        if not model_class:
            raise ValueError(
                f"Unknown model type: {model_type}. Choose from: {list(MODEL_CLASSES.keys())}"
            )

        try:
            model, scaler, metadata = load_model_version(
                version=version, model_class=model_class, device=_device
            )
            _model_cache[cache_key] = model
            _scaler_cache[cache_key] = scaler
            _metadata_cache[cache_key] = metadata

            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
        except FileNotFoundError as e:
            logger.error(f"Model not found: {e}")
            raise HTTPException(
                status_code=404, detail=f"Model version {version} of type {model_type} not found"
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    return _model_cache[cache_key], _scaler_cache[cache_key], _metadata_cache[cache_key]


@app.get("/", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.

    Returns:
        Health status of the API
    """
    return HealthResponse(
        status="healthy", message="Predictive Maintenance API is running", version="1.0.0"
    )


@app.get("/health")
def health():
    """Simple health check for load balancers."""
    return {"status": "ok"}


@app.get("/models", response_model=dict)
def list_available_models():
    """
    List all available model versions.
    Shows MLflow models if MLflow is enabled, otherwise shows legacy registry models.

    Returns:
        Dictionary with list of available models and their metadata
    """
    try:
        # If MLflow is enabled, show MLflow models
        if _use_mlflow and MLFLOW_AVAILABLE:
            from src.mlflow_utils import list_registered_models
            from mlflow.tracking import MlflowClient
            
            client = MlflowClient()
            registered_models = list_registered_models()
            
            # Get detailed info for each model version
            mlflow_models = []
            for model_info in registered_models:
                version_num = int(model_info["version"])
                run_id = model_info["run_id"]
                
                # Get run details to extract metrics and params
                try:
                    run = client.get_run(run_id)
                    params = run.data.params
                    metrics = run.data.metrics
                    
                    # Format similar to legacy registry
                    formatted_model = {
                        "version": version_num,
                        "model_type": params.get("model_type", "unknown"),
                        "rmse": metrics.get("test_rmse", 0.0),
                        "mae": metrics.get("test_mae", 0.0),
                        "timestamp": model_info.get("created_at", ""),
                        "sequence_length": int(params.get("sequence_length", 30)),
                        "input_features": 24,  # Standard for this project
                        "stage": model_info.get("stage", "None"),  # For backward compatibility
                        "aliases": model_info.get("aliases", []),  # New: actual aliases
                        "run_id": run_id,
                        "epochs": int(params.get("epochs", 0)),
                        "batch_size": int(params.get("batch_size", 64)),
                        "learning_rate": float(params.get("learning_rate", 0.001)),
                        "final_loss": metrics.get("train_loss", 0.0),
                        "source": "mlflow_registry",
                    }
                    mlflow_models.append(formatted_model)
                except Exception as e:
                    logger.warning(f"Could not get details for MLflow model version {version_num}: {e}")
                    # Add basic info even if details fail
                    mlflow_models.append({
                        "version": version_num,
                        "model_type": "unknown",
                        "stage": model_info.get("stage", "None"),
                        "source": "mlflow_registry",
                    })
            
            logger.info(f"Listing {len(mlflow_models)} MLflow model versions")
            return {"count": len(mlflow_models), "models": mlflow_models, "source": "mlflow"}
        
        # Legacy registry
        versions = list_model_versions()
        logger.info(f"Listing {len(versions)} legacy model versions")
        return {"count": len(versions), "models": versions, "source": "legacy"}
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
        raise HTTPException(status_code=404, detail=f"Model version {version} not found")
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict_rul(
    request: PredictionRequest,
    version: int = Query(1, description="Model version to use", ge=1),
    model_type: ModelType = Query(ModelType.LSTM, description="Model type to use"),
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
                status_code=400, detail="Invalid input format: sensor_readings must be a 2D array"
            )

        if sensor_data.shape[1] != len(FEATURE_COLS):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(FEATURE_COLS)} features per reading, got {sensor_data.shape[1]}",
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
            sensor_data_scaled.reshape(1, SEQ_LENGTH, len(FEATURE_COLS)), dtype=torch.float32
        ).to(_device)

        # Run inference
        model.eval()
        with torch.no_grad():
            prediction = model(sensor_tensor).cpu().numpy()[0, 0]

        # Ensure non-negative RUL
        predicted_rul = max(0.0, float(prediction))

        inference_time = time.time() - start_time
        logger.info(f"Prediction completed in {inference_time:.3f}s: RUL={predicted_rul:.2f}")

        # Use metadata if available (for MLflow), otherwise use query params
        response_version = metadata.get("version", version)
        response_model_type = metadata.get("model_type", model_type.value)
        
        return PredictionResponse(
            predicted_rul=predicted_rul,
            model_version=response_version,
            model_type=response_model_type,
            confidence_interval=None,  # Can be added later with ensemble models
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal server error during prediction: {str(e)}"
        )


@app.post("/predict/simple", response_model=PredictionResponse)
def predict_rul_simple(
    request: SimplePredictionRequest,
    version: int = Query(1, description="Model version to use", ge=1),
    model_type: ModelType = Query(ModelType.LSTM, description="Model type to use"),
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
                status_code=400, detail="Invalid input format: sensor_data must be a 2D array"
            )

        if sensor_data.shape[1] != len(FEATURE_COLS):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(FEATURE_COLS)} features per reading, got {sensor_data.shape[1]}",
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
            sensor_data_scaled.reshape(1, SEQ_LENGTH, len(FEATURE_COLS)), dtype=torch.float32
        ).to(_device)

        # Run inference
        model.eval()
        with torch.no_grad():
            prediction = model(sensor_tensor).cpu().numpy()[0, 0]

        predicted_rul = max(0.0, float(prediction))

        inference_time = time.time() - start_time
        logger.info(f"Prediction completed in {inference_time:.3f}s: RUL={predicted_rul:.2f}")

        # Use metadata if available (for MLflow), otherwise use query params
        response_version = metadata.get("version", version)
        response_model_type = metadata.get("model_type", model_type.value)

        return PredictionResponse(
            predicted_rul=predicted_rul,
            model_version=response_version,
            model_type=response_model_type,
            confidence_interval=None,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal server error during prediction: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
