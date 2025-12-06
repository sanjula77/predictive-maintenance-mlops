"""
Pydantic schemas for API request/response validation.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelType(str, Enum):
    """Supported model types."""

    LSTM = "lstm"
    BILSTM = "bilstm"
    GRU = "gru"
    TRANSFORMER = "transformer"


class SensorReading(BaseModel):
    """Single sensor reading with all features."""

    op_setting_1: float = Field(0.0, description="Operating setting 1", ge=0.0)
    op_setting_2: float = Field(0.0, description="Operating setting 2", ge=0.0)
    op_setting_3: float = Field(100.0, description="Operating setting 3", ge=0.0)
    sensor_1: float = Field(..., description="Sensor 1 reading")
    sensor_2: float = Field(..., description="Sensor 2 reading")
    sensor_3: float = Field(..., description="Sensor 3 reading")
    sensor_4: float = Field(..., description="Sensor 4 reading")
    sensor_5: float = Field(..., description="Sensor 5 reading")
    sensor_6: float = Field(..., description="Sensor 6 reading")
    sensor_7: float = Field(..., description="Sensor 7 reading")
    sensor_8: float = Field(..., description="Sensor 8 reading")
    sensor_9: float = Field(..., description="Sensor 9 reading")
    sensor_10: float = Field(..., description="Sensor 10 reading")
    sensor_11: float = Field(..., description="Sensor 11 reading")
    sensor_12: float = Field(..., description="Sensor 12 reading")
    sensor_13: float = Field(..., description="Sensor 13 reading")
    sensor_14: float = Field(..., description="Sensor 14 reading")
    sensor_15: float = Field(..., description="Sensor 15 reading")
    sensor_16: float = Field(..., description="Sensor 16 reading")
    sensor_17: float = Field(..., description="Sensor 17 reading")
    sensor_18: float = Field(..., description="Sensor 18 reading")
    sensor_19: float = Field(..., description="Sensor 19 reading")
    sensor_20: float = Field(..., description="Sensor 20 reading")
    sensor_21: float = Field(..., description="Sensor 21 reading")

    model_config = {"extra": "ignore"}

    def to_list(self) -> List[float]:
        """Convert to list in correct feature order."""
        from src.config import FEATURE_COLS

        return [getattr(self, col) for col in FEATURE_COLS]


class PredictionRequest(BaseModel):
    """Request model for RUL prediction."""

    sensor_readings: Optional[List[SensorReading]] = Field(
        None,
        description="List of sensor readings (last 30 cycles recommended)",
    )
    sensor_data: Optional[List[SensorReading]] = Field(
        None,
        description="Alias for sensor_readings (alternative field name)",
    )

    @field_validator("sensor_readings", "sensor_data", mode="before")
    @classmethod
    def validate_readings(cls, v):
        """Validate sensor readings."""
        if v is not None and not v:
            raise ValueError("At least one sensor reading is required")
        return v

    @model_validator(mode="after")
    def check_at_least_one_field(self):
        """Ensure at least one field is provided."""
        if not self.sensor_readings and not self.sensor_data:
            raise ValueError("Either 'sensor_readings' or 'sensor_data' must be provided")
        return self

    def get_readings(self) -> List[SensorReading]:
        """Get sensor readings from either field."""
        if self.sensor_readings:
            return self.sensor_readings
        if self.sensor_data:
            return self.sensor_data
        raise ValueError("Either 'sensor_readings' or 'sensor_data' must be provided")

    model_config = {
        "json_schema_extra": {
            "example": {
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
                        "sensor_21": 23.4190,
                    }
                ]
                * 30  # 30 cycles
            }
        }
    }


class SimplePredictionRequest(BaseModel):
    """Simplified request that accepts raw arrays."""

    sensor_data: List[List[float]] = Field(
        ...,
        description="2D array: [cycles][features]. Each inner array should have 24 features in order: op_setting_1, op_setting_2, op_setting_3, sensor_1 through sensor_21",
        min_items=1,
        max_items=100,
    )

    @field_validator("sensor_data")
    @classmethod
    def validate_shape(cls, v):
        """Validate that each reading has correct number of features."""
        from src.config import FEATURE_COLS

        expected_features = len(FEATURE_COLS)

        if v is None:
            return v

        for i, reading in enumerate(v):
            if len(reading) != expected_features:
                raise ValueError(
                    f"Reading {i} has {len(reading)} features, but expected {expected_features}. "
                    f"Expected order: {', '.join(FEATURE_COLS)}"
                )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "sensor_data": [
                    [
                        0.0,
                        0.0,
                        100.0,
                        518.67,
                        641.82,
                        1589.70,
                        1400.60,
                        14.62,
                        21.61,
                        554.36,
                        2388.06,
                        9046.19,
                        1.3,
                        47.47,
                        521.66,
                        2388.02,
                        8138.62,
                        8.4195,
                        0.03,
                        392,
                        2388,
                        100.0,
                        39.06,
                        23.4190,
                    ]
                ]
                * 30
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response model for RUL prediction."""

    predicted_rul: float = Field(..., description="Predicted Remaining Useful Life")
    model_version: int = Field(..., description="Model version used")
    model_type: str = Field(..., description="Model type used")
    confidence_interval: Optional[dict] = Field(
        None, description="Confidence interval (if available)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_rul": 125.5,
                "model_version": 1,
                "model_type": "lstm",
                "confidence_interval": None,
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    version: int
    model_type: str
    rmse: float
    mae: float
    timestamp: str
    sequence_length: int
    input_features: int
