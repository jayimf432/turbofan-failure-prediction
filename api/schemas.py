from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Union
from datetime import datetime

class SensorData(BaseModel):
    """
    Schema for a single cycle of sensor data.
    """
    unit_number: int = Field(..., description="Unique identifier for the engine/unit")
    time_in_cycles: int = Field(..., description="Current operational cycle number", gt=0)
    
    # Settings
    setting_1: float = Field(..., description="Operational setting 1")
    setting_2: float = Field(..., description="Operational setting 2")
    setting_3: float = Field(..., description="Operational setting 3")
    
    # Sensors s_1 to s_21
    s_1: float
    s_2: float
    s_3: float
    s_4: float
    s_5: float
    s_6: float
    s_7: float
    s_8: float
    s_9: float
    s_10: float
    s_11: float
    s_12: float
    s_13: float
    s_14: float
    s_15: float
    s_16: float
    s_17: float
    s_18: float
    s_19: float
    s_20: float
    s_21: float

    class Config:
        schema_extra = {
            "example": {
                "unit_number": 1,
                "time_in_cycles": 1,
                "setting_1": -0.0007,
                "setting_2": -0.0004,
                "setting_3": 100.0,
                "s_1": 518.67, "s_2": 641.82, "s_3": 1589.70, "s_4": 1400.60,
                "s_5": 14.62, "s_6": 21.61, "s_7": 554.36, "s_8": 2388.06,
                "s_9": 9046.19, "s_10": 1.30, "s_11": 47.47, "s_12": 521.66,
                "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195, "s_16": 0.03,
                "s_17": 392, "s_18": 2388, "s_19": 100.0, "s_20": 39.06,
                "s_21": 23.4190
            }
        }

class PredictionRequest(BaseModel):
    """
    Request model for prediction.
    Expects a list of historical sensor data for a unit to enable feature engineering.
    """
    data: List[SensorData] = Field(..., min_items=1, description="List of sensor readings in chronological order")

class PredictionResponse(BaseModel):
    """
    Response model for prediction.
    """
    unit_number: int
    time_in_cycles: int
    failure_probability: float
    risk_level: str = Field(..., description="Low, Medium, or High risk")
    prediction: int = Field(..., description="1 if failure predicted within 30 cycles, 0 otherwise")
    contributing_factors: Optional[List[Dict[str, Union[str, float]]]] = None
    warning: Optional[str] = None

class BatchPredictionRequest(BaseModel):
    """
    Batch request for multiple units.
    """
    units: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    """
    Response for batch prediction.
    """
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    """
    API Health check response.
    """
    status: str
    model_loaded: bool
    model_version: str
    timestamp: datetime

class FeatureImportanceResponse(BaseModel):
    features: List[Dict[str, float]]
