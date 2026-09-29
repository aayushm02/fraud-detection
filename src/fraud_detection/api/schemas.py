"""Pydantic schemas for the API."""
from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict


class TransactionInput(BaseModel):
    """Single transaction for prediction."""

    features: Dict[str, float]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": {
                    "V1": -1.35,
                    "V2": 1.19,
                    "V3": 0.27,
                    "Amount": 149.62,
                    "V4": -1.23,
                }
            }
        }
    )


class PredictionResponse(BaseModel):
    """Prediction response for a single transaction."""

    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    threshold: float
    risk_level: str
    top_risk_factors: List[Dict[str, Any]]
    processing_time_ms: float


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    transactions: List[TransactionInput]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[PredictionResponse]
    summary: Dict[str, Any]


class ModelInfoResponse(BaseModel):
    """Model information response."""

    name: str
    version: str
    metrics: Dict[str, float]
    feature_count: int
    trained_at: str
    threshold: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float
