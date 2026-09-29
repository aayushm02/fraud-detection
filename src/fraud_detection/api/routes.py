"""API route handlers."""
import time
import uuid
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
import pandas as pd

from fraud_detection.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionInput,
)
from fraud_detection.config import get_settings
from fraud_detection.models.registry import ModelRegistry
from fraud_detection.explainability.shap_explainer import FraudExplainer

router = APIRouter()
settings = get_settings()

class AppContext:
    registry: ModelRegistry = ModelRegistry()
    active_model = None
    model_info = {}
    start_time: float = time.time()
    explainer = None

app_context = AppContext()

def get_risk_level(prob: float) -> str:
    """Get risk level based on fraud probability."""
    if prob < 0.3:
        return "low"
    elif prob < 0.6:
        return "medium"
    elif prob < 0.85:
        return "high"
    else:
        return "critical"

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    uptime = time.time() - app_context.start_time
    is_loaded = app_context.active_model is not None
    version = app_context.model_info.get("version", "unknown") if is_loaded else "none"
    return HealthResponse(
        status="healthy" if is_loaded else "starting",
        model_loaded=is_loaded,
        version=version,
        uptime_seconds=uptime
    )

@router.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionInput) -> PredictionResponse:
    """Predict fraud for a single transaction.
    
    Note: The API expects pre-processed feature vectors.
    """
    if app_context.active_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    start_t = time.time()
    
    try:
        df = pd.DataFrame([transaction.features])
        
        # Assuming predict_proba returns array with prob of fraud at index 1
        prob = float(app_context.active_model.predict_proba(df)[0, 1])
        
        threshold = app_context.model_info.get("threshold", 0.5)
        is_fraud = prob >= threshold
        
        top_factors = []
        if app_context.explainer:
            try:
                top_factors = app_context.explainer.get_top_features(df, top_n=5)
            except Exception:
                top_factors = []
                
        proc_time = (time.time() - start_t) * 1000.0
        
        return PredictionResponse(
            transaction_id=str(uuid.uuid4()),
            fraud_probability=prob,
            is_fraud=is_fraud,
            threshold=threshold,
            risk_level=get_risk_level(prob),
            top_risk_factors=top_factors,
            processing_time_ms=proc_time
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predict fraud for a batch of transactions.
    
    Note: The API expects pre-processed feature vectors.
    """
    if app_context.active_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
        
    start_t = time.time()
    
    try:
        predictions = []
        fraud_count = 0
        threshold = app_context.model_info.get("threshold", 0.5)
        
        df = pd.DataFrame([t.features for t in request.transactions])
        probs = app_context.active_model.predict_proba(df)[:, 1]
        
        top_factors_list = []
        if app_context.explainer:
            try:
                top_factors_list = app_context.explainer.get_top_features(df, top_n=5)
                if not isinstance(top_factors_list[0], list):
                    top_factors_list = [top_factors_list] * len(probs)
            except Exception:
                top_factors_list = [[] for _ in range(len(probs))]
        else:
            top_factors_list = [[] for _ in range(len(probs))]
        
        for i, prob in enumerate(probs):
            prob = float(prob)
            is_fraud = prob >= threshold
            if is_fraud:
                fraud_count += 1
                
            predictions.append(PredictionResponse(
                transaction_id=str(uuid.uuid4()),
                fraud_probability=prob,
                is_fraud=is_fraud,
                threshold=threshold,
                risk_level=get_risk_level(prob),
                top_risk_factors=top_factors_list[i] if i < len(top_factors_list) else [],
                processing_time_ms=0.0
            ))
            
        total_time = (time.time() - start_t) * 1000.0
        for p in predictions:
            p.processing_time_ms = total_time / len(predictions)
            
        summary = {
            "total": len(predictions),
            "fraud_count": fraud_count,
            "avg_processing_time_ms": total_time / len(predictions)
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info() -> ModelInfoResponse:
    """Get information about the currently loaded model."""
    if app_context.active_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
        
    return ModelInfoResponse(
        name=app_context.model_info.get("name", "unknown"),
        version=app_context.model_info.get("version", "unknown"),
        metrics=app_context.model_info.get("metrics", {}),
        feature_count=len(app_context.model_info.get("feature_names", [])),
        trained_at=app_context.model_info.get("timestamp", "unknown"),
        threshold=app_context.model_info.get("threshold", 0.5)
    )

@router.get("/model/feature-importance")
async def get_feature_importance() -> List[Dict[str, float]]:
    """Get global feature importance."""
    if app_context.active_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
        
    if hasattr(app_context.active_model, "feature_importances_"):
        importances = app_context.active_model.feature_importances_
        features = app_context.model_info.get("feature_names", [])
        if len(features) == len(importances):
            return [{"feature": f, "importance": float(imp)} for f, imp in zip(features, importances)]
    
    return []
