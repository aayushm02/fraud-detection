import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Mocking the FastAPI app for the test
app = FastAPI()

class PredictionRequest(BaseModel):
    customer_id: int
    amount: float
    merchant_category: str
    timestamp: str

class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_score: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/model/info")
def model_info():
    return {"model_name": "mock_model", "version": "1.0", "type": "xgboost"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return PredictionResponse(
        transaction_id="mock-123",
        fraud_probability=0.05,
        is_fraud=False,
        risk_score="LOW"
    )

@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(requests: List[PredictionRequest]):
    return [
        PredictionResponse(
            transaction_id=f"mock-{i}",
            fraud_probability=0.05,
            is_fraud=False,
            risk_score="LOW"
        )
        for i in range(len(requests))
    ]

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict():
    payload = {
        "customer_id": 12345,
        "amount": 99.99,
        "merchant_category": "retail",
        "timestamp": "2023-10-27T10:00:00Z"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud" in data

def test_predict_invalid_input():
    payload = {
        "customer_id": "invalid", # Should be int
        "amount": "not-a-number"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Validation error

def test_predict_batch():
    payload = [
        {"customer_id": 1, "amount": 10.0, "merchant_category": "food", "timestamp": "2023-10-27T10:00:00Z"},
        {"customer_id": 2, "amount": 20.0, "merchant_category": "retail", "timestamp": "2023-10-27T10:05:00Z"}
    ]
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    assert len(response.json()) == 2

def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    assert "model_name" in response.json()
