#!/usr/bin/env python3
"""
Simplified TFT Predictor Service for Testing
"""

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict
import uvicorn

app = FastAPI(title="TFT Predictor Service", version="1.0.0")


class PredictionRequest(BaseModel):
    symbol: str
    features: List[float]


class PredictionResponse(BaseModel):
    symbol: str
    prediction: float
    confidence: float
    timestamp: datetime


@app.get("/")
async def root():
    return {"service": "TFT Predictor", "status": "running", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "tft-predictor"}


@app.post("/predict")
async def predict(request: PredictionRequest):
    # Mock prediction logic
    mock_prediction = sum(request.features) * 0.01 + 100.0  # Simple mock calculation
    return PredictionResponse(
        symbol=request.symbol,
        prediction=mock_prediction,
        confidence=0.85,
        timestamp=datetime.now(),
    )


@app.get("/model/info")
async def model_info():
    return {
        "model_name": "TFT",
        "version": "1.0.0",
        "features": ["price", "volume", "sentiment", "technical_indicators"],
        "last_trained": "2025-08-07T10:00:00Z",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
