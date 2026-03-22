#!/usr/bin/env python3
"""
Simple test service to verify FastAPI is working
"""

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import uvicorn

app = FastAPI(title="TFT Test Service", version="1.0.0")


class ServiceStatus(BaseModel):
    service_name: str
    status: str
    timestamp: datetime
    message: str


@app.get("/")
async def root():
    return {"message": "TFT Test Service is running!"}


@app.get("/health")
async def health_check():
    return ServiceStatus(
        service_name="TFT Test Service",
        status="healthy",
        timestamp=datetime.now(),
        message="All systems operational",
    )


@app.get("/services")
async def list_services():
    """List all available TFT services"""
    services = [
        {
            "name": "data-ingestion",
            "description": "Real-time data collection",
            "port": 8001,
        },
        {"name": "sentiment-engine", "description": "Sentiment analysis", "port": 8002},
        {"name": "tft-predictor", "description": "TFT model predictions", "port": 8003},
        {"name": "trading-engine", "description": "Order execution", "port": 8004},
        {"name": "orchestrator", "description": "Workflow coordination", "port": 8005},
    ]
    return {"total_services": len(services), "services": services}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
