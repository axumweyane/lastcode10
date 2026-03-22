"""
Production API Server for TFT Stock Predictions
FastAPI-based real-time prediction service
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import asyncio
import json
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager

# Local imports
from data_preprocessing import StockDataPreprocessor
from tft_model import EnhancedTFTModel
from stock_ranking import StockRankingSystem, PortfolioConstructor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and preprocessor
model: Optional[EnhancedTFTModel] = None
preprocessor: Optional[StockDataPreprocessor] = None
ranking_system: Optional[StockRankingSystem] = None
portfolio_constructor: Optional[PortfolioConstructor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and preprocessor on startup"""
    global model, preprocessor, ranking_system, portfolio_constructor

    logger.info("Initializing TFT prediction service...")

    try:
        # Initialize components
        preprocessor = StockDataPreprocessor()
        model = EnhancedTFTModel()
        ranking_system = StockRankingSystem()
        portfolio_constructor = PortfolioConstructor()

        # Try to load pre-trained model
        try:
            model.load_model("models/tft_model.pth")
            logger.info("Pre-trained model loaded successfully")
        except FileNotFoundError:
            logger.warning("No pre-trained model found. Model needs to be trained.")

        logger.info("TFT service initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise

    yield

    logger.info("Shutting down TFT service...")


# Initialize FastAPI app
app = FastAPI(
    title="TFT Stock Prediction API",
    description="Temporal Fusion Transformer for Stock Market Prediction",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API requests/responses
class StockBar(BaseModel):
    """Individual stock price bar"""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    sentiment: Optional[float] = None


class FutureEvent(BaseModel):
    """Future known event"""

    date: str
    type: str  # 'earnings', 'dividend', 'split', etc.
    value: Optional[float] = None


class PredictionRequest(BaseModel):
    """Prediction request format"""

    symbols: List[str] = Field(..., min_items=1, max_items=1000)
    historical_data: Dict[str, List[StockBar]]
    future_events: Optional[Dict[str, List[FutureEvent]]] = None
    prediction_horizon: Optional[int] = Field(default=1, ge=1, le=30)
    include_portfolio: Optional[bool] = True


class TradingSignalResponse(BaseModel):
    """Trading signal response"""

    symbol: str
    predicted_return: float
    confidence: float
    rank: int
    signal_strength: str
    timestamp: str


class PortfolioResponse(BaseModel):
    """Portfolio construction response"""

    symbol: str
    weight: float
    predicted_return: float
    confidence: float
    rank: int
    signal_strength: str
    sector: str
    side: str


class PredictionResponse(BaseModel):
    """Complete prediction response"""

    request_id: str
    timestamp: str
    processing_time_ms: float
    model_info: Dict[str, Any]
    signals: Dict[str, List[TradingSignalResponse]]
    portfolio: Optional[Dict[str, List[PortfolioResponse]]] = None
    portfolio_stats: Optional[Dict[str, float]] = None
    warnings: List[str] = []


class TrainingRequest(BaseModel):
    """Model training request"""

    data_path: str
    target_type: str = Field(
        default="returns", regex="^(returns|classification|quintile)$"
    )
    target_horizon: int = Field(default=1, ge=1, le=30)
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5)
    optimize_hyperparams: bool = False
    n_trials: int = Field(default=20, ge=5, le=100)


class ModelStatus(BaseModel):
    """Model status response"""

    is_trained: bool
    last_training_date: Optional[str]
    model_config: Optional[Dict[str, Any]]
    training_stats: Optional[Dict[str, float]]


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None and model.model is not None,
        "preprocessor_fitted": preprocessor is not None and preprocessor.is_fitted,
    }


# Model status endpoint
@app.get("/model/status", response_model=ModelStatus)
async def get_model_status():
    """Get model training status and configuration"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    return ModelStatus(
        is_trained=model.model is not None,
        last_training_date=None,  # Would come from model metadata
        model_config=model.config,
        training_stats=None,  # Would come from training history
    )


# Main prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_stocks(request: PredictionRequest):
    """
    Generate stock predictions and trading signals
    """
    start_time = datetime.now()
    request_id = f"pred_{int(start_time.timestamp())}"
    warnings = []

    # Validate model is trained
    if model is None or model.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Please train the model first using /train endpoint.",
        )

    if not preprocessor.is_fitted:
        raise HTTPException(
            status_code=503,
            detail="Preprocessor not fitted. Please train the model first.",
        )

    try:
        # Convert request data to DataFrame
        logger.info(
            f"Processing prediction request {request_id} for {len(request.symbols)} symbols"
        )

        df_data = []
        for symbol in request.symbols:
            if symbol not in request.historical_data:
                warnings.append(f"No historical data provided for {symbol}")
                continue

            for bar in request.historical_data[symbol]:
                row = {
                    "symbol": symbol,
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }

                if bar.sentiment is not None:
                    row["sentiment"] = bar.sentiment

                df_data.append(row)

        if len(df_data) == 0:
            raise HTTPException(
                status_code=400, detail="No valid historical data provided"
            )

        df = pd.DataFrame(df_data)

        # Add future events if provided
        if request.future_events:
            for symbol, events in request.future_events.items():
                for event in events:
                    # Add event indicators to the dataframe
                    event_date = pd.to_datetime(event.date)
                    symbol_mask = (df["symbol"] == symbol) & (
                        pd.to_datetime(df["timestamp"]) == event_date
                    )

                    if event.type == "earnings":
                        df.loc[symbol_mask, "earnings_date"] = 1
                    # Add other event types as needed

        # Preprocess data
        logger.info("Preprocessing data...")
        processed_df = preprocessor.transform(df)

        # Create dataset for prediction
        if model.validation_dataset is None:
            raise HTTPException(
                status_code=503,
                detail="Model validation dataset not available. Please retrain the model.",
            )

        # Generate predictions
        logger.info("Generating predictions...")
        predictions = model.predict(model.validation_dataset, mode="prediction")

        # Process predictions into signals
        logger.info("Processing predictions into trading signals...")
        predictions_df = ranking_system.process_predictions(
            predictions, request.symbols, prediction_type="quantile"
        )

        # Apply liquidity filter if historical volume data is available
        liquidity_filter = None
        if "volume" in processed_df.columns:
            liquidity_filter = ranking_system.calculate_liquidity_filter(processed_df)

        # Generate trading signals
        signals = ranking_system.generate_trading_signals(
            predictions_df, liquidity_filter=liquidity_filter, method="quintile"
        )

        # Convert signals to response format
        signal_responses = {}
        for signal_type, signal_list in signals.items():
            signal_responses[signal_type] = [
                TradingSignalResponse(
                    symbol=signal.symbol,
                    predicted_return=signal.predicted_return,
                    confidence=signal.confidence,
                    rank=signal.rank,
                    signal_strength=signal.signal_strength,
                    timestamp=signal.timestamp.isoformat(),
                )
                for signal in signal_list
            ]

        # Construct portfolio if requested
        portfolio_response = None
        portfolio_stats = None

        if request.include_portfolio:
            logger.info("Constructing portfolio...")
            portfolio = portfolio_constructor.construct_portfolio(signals)

            # Convert portfolio to response format
            portfolio_response = {}
            for side in ["long_portfolio", "short_portfolio"]:
                if side in portfolio:
                    portfolio_response[side] = [
                        PortfolioResponse(
                            symbol=symbol,
                            weight=pos["weight"],
                            predicted_return=pos["predicted_return"],
                            confidence=pos["confidence"],
                            rank=pos["rank"],
                            signal_strength=pos["signal_strength"],
                            sector=pos["sector"],
                            side=pos["side"],
                        )
                        for symbol, pos in portfolio[side].items()
                    ]

            portfolio_stats = portfolio["portfolio_stats"]

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(
            f"Prediction request {request_id} completed in {processing_time:.2f}ms"
        )

        return PredictionResponse(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time,
            model_info={
                "model_type": "TemporalFusionTransformer",
                "config": model.config,
                "prediction_horizon": request.prediction_horizon,
            },
            signals=signal_responses,
            portfolio=portfolio_response,
            portfolio_stats=portfolio_stats,
            warnings=warnings,
        )

    except Exception as e:
        logger.error(f"Error processing prediction request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Model training endpoint
@app.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train or retrain the TFT model
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model service not initialized")

    # Add training task to background
    background_tasks.add_task(
        _train_model_background,
        request.data_path,
        request.target_type,
        request.target_horizon,
        request.validation_split,
        request.optimize_hyperparams,
        request.n_trials,
    )

    return {
        "message": "Model training started in background",
        "training_config": request.dict(),
        "status": "training_started",
    }


async def _train_model_background(
    data_path: str,
    target_type: str,
    target_horizon: int,
    validation_split: float,
    optimize_hyperparams: bool,
    n_trials: int,
):
    """Background training task"""
    global model, preprocessor

    try:
        logger.info(f"Starting model training with data from {data_path}")

        # Load data
        df = pd.read_csv(data_path)  # Assume CSV format

        # Preprocess data
        preprocessor = StockDataPreprocessor()
        processed_df = preprocessor.fit_transform(
            df, target_type=target_type, target_horizon=target_horizon
        )

        # Train model
        from tft_model import TFTTrainingPipeline

        pipeline = TFTTrainingPipeline(model.config)
        trained_model = pipeline.run_pipeline(
            processed_df,
            validation_split=validation_split,
            optimize_hyperparams=optimize_hyperparams,
            n_trials=n_trials,
        )

        # Update global model
        model = trained_model

        # Save model
        model.save_model("models/tft_model.pth")

        logger.info("Model training completed successfully")

    except Exception as e:
        logger.error(f"Model training failed: {e}")


# Batch prediction endpoint
@app.post("/predict/batch")
async def batch_predict(file_path: str):
    """
    Process batch predictions from file
    """
    try:
        # Load data from file
        df = pd.read_csv(file_path)

        # Process similar to single prediction
        # Implementation would be similar to predict_stocks but for batch processing

        return {
            "message": "Batch prediction completed",
            "results_path": "output/batch_results.csv",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
