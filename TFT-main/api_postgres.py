"""
FastAPI Real-Time Prediction Service with PostgreSQL Integration
Provides REST API endpoints for TFT model predictions using PostgreSQL data
"""

"""
# COPILOT PROMPT: Create FastAPI endpoint: /polygon/realtime-predict
# Input: List of Polygon-formatted symbols (e.g., 'O:SPY230818C00325000')
# Output: Predictions with Polygon's native symbol format
# Use Polygon's WebSocket client for live data streaming
# EXPECTED OUTPUT: WebSocket-enabled FastAPI endpoint with real-time processing
# POLYGON INTEGRATION: WebSocket streaming, options symbols, real-time predictions
"""


from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import os
from contextlib import asynccontextmanager
import uvicorn

from postgres_data_loader import PostgresDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class PredictionRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of stock symbols to predict")
    prediction_date: str = Field(..., description="Date for prediction (YYYY-MM-DD)")
    model_version: str = Field(default="latest", description="Model version to use")
    include_attention: bool = Field(default=False, description="Include attention weights")
    lookback_days: int = Field(default=90, description="Days to look back for features")

class DataValidationRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of stock symbols to validate")
    start_date: str = Field(..., description="Validation start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Validation end date (YYYY-MM-DD)")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database_connected: bool

# Global variables for app state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting up Stock Data PostgreSQL API service...")
    
    # Initialize database connection
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'database': os.getenv('POSTGRES_DB', 'stock_trading_analysis'),
        'user': os.getenv('POSTGRES_USER', 'trading_user'),
        'password': os.environ['POSTGRES_PASSWORD'],
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'schema': os.getenv('POSTGRES_SCHEMA', 'public')
    }
    
    try:
        app_state['db_loader'] = PostgresDataLoader(db_config)
        app_state['db_connected'] = True
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        app_state['db_connected'] = False
    
    # Initialize prediction cache
    app_state['prediction_cache'] = {}
    app_state['cache_ttl'] = 3600  # 1 hour TTL
    
    yield
    
    # Shutdown
    logger.info("Shutting down Stock Data PostgreSQL API service...")

# Create FastAPI app
app = FastAPI(
    title="Stock Data API with PostgreSQL",
    description="Data access and validation service for stock market data using PostgreSQL",
    version="1.0.0",
    lifespan=lifespan
)

# Dependency to get database loader
def get_db_loader() -> PostgresDataLoader:
    if not app_state.get('db_connected', False):
        raise HTTPException(status_code=503, detail="Database not connected")
    return app_state['db_loader']

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Stock Data API with PostgreSQL",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(db_loader: PostgresDataLoader = Depends(get_db_loader)):
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        database_connected=app_state.get('db_connected', False)
    )

@app.get("/symbols", response_model=List[str])
async def get_available_symbols(db_loader: PostgresDataLoader = Depends(get_db_loader)):
    """Get list of available stock symbols in database"""
    try:
        symbols = db_loader.get_available_symbols()
        return symbols
    except Exception as e:
        logger.error(f"Error getting symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbols/{symbol}/info", response_model=Dict[str, Any])
async def get_symbol_info(symbol: str, db_loader: PostgresDataLoader = Depends(get_db_loader)):
    """Get information about a specific symbol"""
    try:
        date_range = db_loader.get_date_range(symbol)
        
        if date_range['record_count'] == 0:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        # Get fundamental data
        fundamentals = db_loader.load_fundamentals([symbol])
        fundamental_info = fundamentals.iloc[0].to_dict() if not fundamentals.empty else {}
        
        return {
            "symbol": symbol,
            "date_range": date_range,
            "fundamentals": fundamental_info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting symbol info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate-data", response_model=Dict[str, Any])
async def validate_data(request: DataValidationRequest, 
                       db_loader: PostgresDataLoader = Depends(get_db_loader)):
    """Validate data quality for given symbols and date range"""
    try:
        validation_results = db_loader.validate_data_quality(
            request.symbols,
            request.start_date,
            request.end_date
        )
        return validation_results
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=Dict[str, Any])
async def predict(request: PredictionRequest):
    """Generate predictions for given symbols and date (placeholder - requires trained model)"""
    raise HTTPException(
        status_code=501, 
        detail="Prediction endpoint not implemented. Training functionality removed from this API."
    )

@app.delete("/cache/clear")
async def clear_cache():
    """Clear prediction cache"""
    app_state['prediction_cache'] = {}
    return {"message": "Cache cleared", "cleared_at": datetime.now().isoformat()}

@app.get("/cache/stats", response_model=Dict[str, Any])
async def get_cache_stats():
    """Get cache statistics"""
    cache = app_state['prediction_cache']
    now = datetime.now()
    
    # Count valid and expired entries
    valid_entries = 0
    expired_entries = 0
    
    for entry in cache.values():
        if (now - entry['timestamp']).seconds < app_state['cache_ttl']:
            valid_entries += 1
        else:
            expired_entries += 1
    
    return {
        "total_entries": len(cache),
        "valid_entries": valid_entries,
        "expired_entries": expired_entries,
        "cache_ttl_seconds": app_state['cache_ttl']
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "status_code": 404}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api_postgres:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
        log_level="info"
    )
