"""
🚀 TFT PREDICTOR MICROSERVICE
============================
GPU-optimized prediction service with model lifecycle management
Real-time TFT inference with automated retraining and versioning
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import redis.asyncio as redis
from kafka import KafkaConsumer, KafkaProducer
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import mlflow
import mlflow.pytorch
import os
from contextlib import asynccontextmanager
import psycopg2
from sqlalchemy import create_engine
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://tft_user:tft_password@localhost:5432/tft_trading")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_ARTIFACT_PATH = os.getenv("MODEL_ARTIFACT_PATH", "/app/models")

# GPU Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Kafka Topics
KAFKA_TOPICS = {
    "market_data": "market-data",
    "predictions": "tft-predictions",
    "model_updates": "model-updates",
    "system_health": "system-health"
}

# Pydantic Models
class PredictionRequest(BaseModel):
    ticker: str
    features: Dict[str, float]
    horizons: List[int] = [1, 4, 24]  # 1h, 4h, 24h

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]

class TrainingRequest(BaseModel):
    model_type: str = "incremental"
    retrain_threshold: float = 0.05
    force_retrain: bool = False

class TFTPredictor:
    def __init__(self):
        self.kafka_consumer = None
        self.kafka_producer = None
        self.redis_client = None
        self.db_engine = None
        self.current_model = None
        self.model_version = "v1.0.0"
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_performance = {"accuracy": 0.0, "mape": 0.0}
        
    async def initialize(self):
        """Initialize connections and load model"""
        try:
            # Initialize MLflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("TFT_Trading_Predictor")
            
            # Initialize connections
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                retries=3
            )
            
            self.redis_client = redis.from_url(REDIS_URL)
            self.db_engine = create_engine(DATABASE_URL)
            
            # Load or initialize model
            await self.load_model()
            
            # Start background tasks
            asyncio.create_task(self.model_performance_monitor())
            asyncio.create_task(self.consume_market_data())
            
            logger.info("TFT Predictor service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TFT Predictor: {e}")
            raise
    
    async def load_model(self):
        """Load the latest model from MLflow or create new one"""
        try:
            # Try to load latest model from MLflow
            client = mlflow.tracking.MlflowClient()
            try:
                latest_version = client.get_latest_versions(
                    name="tft_trading_model", 
                    stages=["Production"]
                )[0]
                model_uri = f"models:/tft_trading_model/{latest_version.version}"
                self.current_model = mlflow.pytorch.load_model(model_uri)
                self.model_version = f"v{latest_version.version}"
                logger.info(f"Loaded model version {self.model_version}")
            except:
                # Create new model if none exists
                self.current_model = self.create_new_model()
                logger.info("Created new TFT model")
                
            # Load scaler and feature columns
            await self.load_preprocessing_artifacts()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.current_model = self.create_new_model()
    
    def create_new_model(self) -> nn.Module:
        """Create a new TFT model architecture"""
        class SimplifiedTFT(nn.Module):
            def __init__(self, input_dim: int = 50, hidden_dim: int = 128, num_layers: int = 3):
                super(SimplifiedTFT, self).__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                
                # Feature embedding
                self.feature_embedding = nn.Linear(input_dim, hidden_dim)
                
                # LSTM layers for temporal modeling
                self.lstm = nn.LSTM(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.2
                )
                
                # Attention mechanism
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=8,
                    batch_first=True
                )
                
                # Output heads for different horizons
                self.output_heads = nn.ModuleDict({
                    '1h': nn.Linear(hidden_dim, 1),
                    '4h': nn.Linear(hidden_dim, 1),
                    '24h': nn.Linear(hidden_dim, 1)
                })
                
                # Confidence estimation
                self.confidence_head = nn.Linear(hidden_dim, 3)  # 3 horizons
                
            def forward(self, x):
                batch_size, seq_len, features = x.shape
                
                # Feature embedding
                embedded = self.feature_embedding(x)
                
                # LSTM processing
                lstm_out, (hidden, cell) = self.lstm(embedded)
                
                # Attention
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                
                # Use last timestep for prediction
                final_repr = attn_out[:, -1, :]
                
                # Generate predictions for different horizons
                predictions = {}
                for horizon, head in self.output_heads.items():
                    predictions[horizon] = head(final_repr)
                
                # Confidence scores
                confidence = torch.sigmoid(self.confidence_head(final_repr))
                
                return predictions, confidence
        
        return SimplifiedTFT().to(device)
    
    async def load_preprocessing_artifacts(self):
        """Load preprocessing artifacts from cache or database"""
        try:
            # Try to load from Redis cache first
            scaler_data = await self.redis_client.get("tft_scaler")
            features_data = await self.redis_client.get("tft_features")
            
            if scaler_data and features_data:
                self.scaler = pickle.loads(scaler_data)
                self.feature_columns = json.loads(features_data)
                logger.info("Loaded preprocessing artifacts from cache")
            else:
                # Load from database or create default
                await self.initialize_preprocessing_artifacts()
                
        except Exception as e:
            logger.warning(f"Could not load preprocessing artifacts: {e}")
            await self.initialize_preprocessing_artifacts()
    
    async def initialize_preprocessing_artifacts(self):
        """Initialize default preprocessing artifacts"""
        # Default feature columns (this would typically come from training data)
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_5', 'sma_20', 'rsi', 'macd', 'bollinger_upper',
            'sentiment_score', 'sentiment_momentum', 'bullish_pct',
            'hour', 'day_of_week', 'volatility'
        ] + [f'lag_{i}' for i in range(1, 25)]  # 24 hour lags
        
        # Initialize scaler with dummy data
        dummy_data = np.random.randn(100, len(self.feature_columns))
        self.scaler.fit(dummy_data)
        
        # Cache for future use
        await self.redis_client.set("tft_scaler", pickle.dumps(self.scaler), ex=3600)
        await self.redis_client.set("tft_features", json.dumps(self.feature_columns), ex=3600)
    
    async def predict_single(self, ticker: str, features: Dict[str, float], horizons: List[int]) -> Dict:
        """Generate predictions for a single ticker"""
        try:
            # Prepare feature vector
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0.0))
            
            # Scale features
            feature_array = np.array(feature_vector).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_array)
            
            # Convert to torch tensor
            input_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).to(device)
            
            # Generate predictions
            self.current_model.eval()
            with torch.no_grad():
                predictions, confidence = self.current_model(input_tensor)
            
            # Format results
            result = {
                "ticker": ticker,
                "timestamp": datetime.utcnow().isoformat(),
                "model_version": self.model_version,
                "predictions": {},
                "confidence": {}
            }
            
            horizon_mapping = {1: '1h', 4: '4h', 24: '24h'}
            for i, horizon in enumerate(horizons):
                if horizon in horizon_mapping:
                    horizon_key = horizon_mapping[horizon]
                    if horizon_key in predictions:
                        result["predictions"][f"{horizon}h"] = float(predictions[horizon_key].cpu().item())
                        result["confidence"][f"{horizon}h"] = float(confidence[0, i].cpu().item())
            
            # Cache prediction
            cache_key = f"prediction:{ticker}:{datetime.utcnow().strftime('%Y-%m-%d-%H')}"
            await self.redis_client.set(cache_key, json.dumps(result), ex=300)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}
    
    async def predict_batch(self, requests: List[PredictionRequest]) -> List[Dict]:
        """Generate predictions for multiple tickers"""
        results = []
        
        # Process in parallel batches
        batch_size = 32
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i+batch_size]
            batch_tasks = [
                self.predict_single(req.ticker, req.features, req.horizons)
                for req in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        
        return results
    
    async def trigger_training(self, request: TrainingRequest) -> Dict:
        """Trigger model retraining"""
        try:
            logger.info(f"Starting {request.model_type} training")
            
            # Check if retraining is needed
            if not request.force_retrain:
                current_mape = self.model_performance.get("mape", 0.0)
                if current_mape < request.retrain_threshold:
                    return {
                        "status": "skipped",
                        "reason": f"Current MAPE {current_mape} below threshold {request.retrain_threshold}"
                    }
            
            # Start training in background
            asyncio.create_task(self.train_model(request.model_type))
            
            return {
                "status": "started",
                "training_type": request.model_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Training trigger failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def train_model(self, training_type: str):
        """Perform model training"""
        try:
            with mlflow.start_run():
                # Load training data
                training_data = await self.load_training_data()
                
                if training_data is None or len(training_data) < 1000:
                    logger.warning("Insufficient training data")
                    return
                
                # Prepare training dataset
                X_train, y_train = self.prepare_training_data(training_data)
                
                # Create new model or update existing
                if training_type == "full":
                    model = self.create_new_model()
                else:
                    model = self.current_model
                
                # Training setup
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.MSELoss()
                
                # Training loop
                model.train()
                for epoch in range(50):  # Reduced for demo
                    total_loss = 0
                    for batch_idx, (batch_x, batch_y) in enumerate(X_train):
                        optimizer.zero_grad()
                        
                        predictions, confidence = model(batch_x)
                        
                        # Calculate loss for all horizons
                        loss = 0
                        for horizon in ['1h', '4h', '24h']:
                            if horizon in predictions and horizon in batch_y:
                                loss += criterion(predictions[horizon], batch_y[horizon])
                        
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    
                    if epoch % 10 == 0:
                        logger.info(f"Epoch {epoch}, Loss: {total_loss/len(X_train):.4f}")
                        mlflow.log_metric("training_loss", total_loss/len(X_train), step=epoch)
                
                # Evaluate model
                val_mape = await self.evaluate_model(model, training_data[-1000:])  # Use last 1000 for validation
                
                # Save model if improved
                if val_mape < self.model_performance.get("mape", float('inf')):
                    # Log model to MLflow
                    mlflow.pytorch.log_model(
                        model,
                        "tft_trading_model",
                        signature=mlflow.models.infer_signature(
                            X_train[0][0][:1].cpu().numpy(),
                            {"1h": torch.randn(1, 1), "4h": torch.randn(1, 1), "24h": torch.randn(1, 1)}
                        )
                    )
                    
                    # Update current model
                    self.current_model = model
                    self.model_performance["mape"] = val_mape
                    
                    # Increment version
                    version_parts = self.model_version.replace('v', '').split('.')
                    new_version = f"v{version_parts[0]}.{int(version_parts[1])+1}.0"
                    self.model_version = new_version
                    
                    logger.info(f"Model updated to {self.model_version}, MAPE: {val_mape:.4f}")
                    
                    # Notify other services
                    await self.publish_model_update({
                        "version": self.model_version,
                        "mape": val_mape,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    async def load_training_data(self) -> Optional[pd.DataFrame]:
        """Load training data from database"""
        try:
            query = """
            SELECT md.*, sc.sentiment_score, sc.bullish_pct
            FROM market_data md
            LEFT JOIN sentiment_metrics sc ON md.ticker = sc.ticker 
                AND DATE(md.timestamp) = DATE(sc.timestamp)
            WHERE md.timestamp >= NOW() - INTERVAL '30 days'
            ORDER BY md.timestamp
            """
            
            df = pd.read_sql(query, self.db_engine)
            return df if len(df) > 0 else None
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return None
    
    def prepare_training_data(self, data: pd.DataFrame) -> Tuple:
        """Prepare training data for model"""
        # This is a simplified version - in production you'd have more sophisticated feature engineering
        try:
            # Select numeric columns
            feature_cols = [col for col in data.columns if col in self.feature_columns]
            
            # Fill missing values
            data[feature_cols] = data[feature_cols].fillna(0)
            
            # Scale features
            X = self.scaler.fit_transform(data[feature_cols].values)
            
            # Create sequences for time series
            sequence_length = 24  # 24 hours
            sequences = []
            targets = []
            
            for i in range(sequence_length, len(X) - 24):  # Ensure we have future targets
                sequences.append(X[i-sequence_length:i])
                
                # Targets for different horizons (simplified)
                target_1h = data.iloc[i+1]['close'] / data.iloc[i]['close'] - 1
                target_4h = data.iloc[i+4]['close'] / data.iloc[i]['close'] - 1 if i+4 < len(data) else 0
                target_24h = data.iloc[i+24]['close'] / data.iloc[i]['close'] - 1 if i+24 < len(data) else 0
                
                targets.append({
                    '1h': torch.tensor([[target_1h]], dtype=torch.float32),
                    '4h': torch.tensor([[target_4h]], dtype=torch.float32),
                    '24h': torch.tensor([[target_24h]], dtype=torch.float32)
                })
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(sequences).to(device)
            
            # Create batches
            batch_size = 16
            batches = []
            for i in range(0, len(X_tensor), batch_size):
                batch_x = X_tensor[i:i+batch_size]
                batch_y = {}
                for horizon in ['1h', '4h', '24h']:
                    batch_targets = torch.cat([targets[j][horizon] for j in range(i, min(i+batch_size, len(targets)))])
                    batch_y[horizon] = batch_targets.to(device)
                batches.append((batch_x, batch_y))
            
            return batches, targets
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return [], []
    
    async def evaluate_model(self, model: nn.Module, validation_data: pd.DataFrame) -> float:
        """Evaluate model performance"""
        try:
            model.eval()
            predictions = []
            actuals = []
            
            # Generate predictions on validation set
            feature_cols = [col for col in validation_data.columns if col in self.feature_columns]
            X_val = self.scaler.transform(validation_data[feature_cols].fillna(0).values)
            
            with torch.no_grad():
                for i in range(24, len(X_val) - 1):
                    sequence = torch.FloatTensor(X_val[i-24:i]).unsqueeze(0).to(device)
                    pred, _ = model(sequence)
                    
                    # Use 1h prediction for simplicity
                    if '1h' in pred:
                        predictions.append(pred['1h'].cpu().item())
                        actual = validation_data.iloc[i+1]['close'] / validation_data.iloc[i]['close'] - 1
                        actuals.append(actual)
            
            if len(predictions) > 0:
                mape = mean_absolute_percentage_error(actuals, predictions)
                return mape
            
            return float('inf')
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return float('inf')
    
    async def publish_model_update(self, update_info: Dict):
        """Publish model update to Kafka"""
        try:
            message = {
                "type": "model_update",
                "service": "tft-predictor",
                "data": update_info
            }
            self.kafka_producer.send(KAFKA_TOPICS["model_updates"], message)
            
        except Exception as e:
            logger.error(f"Failed to publish model update: {e}")
    
    async def consume_market_data(self):
        """Consume market data for real-time feature updates"""
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPICS["market_data"],
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                group_id="tft-predictor-group",
                enable_auto_commit=False,
            )

            for message in consumer:
                try:
                    data = message.value
                    # Update feature cache for real-time predictions
                    await self.update_feature_cache(data)
                    self.kafka_producer.flush()
                    consumer.commit()

                except Exception as e:
                    logger.error(f"Error processing market data: {e}")

        except Exception as e:
            logger.error(f"Kafka consumer error: {e}")
    
    async def update_feature_cache(self, market_data: Dict):
        """Update feature cache with latest market data"""
        try:
            ticker = market_data.get('ticker')
            if ticker:
                cache_key = f"features:{ticker}"
                await self.redis_client.set(cache_key, json.dumps(market_data), ex=300)
                
        except Exception as e:
            logger.error(f"Feature cache update failed: {e}")
    
    async def model_performance_monitor(self):
        """Monitor model performance and trigger retraining if needed"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Get recent predictions and actuals
                current_mape = await self.calculate_current_performance()
                self.model_performance["mape"] = current_mape
                
                # Trigger retraining if performance degrades
                if current_mape > 0.05:  # 5% threshold
                    logger.warning(f"Model performance degraded: MAPE {current_mape:.4f}")
                    await self.train_model("incremental")
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    async def calculate_current_performance(self) -> float:
        """Calculate current model performance"""
        try:
            # This would compare recent predictions with actual outcomes
            # Simplified implementation
            return self.model_performance.get("mape", 0.03)
            
        except Exception as e:
            logger.error(f"Performance calculation failed: {e}")
            return 0.05
    
    async def get_model_status(self) -> Dict:
        """Get current model status and performance"""
        return {
            "model_version": self.model_version,
            "performance": self.model_performance,
            "device": str(device),
            "last_training": "2024-01-01T00:00:00Z",  # Would track actual timestamp
            "prediction_count_24h": await self.redis_client.get("prediction_count") or 0
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.kafka_producer:
            self.kafka_producer.close()
        if self.redis_client:
            await self.redis_client.close()

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global tft_predictor
    tft_predictor = TFTPredictor()
    await tft_predictor.initialize()
    yield
    # Shutdown
    await tft_predictor.cleanup()

app = FastAPI(title="TFT Predictor Service", lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "tft-predictor", "timestamp": datetime.utcnow().isoformat()}

@app.post("/predict")
async def predict_single_endpoint(request: PredictionRequest):
    """Generate prediction for single ticker"""
    try:
        result = await tft_predictor.predict_single(request.ticker, request.features, request.horizons)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch_endpoint(request: BatchPredictionRequest):
    """Generate predictions for multiple tickers"""
    try:
        results = await tft_predictor.predict_batch(request.requests)
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def trigger_training_endpoint(request: TrainingRequest):
    """Trigger model training"""
    try:
        result = await tft_predictor.trigger_training(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status")
async def get_model_status():
    """Get current model status"""
    try:
        status = await tft_predictor.get_model_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/performance")
async def get_model_performance():
    """Get model performance metrics"""
    try:
        return tft_predictor.model_performance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
