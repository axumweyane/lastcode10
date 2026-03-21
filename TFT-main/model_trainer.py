"""
🚀 TFT Model Training Pipeline
Generated using Advanced Copilot Prompts
"""

import asyncio
import asyncpg
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MAPE
from sklearn.preprocessing import RobustScaler
import mlflow
import mlflow.pytorch
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightningTFT(pl.LightningModule):
    """PyTorch Lightning wrapper for TemporalFusionTransformer"""
    
    def __init__(self, model):
        super().__init__()
        self.tft_model = model
        # Copy important attributes
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, x):
        return self.tft_model(x)
        
    def training_step(self, batch, batch_idx):
        # Manually set trainer reference
        self.tft_model._trainer = self.trainer
        return self.tft_model.training_step(batch, batch_idx)
        
    def validation_step(self, batch, batch_idx):
        # Manually set trainer reference  
        self.tft_model._trainer = self.trainer
        return self.tft_model.validation_step(batch, batch_idx)
        
    def configure_optimizers(self):
        return self.tft_model.configure_optimizers()
        
    def predict_step(self, batch, batch_idx):
        self.tft_model._trainer = self.trainer
        return self.tft_model.predict_step(batch, batch_idx)

def train_tft_model_from_database():
    """
    FILE: model_trainer.py
    CONTEXT: TFT model training from PostgreSQL market data
    TASK: Build production TFT training pipeline with database integration
    INPUT: minute_aggregates table with OHLCV data, reddit sentiment data
    OUTPUT: Trained TFT model with <2.5% MAPE validation error
    
    DATABASE_SCHEMA:
        - minute_aggregates: ticker, timestamp, open, high, low, close, volume
        - reddit_sentiment_aggregated: ticker, sentiment_score, timestamp
        - Features: OHLCV + technical indicators + sentiment scores
    
    MODEL_ARCHITECTURE:
        - Temporal Fusion Transformer with attention mechanisms
        - Multi-horizon forecasting (1h, 4h, 24h predictions)
        - Input sequence length: 168 (7 days of hourly data)
        - Prediction horizons: [1, 4, 24] hours
    
    FEATURE_ENGINEERING:
        - Technical indicators: RSI, MACD, Bollinger Bands, EMA
        - Sentiment features: rolling sentiment, momentum, volatility
        - Time features: hour_of_day, day_of_week, month
        - Volume features: volume_sma, volume_ratio
    
    TRAINING_CONFIGURATION:
        - Batch size: 64 for GPU optimization
        - Learning rate: 1e-3 with ReduceLROnPlateau
        - Early stopping: patience=10, monitor='val_loss'
        - Max epochs: 100 with validation split 0.2
        - Loss function: QuantileLoss for prediction intervals
    
    PERFORMANCE_TARGETS:
        - Validation MAPE: <2.5% target
        - Training time: <2 hours on GPU
        - Memory usage: <8GB GPU memory
        - Model size: <50MB for deployment
    
    MLFLOW_INTEGRATION:
        - Log hyperparameters, metrics, and model artifacts
        - Track training progress and model versions
        - Enable model comparison and A/B testing
        - Auto-register best performing models
    
    CONSTRAINTS:
        - Use async database connections for performance
        - Implement data validation and quality checks
        - Handle missing data with forward fill
        - Scale features using RobustScaler
        - Save model checkpoints every 10 epochs
    """
    
    async def load_market_data(conn: asyncpg.Connection, tickers: List[str], limit: int = 50000) -> pd.DataFrame:
        """Load OHLCV data from PostgreSQL with optimized async query"""
        logger.info(f"Loading market data for {len(tickers)} tickers...")
        
        # Convert window_start (nanoseconds) to timestamp
        query = """
        SELECT 
            ticker,
            TO_TIMESTAMP(window_start / 1000000000) as timestamp,
            open::float as open,
            high::float as high, 
            low::float as low,
            close::float as close,
            volume::int as volume,
            transactions::int as transactions
        FROM stocks_minute_candlesticks_example 
        WHERE ticker = ANY($1::text[])
        ORDER BY ticker, window_start
        LIMIT $2
        """
        
        rows = await conn.fetch(query, tickers, limit)
        
        # Convert to DataFrame with proper column names
        if rows:
            # Define explicit column names matching the SQL query
            columns = ['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'transactions']
            df = pd.DataFrame([dict(row) for row in rows])
        else:
            df = pd.DataFrame()
        
        if df.empty:
            raise ValueError(f"No market data found for tickers: {tickers}")
            
        logger.info(f"Loaded {len(df)} market data records")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"First few rows:\n{df.head()}")
        return df
    
    async def load_sentiment_data(conn: asyncpg.Connection, tickers: List[str]) -> pd.DataFrame:
        """Load Reddit sentiment data"""
        logger.info(f"Loading sentiment data for {len(tickers)} tickers...")
        
        query = """
        SELECT 
            ticker,
            time_window as timestamp,
            avg_sentiment::float as sentiment_score,
            mention_count::int as mention_count,
            positive_mentions::int as positive_mentions,
            negative_mentions::int as negative_mentions
        FROM reddit_sentiment_aggregated 
        WHERE ticker = ANY($1::text[])
        ORDER BY ticker, time_window
        """
        
        rows = await conn.fetch(query, tickers)
        
        # Convert to DataFrame with proper column names
        if rows:
            # Define explicit column names matching the SQL query
            columns = ['ticker', 'timestamp', 'sentiment_score', 'mention_count', 'positive_mentions', 'negative_mentions']
            df = pd.DataFrame([dict(row) for row in rows])
        else:
            df = pd.DataFrame()
        
        if df.empty:
            logger.warning(f"No sentiment data found for tickers: {tickers}")
            return pd.DataFrame()
            
        logger.info(f"Loaded {len(df)} sentiment records")
        return df
    
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset"""
        logger.info("Computing technical indicators...")
        
        df = df.copy()
        df = df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
        
        # Group by ticker for technical indicators
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df.loc[mask].copy()
            
            # RSI (14-period)
            delta = ticker_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df.loc[mask, 'rsi'] = 100 - (100 / (1 + rs))
            
            # EMA (Exponential Moving Average)
            df.loc[mask, 'ema_20'] = ticker_data['close'].ewm(span=20).mean()
            df.loc[mask, 'ema_50'] = ticker_data['close'].ewm(span=50).mean()
            
            # Bollinger Bands
            rolling_mean = ticker_data['close'].rolling(window=20).mean()
            rolling_std = ticker_data['close'].rolling(window=20).std()
            df.loc[mask, 'bb_upper'] = rolling_mean + (rolling_std * 2)
            df.loc[mask, 'bb_lower'] = rolling_mean - (rolling_std * 2)
            df.loc[mask, 'bb_middle'] = rolling_mean
            
            # Volume indicators
            df.loc[mask, 'volume_sma'] = ticker_data['volume'].rolling(window=20).mean()
            df.loc[mask, 'volume_ratio'] = ticker_data['volume'] / df.loc[mask, 'volume_sma']
            
            # Price returns
            df.loc[mask, 'returns'] = ticker_data['close'].pct_change()
            df.loc[mask, 'returns_1h'] = ticker_data['close'].pct_change(60)  # 1-hour returns
            
        logger.info("Technical indicators computed successfully")
        return df
    
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        
        # Market session indicators (timezone-aware: convert to US/Eastern)
        eastern = ZoneInfo("America/New_York")
        ts_eastern = df['timestamp'].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT').dt.tz_convert(eastern) if df['timestamp'].dt.tz is None else df['timestamp'].dt.tz_convert(eastern)
        et_hour = ts_eastern.dt.hour
        df['is_market_open'] = ((et_hour >= 9) & (et_hour < 16)).astype(int)
        df['is_premarket'] = ((et_hour >= 4) & (et_hour < 9)).astype(int)
        df['is_afterhours'] = ((et_hour >= 16) | (et_hour < 4)).astype(int)
        
        return df
    
    async def prepare_dataset(tickers: List[str] = ['AAPL'], limit: int = 50000) -> pd.DataFrame:
        """Prepare the complete dataset with all features"""
        
        logger.info("Starting dataset preparation...")
        
        # Database connection
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='kibrom',
            password='beriha@123KB!',
            database='stock_trading_analysis'
        )
        
        try:
            # Load market data
            logger.info("Loading market data...")
            market_df = await load_market_data(conn, tickers, limit)
            logger.info(f"Market data loaded. Columns: {market_df.columns.tolist()}")
            
            # Load sentiment data
            logger.info("Loading sentiment data...")
            sentiment_df = await load_sentiment_data(conn, tickers)
            
            # Merge data if sentiment exists
            if not sentiment_df.empty:
                logger.info("Merging market and sentiment data...")
                # Align timestamps for merging
                market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
                sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
                
                # Merge on nearest timestamp within 1 hour window
                df = pd.merge_asof(
                    market_df.sort_values(['ticker', 'timestamp']),
                    sentiment_df.sort_values(['ticker', 'timestamp']),
                    on='timestamp',
                    by='ticker',
                    direction='backward',
                    tolerance=pd.Timedelta('1H')
                )
            else:
                logger.info("No sentiment data found, using market data only...")
                df = market_df
                # Add dummy sentiment columns
                df['sentiment_score'] = 0.0
                df['mention_count'] = 0
                df['positive_mentions'] = 0
                df['negative_mentions'] = 0
            
            logger.info(f"Data after merge/processing. Columns: {df.columns.tolist()}")
            logger.info(f"Data shape: {df.shape}")
            
            # Verify we have the required columns before proceeding
            required_cols = ['ticker', 'timestamp']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                raise ValueError(f"DataFrame missing required columns: {missing_cols}")
            
            # Add technical indicators
            logger.info("Adding technical indicators...")
            df = add_technical_indicators(df)
            
            # Add time features  
            df = add_time_features(df)
            
            # Fill missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Add time index for TFT
            df = df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
            df['time_idx'] = df.groupby('ticker').cumcount()
            
            logger.info(f"Final dataset shape: {df.shape}")
            return df
            
        finally:
            await conn.close()
    
    def create_tft_dataset(df: pd.DataFrame, max_encoder_length: int = 168, 
                          max_prediction_length: int = 24) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """Create PyTorch Forecasting TimeSeriesDataSet"""
        
        # Define features
        time_varying_known_reals = [
            'hour', 'day_of_week', 'month', 'quarter',
            'is_market_open', 'is_premarket', 'is_afterhours'
        ]
        
        time_varying_unknown_reals = [
            'open', 'high', 'low', 'close', 'volume', 'transactions',
            'rsi', 'ema_20', 'ema_50', 'bb_upper', 'bb_lower', 'bb_middle',
            'volume_sma', 'volume_ratio', 'returns',
            'sentiment_score', 'mention_count'
        ]
        
        # Create training dataset
        training = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= x.time_idx.max() - max_prediction_length],
            time_idx='time_idx',
            target='returns_1h',  # Predict 1-hour returns
            group_ids=['ticker'],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(
                groups=['ticker'], transformation='softplus'
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        
        # Create validation dataset
        validation = TimeSeriesDataSet.from_dataset(
            training, 
            df,
            predict=True,
            stop_randomization=True
        )
        
        return training, validation
    
    def train_model(training_dataset: TimeSeriesDataSet, 
                   validation_dataset: TimeSeriesDataSet,
                   max_epochs: int = 100,
                   batch_size: int = 64) -> TemporalFusionTransformer:
        """Train the TFT model using Lightning wrapper"""
        
        # Create data loaders
        train_dataloader = training_dataset.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0
        )
        val_dataloader = validation_dataset.to_dataloader(
            train=False, batch_size=batch_size * 2, num_workers=0
        )
        
        # Configure model with simpler parameters
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=0.03,
            hidden_size=16,  # Much smaller for testing
            attention_head_size=1,  # Simplified
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4
        )
        
        # Wrap the model in Lightning module
        lightning_tft = LightningTFT(tft)
        
        # Configure PyTorch Lightning trainer with compatible settings
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=0,  # Force CPU for compatibility
            enable_progress_bar=True,
            enable_model_summary=True,
            logger=False  # Disable built-in logger to avoid conflicts
        )
        
        # Start MLflow run
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                'max_encoder_length': training_dataset.max_encoder_length,
                'max_prediction_length': training_dataset.max_prediction_length,
                'batch_size': batch_size,
                'learning_rate': 0.03,
                'hidden_size': 16,
                'attention_head_size': 1,
                'dropout': 0.1,
                'max_epochs': max_epochs
            })
            
            # Train model
            logger.info("Starting training...")
            trainer.fit(lightning_tft, train_dataloader, val_dataloader)
            
            # Log final metrics
            if hasattr(trainer, 'callback_metrics'):
                val_loss = trainer.callback_metrics.get('val_loss')
                if val_loss:
                    mlflow.log_metric('val_loss', float(val_loss))
            
            # Save the underlying TFT model
            try:
                mlflow.pytorch.log_model(lightning_tft.tft_model, "tft_model")
            except Exception as e:
                logger.warning(f"Could not log model to MLflow: {e}")
            
        return lightning_tft.tft_model
    
    # Main execution
    async def main():
        logger.info("🚀 Starting TFT Model Training Pipeline")
        
        try:
            # Set MLflow tracking URI
            os.makedirs('mlruns', exist_ok=True)
            mlflow.set_tracking_uri('file:./mlruns')
            mlflow.set_experiment('TFT_Stock_Prediction')
            
            # Prepare dataset
            tickers = ['AAPL']  # Start with just one ticker for testing
            logger.info(f"Preparing dataset for tickers: {tickers}")
            
            df = await prepare_dataset(tickers=tickers, limit=10000)  # Smaller dataset for testing
            
            # Create TFT datasets
            training_dataset, validation_dataset = create_tft_dataset(df, max_encoder_length=24, max_prediction_length=4)
            logger.info("Created TFT training and validation datasets")
            
            # Train model
            logger.info("Starting model training...")
            model = train_model(training_dataset, validation_dataset, max_epochs=5, batch_size=16)  # Reduced for testing
            
            # Evaluate model
            predictions = model.predict(validation_dataset.to_dataloader(train=False, batch_size=64))
            mape = MAPE()(predictions, validation_dataset[:]['target'])
            
            logger.info(f"✅ Training completed! Validation MAPE: {mape:.4f}")
            
            if mape < 0.025:  # 2.5% target
                logger.info("🎯 Target MAPE achieved!")
            else:
                logger.warning(f"⚠️  MAPE {mape:.4f} above 2.5% target")
                
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    # Run the pipeline
    asyncio.run(main())

if __name__ == "__main__":
    train_tft_model_from_database()
