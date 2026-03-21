"""
Simple TFT Training Test - Bypassing Lightning compatibility issues
"""

import asyncio
import os
import asyncpg
import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import mlflow
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def load_simple_data():
    """Load minimal data for testing"""
    conn = await asyncpg.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', 5432)),
        user=os.getenv('POSTGRES_USER', 'trading_user'),
        password=os.environ['POSTGRES_PASSWORD'],
        database=os.getenv('POSTGRES_DB', 'stock_trading_analysis'),
    )
    
    query = """
    SELECT 
        ticker,
        TO_TIMESTAMP(window_start / 1000000000) as timestamp,
        close::float as price
    FROM stocks_minute_candlesticks_example 
    WHERE ticker = 'AAPL'
    ORDER BY window_start
    LIMIT 1000
    """
    
    rows = await conn.fetch(query)
    await conn.close()
    
    # Create DataFrame with proper column names
    data = []
    for row in rows:
        data.append({
            'ticker': row['ticker'],
            'timestamp': row['timestamp'],
            'price': row['price']
        })
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add time index and simple features
    df = df.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
    df['time_idx'] = df.index
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Add target (next price)
    df['target'] = df['price'].shift(-1)
    df = df.dropna()
    
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    return df

def create_simple_dataset(df):
    """Create simple TFT dataset"""
    
    # Split data
    cutoff = int(len(df) * 0.8)
    
    training = TimeSeriesDataSet(
        df[:cutoff],
        time_idx='time_idx',
        target='target',
        group_ids=['ticker'],
        min_encoder_length=10,
        max_encoder_length=20,
        min_prediction_length=1,
        max_prediction_length=5,
        time_varying_known_reals=['hour', 'day_of_week'],
        time_varying_unknown_reals=['price'],
        target_normalizer=GroupNormalizer(groups=['ticker']),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True
    )
    
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    
    return training, validation

def manual_train(training_dataset, validation_dataset, epochs=3):
    """Manual training without Lightning"""
    
    # Create data loaders
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=8, num_workers=0)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=16, num_workers=0)
    
    # Create model
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=8,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=4,
        output_size=7,
        loss=QuantileLoss(),
    )
    
    # Manual training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    
    with mlflow.start_run():
        mlflow.log_params({
            'epochs': epochs,
            'batch_size': 8,
            'learning_rate': 0.03,
            'hidden_size': 8
        })
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_losses = []
            
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                # Forward pass
                x, y = batch
                y_hat = model(x)
                
                # Handle output format - TFT outputs quantiles [batch, time, quantiles]
                if isinstance(y_hat, tuple):
                    y_hat = y_hat[0]
                if isinstance(y, tuple):
                    y = y[0]
                
                # For quantile outputs, use the median (middle quantile)
                if hasattr(y_hat, 'shape') and len(y_hat.shape) == 3:
                    # Take median quantile (index 3 out of 7 quantiles: 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9)
                    y_hat = y_hat[:, :, 3]  # 0.5 quantile (median)
                
                loss = torch.nn.functional.mse_loss(y_hat, y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_dataloader:
                    x, y = batch
                    y_hat = model(x)
                    
                    # Handle output format - TFT outputs quantiles
                    if isinstance(y_hat, tuple):
                        y_hat = y_hat[0]
                    if isinstance(y, tuple):
                        y = y[0]
                    
                    # For quantile outputs, use the median
                    if hasattr(y_hat, 'shape') and len(y_hat.shape) == 3:
                        y_hat = y_hat[:, :, 3]  # 0.5 quantile (median)
                    
                    loss = torch.nn.functional.mse_loss(y_hat, y)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            
            logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            mlflow.log_metrics({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, step=epoch)
        
        # Save model
        try:
            mlflow.pytorch.log_model(model, "simple_tft_model")
            logger.info("Model saved to MLflow")
        except Exception as e:
            logger.warning(f"Could not save to MLflow: {e}")
    
    return model

async def main():
    logger.info("🚀 Starting Simple TFT Training")
    
    try:
        # Set up MLflow
        mlflow.set_tracking_uri('file:./mlruns')
        mlflow.set_experiment('Simple_TFT_Test')
        
        # Load data
        df = await load_simple_data()
        
        # Create datasets
        training, validation = create_simple_dataset(df)
        logger.info("Created datasets")
        
        # Train model
        model = manual_train(training, validation, epochs=3)
        
        logger.info("✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
