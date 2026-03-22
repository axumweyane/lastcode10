"""
🚀 DYNAMIC TFT TRAINING PIPELINE
================================

Trains TFT model on ANY ticker using your existing database:
- Reddit sentiment features
- Market data (trades, quotes, candlesticks)
- TFT predictions table

Usage: python dynamic_tft_trainer.py --ticker NVDA --days 90
"""

import pandas as pd
import numpy as np
import psycopg2
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from sklearn.preprocessing import StandardScaler
import argparse
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicTFTTrainer:
    def __init__(self, ticker: str, db_name: str = "stock_trading_analysis"):
        self.ticker = ticker.upper()
        self.db_name = db_name
        self.connection = None
        self.model = None
        self.trainer = None

    def connect_database(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host="localhost", database=self.db_name, user="kibrom", port=5432
            )
            logger.info(f"✅ Connected to database: {self.db_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False

    def extract_features(self, days_back: int = 90):
        """
        Extract all features for the ticker from multiple tables
        Creates a unified dataset for TFT training
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        logger.info(f"🔍 Extracting features for {self.ticker} ({days_back} days)")

        # 1. Reddit Sentiment Features
        sentiment_query = f"""
        SELECT 
            ticker,
            time_window as timestamp,
            avg_sentiment,
            mention_count,
            positive_mentions,
            negative_mentions,
            CASE 
                WHEN dominant_sentiment = 'positive' THEN 1
                WHEN dominant_sentiment = 'negative' THEN -1
                ELSE 0
            END as sentiment_direction
        FROM reddit_sentiment_aggregated 
        WHERE ticker = '{self.ticker}'
        AND time_window >= '{start_date}'
        ORDER BY time_window
        """

        # 2. Market Data Features (Candlesticks)
        market_query = f"""
        SELECT 
            ticker,
            TO_TIMESTAMP(window_start) as timestamp,
            open, high, low, close, volume,
            transactions,
            (close - open) / open as return_1min,
            (high - low) / open as volatility_1min
        FROM stocks_minute_candlesticks_example 
        WHERE ticker = '{self.ticker}'
        AND TO_TIMESTAMP(window_start) >= '{start_date}'
        ORDER BY window_start
        """

        # 3. Trade Flow Features
        trades_query = f"""
        SELECT 
            ticker,
            TO_TIMESTAMP(sip_timestamp / 1000000000) as timestamp,
            price,
            size,
            price * size as trade_value,
            EXTRACT(HOUR FROM TO_TIMESTAMP(sip_timestamp / 1000000000)) as hour_of_day
        FROM stock_trades_sample 
        WHERE ticker = '{self.ticker}'
        AND TO_TIMESTAMP(sip_timestamp / 1000000000) >= '{start_date}'
        ORDER BY sip_timestamp
        """

        try:
            # Load data
            sentiment_df = pd.read_sql(sentiment_query, self.connection)
            market_df = pd.read_sql(market_query, self.connection)
            trades_df = pd.read_sql(trades_query, self.connection)

            logger.info(
                f"📊 Loaded: {len(sentiment_df)} sentiment, {len(market_df)} market, {len(trades_df)} trade records"
            )

            # Aggregate trades to 1-minute intervals
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
            trades_df.set_index("timestamp", inplace=True)

            trade_features = (
                trades_df.groupby("ticker")
                .resample("1min")
                .agg(
                    {
                        "price": ["mean", "std"],
                        "size": "sum",
                        "trade_value": "sum",
                        "hour_of_day": "first",
                    }
                )
                .reset_index()
            )

            # Flatten column names
            trade_features.columns = [
                "ticker",
                "timestamp",
                "avg_price",
                "price_volatility",
                "total_volume",
                "total_value",
                "hour_of_day",
            ]

            # Merge all features
            market_df["timestamp"] = pd.to_datetime(market_df["timestamp"])
            sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"])
            trade_features["timestamp"] = pd.to_datetime(trade_features["timestamp"])

            # Start with market data as base
            combined_df = market_df.copy()

            # Add trade features
            combined_df = pd.merge(
                combined_df, trade_features, on=["ticker", "timestamp"], how="left"
            )

            # Add sentiment features (forward fill for missing values)
            combined_df = pd.merge(
                combined_df, sentiment_df, on=["ticker", "timestamp"], how="left"
            )

            # Forward fill sentiment data (it's aggregated hourly/daily)
            sentiment_cols = [
                "avg_sentiment",
                "mention_count",
                "positive_mentions",
                "negative_mentions",
                "sentiment_direction",
            ]
            combined_df[sentiment_cols] = combined_df[sentiment_cols].fillna(
                method="ffill"
            )

            # Create additional technical features
            combined_df = self.create_technical_features(combined_df)

            # Clean and prepare for TFT
            combined_df = combined_df.dropna()
            combined_df = combined_df.sort_values("timestamp")

            logger.info(
                f"✅ Combined dataset: {len(combined_df)} records with {len(combined_df.columns)} features"
            )
            return combined_df

        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            return None

    def create_technical_features(self, df):
        """Create technical indicators and features"""

        # Price-based features
        df["price_change"] = df["close"].pct_change()
        df["price_acceleration"] = df["price_change"].diff()

        # Volume features
        df["volume_ma_5"] = df["volume"].rolling(5).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma_5"]

        # Volatility features
        df["true_range"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1)),
            ),
        )
        df["atr_14"] = df["true_range"].rolling(14).mean()

        # Momentum features
        df["rsi"] = self.calculate_rsi(df["close"])
        df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()

        # Sentiment momentum
        if "avg_sentiment" in df.columns:
            df["sentiment_ma_3"] = df["avg_sentiment"].rolling(3).mean()
            df["sentiment_momentum"] = df["avg_sentiment"] - df["sentiment_ma_3"]

        # Time features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_market_open"] = ((df["hour"] >= 9) & (df["hour"] <= 16)).astype(int)

        return df

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_tft_dataset(self, df, prediction_length=6, context_length=24):
        """
        Prepare data for TFT training
        - prediction_length: how many minutes ahead to predict
        - context_length: how many minutes of history to use
        """

        # Add time index
        df = df.reset_index(drop=True)
        df["time_idx"] = range(len(df))

        # Target variable (next period return)
        df["target"] = df["price_change"].shift(-prediction_length)

        # Remove rows without targets
        df = df[:-prediction_length]

        # Define static and dynamic features
        static_categoricals = ["ticker"]
        static_reals = []

        time_varying_known_categoricals = ["hour", "day_of_week", "is_market_open"]
        time_varying_known_reals = ["time_idx"]

        time_varying_unknown_categoricals = []
        time_varying_unknown_reals = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "price_change",
            "price_acceleration",
            "volume_ratio",
            "atr_14",
            "rsi",
            "macd",
            "target",
        ]

        # Add sentiment features if available
        if "avg_sentiment" in df.columns:
            time_varying_unknown_reals.extend(
                ["avg_sentiment", "mention_count", "sentiment_momentum"]
            )

        # Create TFT dataset
        max_prediction_length = prediction_length
        max_encoder_length = context_length

        training = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= df["time_idx"].max() - prediction_length],
            time_idx="time_idx",
            target="target",
            group_ids=["ticker"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(
                groups=["ticker"], transformation="softplus"
            ),
        )

        # Create validation dataset
        validation = TimeSeriesDataSet.from_dataset(
            training,
            df[
                lambda x: x.time_idx
                > df["time_idx"].max() - prediction_length - context_length
            ],
            predict=True,
            stop_randomization=True,
        )

        logger.info(
            f"✅ TFT Dataset prepared: {len(training)} training, {len(validation)} validation samples"
        )

        return training, validation

    def train_model(self, training_dataset, validation_dataset, epochs=30):
        """Train the TFT model"""

        # Create data loaders
        train_dataloader = training_dataset.to_dataloader(
            train=True, batch_size=64, num_workers=0
        )
        val_dataloader = validation_dataset.to_dataloader(
            train=False, batch_size=64 * 10, num_workers=0
        )

        # Configure TFT model
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,  # 7 quantiles by default
            loss=torch.nn.MSELoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        logger.info(
            f"🧠 TFT Model: {tft.hparams.hidden_size} hidden units, {tft.hparams.attention_head_size} attention heads"
        )

        # Train model
        trainer = pl.Trainer(
            max_epochs=epochs,
            gpus=0,  # Use CPU for now
            enable_model_summary=True,
            gradient_clip_val=0.1,
            limit_train_batches=50,  # Speed up training
            enable_checkpointing=True,
        )

        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        self.model = tft
        self.trainer = trainer

        logger.info("✅ Model training completed!")
        return tft

    def generate_predictions(self, validation_dataset):
        """Generate predictions and save to database"""

        # Make predictions
        val_dataloader = validation_dataset.to_dataloader(
            train=False, batch_size=64, num_workers=0
        )
        predictions = self.model.predict(val_dataloader, return_y=True)

        # Extract predictions and actuals
        predictions_tensor = predictions[0]
        actuals_tensor = predictions[1]

        # Convert to numpy
        pred_values = predictions_tensor.numpy()
        actual_values = actuals_tensor.numpy()

        # Calculate metrics
        mse = np.mean((pred_values - actual_values) ** 2)
        mae = np.mean(np.abs(pred_values - actual_values))

        logger.info(f"📊 Model Performance: MSE={mse:.6f}, MAE={mae:.6f}")

        # Save predictions to database
        self.save_predictions_to_db(pred_values, actual_values)

        return pred_values, actual_values

    def save_predictions_to_db(self, predictions, actuals):
        """Save predictions to tft_predictions table"""

        current_time = datetime.now()

        # Create prediction records
        prediction_records = []
        for i, (pred, actual) in enumerate(
            zip(predictions.flatten(), actuals.flatten())
        ):

            # Calculate confidence (inverse of prediction error)
            confidence = max(0.1, min(0.99, 1.0 / (1.0 + abs(pred - actual))))

            record = {
                "ticker": self.ticker,
                "timestamp": current_time,
                "predicted_return": float(pred),
                "direction": 1 if pred > 0 else -1,
                "confidence": float(confidence),
                "signal_strength": float(abs(pred)),
                "model_version": f"TFT_v1.0_{self.ticker}",
                "rank_score": float(abs(pred) * confidence),
            }
            prediction_records.append(record)

        # Insert to database
        try:
            cursor = self.connection.cursor()

            insert_query = """
            INSERT INTO tft_predictions 
            (ticker, timestamp, predicted_return, direction, confidence, signal_strength, model_version, rank_score)
            VALUES (%(ticker)s, %(timestamp)s, %(predicted_return)s, %(direction)s, 
                   %(confidence)s, %(signal_strength)s, %(model_version)s, %(rank_score)s)
            """

            cursor.executemany(insert_query, prediction_records)
            self.connection.commit()

            logger.info(f"✅ Saved {len(prediction_records)} predictions to database")

        except Exception as e:
            logger.error(f"❌ Failed to save predictions: {e}")

    def run_full_pipeline(self, days_back=90, epochs=30, prediction_length=6):
        """Run the complete training pipeline"""

        logger.info(f"🚀 Starting TFT Training Pipeline for {self.ticker}")

        # 1. Connect to database
        if not self.connect_database():
            return False

        # 2. Extract features
        df = self.extract_features(days_back)
        if df is None or len(df) < 100:
            logger.error("❌ Insufficient data for training")
            return False

        # 3. Prepare TFT datasets
        training_dataset, validation_dataset = self.prepare_tft_dataset(
            df, prediction_length=prediction_length
        )

        # 4. Train model
        model = self.train_model(training_dataset, validation_dataset, epochs)

        # 5. Generate predictions
        predictions, actuals = self.generate_predictions(validation_dataset)

        # 6. Close database connection
        self.connection.close()

        logger.info(f"🎉 Pipeline completed successfully for {self.ticker}!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Dynamic TFT Training Pipeline")
    parser.add_argument(
        "--ticker", required=True, help="Stock ticker to train on (e.g., NVDA)"
    )
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument(
        "--prediction-length", type=int, default=6, help="Minutes ahead to predict"
    )

    args = parser.parse_args()

    # Create trainer and run pipeline
    trainer = DynamicTFTTrainer(args.ticker)
    success = trainer.run_full_pipeline(
        days_back=args.days,
        epochs=args.epochs,
        prediction_length=args.prediction_length,
    )

    if success:
        print(f"🎯 SUCCESS: {args.ticker} model trained and predictions saved!")
        print(f"🔍 Check tft_predictions table for results")
        print(f"💡 Next: Use these predictions in your trading system")
    else:
        print(f"❌ FAILED: Could not train model for {args.ticker}")


if __name__ == "__main__":
    main()
