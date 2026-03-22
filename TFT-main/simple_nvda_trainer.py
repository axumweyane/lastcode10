"""
🚀 SIMPLE NVDA MODEL TRAINER
============================

Train a model on NVIDIA (NVDA) using your existing database
Works with: stock_trading_analysis database
"""

import pandas as pd
import numpy as np
import psycopg2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class NVDATrainer:
    def __init__(self):
        self.db_name = "stock_trading_analysis"
        self.ticker = "NVDA"
        self.connection = None
        self.model = None

    def connect_database(self):
        """Connect to PostgreSQL"""
        try:
            # Try different connection methods
            connection_params = [
                # Method 1: Direct connection as kibrom
                {
                    "host": "localhost",
                    "database": self.db_name,
                    "user": "kibrom",
                    "password": "",
                },
                # Method 2: Connection via postgres user
                {"host": "localhost", "database": self.db_name, "user": "postgres"},
                # Method 3: Unix socket connection
                {"database": self.db_name, "user": "kibrom"},
            ]

            for params in connection_params:
                try:
                    self.connection = psycopg2.connect(**params)
                    print(
                        f"✅ Connected to {self.db_name} as {params.get('user', 'default')}"
                    )
                    return True
                except:
                    continue

            print("❌ All connection methods failed")
            return False
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def load_nvda_data(self, days_back=30):
        """Load NVDA market data"""

        # Get recent data
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        start_timestamp = int(start_time.timestamp())

        query = f"""
        SELECT 
            ticker,
            window_start,
            open, high, low, close, volume, transactions,
            TO_TIMESTAMP(window_start) as datetime
        FROM stocks_minute_candlesticks_example 
        WHERE ticker = 'NVDA'
        AND window_start > {start_timestamp}
        ORDER BY window_start
        """

        try:
            df = pd.read_sql(query, self.connection)
            print(f"📊 Loaded {len(df)} NVDA records")
            return df
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return None

    def create_features(self, df):
        """Create trading features"""

        # Sort by time
        df = df.sort_values("window_start").reset_index(drop=True)

        # Price features
        df["price_change"] = df["close"].pct_change()
        df["price_range"] = (df["high"] - df["low"]) / df["open"]
        df["body_size"] = abs(df["close"] - df["open"]) / df["open"]

        # Volume features
        df["volume_ma_5"] = df["volume"].rolling(5).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma_5"]

        # Technical indicators
        df["sma_5"] = df["close"].rolling(5).mean()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["price_vs_sma5"] = (df["close"] - df["sma_5"]) / df["sma_5"]
        df["price_vs_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]

        # Momentum
        df["momentum_3"] = df["close"] / df["close"].shift(3) - 1
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1

        # Volatility
        df["volatility_5"] = df["price_change"].rolling(5).std()
        df["volatility_20"] = df["price_change"].rolling(20).std()

        # Time features
        df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
        df["day_of_week"] = pd.to_datetime(df["datetime"]).dt.dayofweek
        df["is_market_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 16)).astype(int)

        # Target: next period return
        df["target"] = df["price_change"].shift(-1)

        # Remove rows without target
        df = df[:-1]

        print(f"✅ Created features: {len(df)} samples")
        return df

    def train_model(self, df):
        """Train RandomForest model"""

        # Select features
        feature_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "price_change",
            "price_range",
            "body_size",
            "volume_ratio",
            "price_vs_sma5",
            "price_vs_sma20",
            "momentum_3",
            "momentum_10",
            "volatility_5",
            "volatility_20",
            "hour",
            "day_of_week",
            "is_market_hours",
        ]

        # Clean data
        df_clean = df[feature_columns + ["target"]].dropna()

        if len(df_clean) < 100:
            print("❌ Insufficient clean data for training")
            return None

        # Prepare training data
        X = df_clean[feature_columns]
        y = df_clean["target"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        print(f"🎯 Training: {len(X_train)} samples, Testing: {len(X_test)} samples")

        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)

        print(f"📈 Training R²: {train_r2:.4f}, MSE: {train_mse:.6f}")
        print(f"📊 Testing R²: {test_r2:.4f}, MSE: {test_mse:.6f}")

        # Feature importance
        feature_importance = pd.DataFrame(
            {"feature": feature_columns, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print("\n🔍 Top 5 Features:")
        print(feature_importance.head())

        return self.model

    def generate_predictions(self, df):
        """Generate predictions and save to database"""

        if self.model is None:
            print("❌ No trained model available")
            return

        # Get latest data for prediction
        feature_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "price_change",
            "price_range",
            "body_size",
            "volume_ratio",
            "price_vs_sma5",
            "price_vs_sma20",
            "momentum_3",
            "momentum_10",
            "volatility_5",
            "volatility_20",
            "hour",
            "day_of_week",
            "is_market_hours",
        ]

        # Get last 10 rows for prediction
        recent_data = df[feature_columns].tail(10).dropna()

        if len(recent_data) == 0:
            print("❌ No recent data for prediction")
            return

        # Make predictions
        predictions = self.model.predict(recent_data)

        # Save to database
        current_time = datetime.now()

        prediction_records = []
        for i, pred in enumerate(predictions):

            # Calculate confidence based on historical accuracy
            confidence = min(0.95, max(0.1, 0.7 - abs(pred) * 10))  # Simple confidence

            record = {
                "ticker": "NVDA",
                "timestamp": current_time,
                "predicted_return": float(pred),
                "direction": 1 if pred > 0 else -1,
                "confidence": float(confidence),
                "signal_strength": float(abs(pred)),
                "model_version": "RF_v1.0_NVDA",
            }
            prediction_records.append(record)

        # Insert to tft_predictions table
        try:
            cursor = self.connection.cursor()

            for record in prediction_records:
                insert_query = """
                INSERT INTO tft_predictions 
                (ticker, timestamp, predicted_return, direction, confidence, signal_strength, model_version)
                VALUES (%(ticker)s, %(timestamp)s, %(predicted_return)s, %(direction)s, 
                       %(confidence)s, %(signal_strength)s, %(model_version)s)
                """
                cursor.execute(insert_query, record)

            self.connection.commit()
            print(f"✅ Saved {len(prediction_records)} predictions to database")

        except Exception as e:
            print(f"❌ Failed to save predictions: {e}")

    def run_training_pipeline(self, days_back=30):
        """Run complete training pipeline"""

        print(f"🚀 Starting NVDA Training Pipeline")
        print(f"📅 Using last {days_back} days of data")

        # 1. Connect to database
        if not self.connect_database():
            return False

        # 2. Load NVDA data
        df = self.load_nvda_data(days_back)
        if df is None or len(df) < 100:
            print("❌ Insufficient NVDA data")
            return False

        # 3. Create features
        df_features = self.create_features(df)
        if df_features is None:
            return False

        # 4. Train model
        model = self.train_model(df_features)
        if model is None:
            return False

        # 5. Generate predictions
        self.generate_predictions(df_features)

        # 6. Close connection
        self.connection.close()

        print("🎉 NVDA Training Pipeline Completed!")
        print("💡 Check tft_predictions table for results")
        return True


def main():
    trainer = NVDATrainer()
    success = trainer.run_training_pipeline(days_back=30)

    if success:
        print("\n🎯 SUCCESS: NVDA model trained!")
        print("🔍 Next steps:")
        print(
            "  1. Check predictions: SELECT * FROM tft_predictions WHERE ticker='NVDA' ORDER BY timestamp DESC LIMIT 5;"
        )
        print("  2. Use predictions in your trading system")
        print("  3. Monitor model performance")
    else:
        print("\n❌ Training failed!")


if __name__ == "__main__":
    main()
