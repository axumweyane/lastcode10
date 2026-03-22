"""
Enhanced Data Preprocessing Pipeline for TFT Stock Prediction
"""

"""
# COPILOT PROMPT: Calculate technical indicators from Polygon's vwap:
# - Volume-weighted RSI(14) using vwap instead of close
# - MACD(12,26,9) using vwap for signal line calculation
# - Bollinger %B(20,2) with vwap-based standard deviation
# Handle corporate actions via Polygon's 'adjusted' flag
# EXPECTED OUTPUT: Enhanced technical indicator functions with VWAP integration
# POLYGON INTEGRATION: VWAP data, adjusted prices, volume-weighted calculations
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class StockDataPreprocessor:
    """
    Advanced stock data preprocessing for TFT model
    Handles normalization, feature engineering, and temporal indexing
    """

    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.imputer = KNNImputer(n_neighbors=5)
        self.is_fitted = False

    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators for each symbol"""
        df = df.copy()

        for symbol in df["symbol"].unique():
            symbol_mask = df["symbol"] == symbol
            symbol_data = df[symbol_mask].copy()

            # RSI
            delta = symbol_data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df.loc[symbol_mask, "rsi"] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = symbol_data["close"].ewm(span=12).mean()
            exp2 = symbol_data["close"].ewm(span=26).mean()
            df.loc[symbol_mask, "macd"] = exp1 - exp2
            df.loc[symbol_mask, "macd_signal"] = (
                df.loc[symbol_mask, "macd"].ewm(span=9).mean()
            )

            # Bollinger Bands
            rolling_mean = symbol_data["close"].rolling(window=20).mean()
            rolling_std = symbol_data["close"].rolling(window=20).std()
            df.loc[symbol_mask, "bollinger_upper"] = rolling_mean + (rolling_std * 2)
            df.loc[symbol_mask, "bollinger_lower"] = rolling_mean - (rolling_std * 2)
            df.loc[symbol_mask, "bollinger_ratio"] = (
                symbol_data["close"] - rolling_mean
            ) / rolling_std

            # Volume indicators
            df.loc[symbol_mask, "volume_sma"] = (
                symbol_data["volume"].rolling(window=20).mean()
            )
            df.loc[symbol_mask, "volume_ratio"] = (
                symbol_data["volume"] / df.loc[symbol_mask, "volume_sma"]
            )

            # Price momentum
            df.loc[symbol_mask, "returns_1d"] = symbol_data["close"].pct_change(1)
            df.loc[symbol_mask, "returns_5d"] = symbol_data["close"].pct_change(5)
            df.loc[symbol_mask, "returns_20d"] = symbol_data["close"].pct_change(20)

        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal and cyclical features"""
        df = df.copy()

        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Basic time features
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["quarter"] = df["timestamp"].dt.quarter
        df["year"] = df["timestamp"].dt.year

        # Cyclical encoding
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Market structure features
        df["is_month_end"] = df["timestamp"].dt.is_month_end.astype(int)
        df["is_quarter_end"] = df["timestamp"].dt.is_quarter_end.astype(int)

        return df

    def create_target_variable(
        self, df: pd.DataFrame, target_type: str = "returns", horizon: int = 1
    ) -> pd.DataFrame:
        """Create target variable for prediction"""
        df = df.copy()

        if target_type == "returns":
            # Future returns
            for symbol in df["symbol"].unique():
                symbol_mask = df["symbol"] == symbol
                symbol_data = df[symbol_mask].copy()
                future_returns = (
                    symbol_data["close"].shift(-horizon) / symbol_data["close"] - 1
                )
                df.loc[symbol_mask, "target"] = future_returns

        elif target_type == "classification":
            # Binary classification: up/down
            for symbol in df["symbol"].unique():
                symbol_mask = df["symbol"] == symbol
                symbol_data = df[symbol_mask].copy()
                future_returns = (
                    symbol_data["close"].shift(-horizon) / symbol_data["close"] - 1
                )
                df.loc[symbol_mask, "target"] = (future_returns > 0).astype(int)

        elif target_type == "quintile":
            # Cross-sectional quintile ranking
            df["future_returns"] = np.nan
            for symbol in df["symbol"].unique():
                symbol_mask = df["symbol"] == symbol
                symbol_data = df[symbol_mask].copy()
                future_returns = (
                    symbol_data["close"].shift(-horizon) / symbol_data["close"] - 1
                )
                df.loc[symbol_mask, "future_returns"] = future_returns

            # Rank within each time period
            df["target"] = df.groupby("timestamp")["future_returns"].transform(
                lambda x: (
                    pd.qcut(x, 5, labels=False, duplicates="drop")
                    if len(x) >= 5
                    else np.nan
                )
            )
            df = df.drop("future_returns", axis=1)

        return df

    def fit_transform(
        self, df: pd.DataFrame, target_type: str = "returns", target_horizon: int = 1
    ) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        print("Starting data preprocessing...")

        # Ensure data is sorted
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        # Create time index
        df["time_idx"] = df.groupby("symbol").cumcount()

        # Create technical indicators
        print("Creating technical indicators...")
        df = self.create_technical_indicators(df)

        # Create temporal features
        print("Creating temporal features...")
        df = self.create_temporal_features(df)

        # Create target variable
        print(
            f"Creating target variable (type: {target_type}, horizon: {target_horizon})..."
        )
        df = self.create_target_variable(df, target_type, target_horizon)

        # Handle categorical variables
        categorical_cols = (
            ["symbol", "sector", "industry"] if "sector" in df.columns else ["symbol"]
        )
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))

        # Scale numerical features per symbol
        print("Scaling numerical features...")
        numerical_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "macd",
            "macd_signal",
            "bollinger_ratio",
            "volume_ratio",
            "returns_1d",
            "returns_5d",
            "returns_20d",
        ]

        for symbol in df["symbol"].unique():
            symbol_mask = df["symbol"] == symbol
            if symbol not in self.scalers:
                self.scalers[symbol] = StandardScaler()

            # Get available columns
            available_cols = [col for col in numerical_cols if col in df.columns]
            if available_cols:
                df.loc[symbol_mask, available_cols] = self.scalers[
                    symbol
                ].fit_transform(df.loc[symbol_mask, available_cols])

        # Handle missing values
        print("Handling missing values...")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])

        # Remove rows with missing targets
        df = df.dropna(subset=["target"])

        self.is_fitted = True
        print(f"Preprocessing complete. Dataset shape: {df.shape}")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessors"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        df = df.copy()
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        # Create time index
        df["time_idx"] = df.groupby("symbol").cumcount()

        # Apply same transformations
        df = self.create_technical_indicators(df)
        df = self.create_temporal_features(df)

        # Apply fitted encoders and scalers
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                # Handle unseen categories
                df[col] = df[col].astype(str)
                mask = df[col].isin(encoder.classes_)
                df.loc[~mask, col] = encoder.classes_[0]  # Default to first class
                df[col] = encoder.transform(df[col])

        # Apply fitted scalers
        numerical_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "macd",
            "macd_signal",
            "bollinger_ratio",
            "volume_ratio",
            "returns_1d",
            "returns_5d",
            "returns_20d",
        ]

        for symbol in df["symbol"].unique():
            symbol_mask = df["symbol"] == symbol
            if symbol in self.scalers:
                available_cols = [col for col in numerical_cols if col in df.columns]
                if available_cols:
                    df.loc[symbol_mask, available_cols] = self.scalers[
                        symbol
                    ].transform(df.loc[symbol_mask, available_cols])

        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = self.imputer.transform(df[numeric_columns])

        return df


def load_sample_data() -> pd.DataFrame:
    """Load sample stock data for testing"""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

    data = []
    for symbol in symbols:
        for date in dates:
            if date.weekday() < 5:  # Only weekdays
                base_price = np.random.uniform(100, 500)
                data.append(
                    {
                        "symbol": symbol,
                        "timestamp": date,
                        "open": base_price * (1 + np.random.normal(0, 0.02)),
                        "high": base_price * (1 + np.random.uniform(0, 0.05)),
                        "low": base_price * (1 - np.random.uniform(0, 0.05)),
                        "close": base_price * (1 + np.random.normal(0, 0.02)),
                        "volume": np.random.randint(1000000, 10000000),
                        "sentiment": np.random.normal(0, 1),
                    }
                )

    return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the preprocessor
    print("Loading sample data...")
    df = load_sample_data()
    print(f"Sample data shape: {df.shape}")

    # Initialize and fit preprocessor
    preprocessor = StockDataPreprocessor()
    processed_df = preprocessor.fit_transform(
        df, target_type="returns", target_horizon=1
    )

    print(f"Processed data shape: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns)}")
    print(f"Target statistics:\n{processed_df['target'].describe()}")
