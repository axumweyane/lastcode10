"""
Enhanced Data Pipeline with PostgreSQL Integration
Builds complete datasets for TFT model training and prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Optional, Dict, Any, Tuple
import warnings
from postgres_data_loader import PostgresDataLoader, TechnicalIndicators

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class PostgresDataPipeline:
    """Enhanced data pipeline using PostgreSQL as data source"""
    
    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize data pipeline with PostgreSQL connection
        
        Args:
            db_config: Database configuration dictionary
        """
        self.loader = PostgresDataLoader(db_config)
        self.technical_indicators = TechnicalIndicators()
        
    def build_dataset(self, 
                     symbols: List[str], 
                     start_date: str, 
                     end_date: Optional[str] = None,
                     include_fundamentals: bool = True,
                     include_sentiment: bool = True,
                     include_earnings: bool = True,
                     target_type: str = 'returns',
                     prediction_horizon: int = 1) -> pd.DataFrame:
        """
        Build complete dataset from PostgreSQL for TFT model
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_fundamentals: Whether to include fundamental data
            include_sentiment: Whether to include sentiment data
            include_earnings: Whether to include earnings calendar
            target_type: Type of target variable ('returns', 'classification', 'quintile')
            prediction_horizon: Number of days ahead to predict
            
        Returns:
            Complete dataset ready for TFT training/prediction
        """
        logger.info(f"Building dataset for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # 1. Load core OHLCV data
        df = self.loader.load_ohlcv(symbols, start_date, end_date)
        if df.empty:
            raise ValueError("No OHLCV data found for specified symbols and date range")
        
        # 2. Add technical indicators
        df = self._add_all_technical_indicators(df)
        
        # 3. Add temporal features
        df = self._add_temporal_features(df)
        
        # 4. Merge with fundamental data
        if include_fundamentals:
            try:
                fundamentals = self.loader.load_fundamentals(symbols, end_date)
                if not fundamentals.empty:
                    df = df.merge(fundamentals[['symbol', 'market_cap', 'pe_ratio', 'eps', 
                                              'dividend_yield', 'sector', 'industry', 'exchange']], 
                                on='symbol', how='left')
                    logger.info("Added fundamental data")
                else:
                    logger.warning("No fundamental data available")
                    # Add placeholder columns with zeros (NaN breaks TimeSeriesDataSet)
                    df['market_cap'] = 0.0
                    df['pe_ratio'] = 0.0
                    df['eps'] = 0.0
                    df['dividend_yield'] = 0.0
                    df['sector'] = 'Unknown'
                    df['industry'] = 'Unknown'
                    df['exchange'] = 'Unknown'
            except Exception as e:
                logger.warning(f"Failed to load fundamentals: {e}")
                # Add placeholder columns with zeros (NaN breaks TimeSeriesDataSet)
                df['market_cap'] = 0.0
                df['pe_ratio'] = 0.0
                df['eps'] = 0.0
                df['dividend_yield'] = 0.0
                df['sector'] = 'Unknown'
                df['industry'] = 'Unknown'
                df['exchange'] = 'Unknown'
        
        # 5. Merge with sentiment data
        if include_sentiment:
            try:
                sentiment = self.loader.load_sentiment(symbols, start_date, end_date)
                if not sentiment.empty:
                    df = df.merge(sentiment, on=['symbol', 'date'], how='left')
                    # Fill missing sentiment values
                    df['sentiment_score'] = df['sentiment_score'].fillna(0)
                    df['sentiment_magnitude'] = df['sentiment_magnitude'].fillna(0)
                    df['news_count'] = df['news_count'].fillna(0)
                    logger.info("Added sentiment data")
                else:
                    logger.warning("No sentiment data available")
                    df['sentiment_score'] = 0
                    df['sentiment_magnitude'] = 0
                    df['news_count'] = 0
            except Exception as e:
                logger.warning(f"Failed to load sentiment: {e}")
                df['sentiment_score'] = 0
                df['sentiment_magnitude'] = 0
                df['news_count'] = 0
        
        # 6. Add earnings calendar flags
        if include_earnings:
            try:
                earnings = self.loader.load_earnings_calendar(symbols, start_date, end_date)
                if not earnings.empty:
                    # Create earnings flag for upcoming earnings
                    earnings_dates = set(earnings['earnings_date'].dt.strftime('%Y-%m-%d'))
                    df['earnings_flag'] = df['date'].astype(str).isin(earnings_dates).astype(int)
                    
                    # Add days until next earnings
                    df['days_to_earnings'] = self._calculate_days_to_earnings(df, earnings)
                    logger.info("Added earnings calendar data")
                else:
                    logger.warning("No earnings calendar data available")
                    df['earnings_flag'] = 0
                    df['days_to_earnings'] = 999  # Large number for no upcoming earnings
            except Exception as e:
                logger.warning(f"Failed to load earnings calendar: {e}")
                df['earnings_flag'] = 0
                df['days_to_earnings'] = 999
        
        # 7. Create target variable
        df = self._create_target_variable(df, target_type, prediction_horizon)
        
        # 8. Add group and time indices
        df = self._add_indices(df)
        
        # 9. Handle missing values and clean data
        df = self._clean_dataset(df)
        
        # 10. Filter out symbols with insufficient data
        df = self._filter_sufficient_data(df, min_observations=100)
        
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Symbols: {df['symbol'].nunique()}")
        
        return df
    
    def _add_all_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the dataset"""
        logger.info("Adding technical indicators...")
        
        # Ensure data is sorted
        df = df.sort_values(['symbol', 'date'])
        
        # Add moving averages
        df = self.technical_indicators.add_moving_averages(df)
        
        # Add RSI
        df = self.technical_indicators.add_rsi(df)
        
        # Add MACD
        df = self.technical_indicators.add_macd(df)
        
        # Add Bollinger Bands
        df = self.technical_indicators.add_bollinger_bands(df)
        
        # Add volume indicators
        df = self.technical_indicators.add_volume_indicators(df)
        
        # Add price-based features
        df['price_change'] = df.groupby('symbol')['adj_close'].pct_change()
        df['high_low_ratio'] = df['adj_high'] / df['adj_low']
        df['close_open_ratio'] = df['adj_close'] / df['adj_open']
        
        # Add volatility measures
        df['returns_volatility'] = df.groupby('symbol')['price_change'].transform(
            lambda x: x.rolling(20, min_periods=1).std()
        )
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        logger.info("Adding temporal features...")
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Basic time features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Cyclical encoding
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Market structure features
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        return df
    
    def _calculate_days_to_earnings(self, df: pd.DataFrame, earnings: pd.DataFrame) -> pd.Series:
        """Calculate days until next earnings for each stock-date combination"""
        days_to_earnings = []
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            date = pd.to_datetime(row['date'])
            
            # Find next earnings date for this symbol
            symbol_earnings = earnings[earnings['symbol'] == symbol]
            future_earnings = symbol_earnings[symbol_earnings['earnings_date'] > date]
            
            if len(future_earnings) > 0:
                next_earnings = future_earnings['earnings_date'].min()
                days_diff = (next_earnings - date).days
                days_to_earnings.append(min(days_diff, 999))  # Cap at 999
            else:
                days_to_earnings.append(999)  # No upcoming earnings
        
        return pd.Series(days_to_earnings, index=df.index)
    
    def _create_target_variable(self, df: pd.DataFrame, target_type: str, horizon: int) -> pd.DataFrame:
        """Create target variable based on specified type and horizon"""
        logger.info(f"Creating {target_type} target variable with {horizon}-day horizon...")
        
        if target_type == 'returns':
            # Future returns
            df['target'] = df.groupby('symbol')['adj_close'].pct_change(periods=horizon).shift(-horizon)
            
        elif target_type == 'classification':
            # Binary classification (up/down)
            returns = df.groupby('symbol')['adj_close'].pct_change(periods=horizon).shift(-horizon)
            df['target'] = (returns > 0).astype(int)
            
        elif target_type == 'quintile':
            # Cross-sectional quintile ranking
            returns = df.groupby('symbol')['adj_close'].pct_change(periods=horizon).shift(-horizon)
            df['daily_returns'] = returns
            
            # Create quintile rankings within each date
            df['target'] = df.groupby('date')['daily_returns'].transform(
                lambda x: pd.qcut(x.rank(method='first'), 5, labels=False)
            )
            
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        return df
    
    def _add_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time and group indices required by TFT"""
        logger.info("Adding indices...")
        
        # Sort by symbol and date
        df = df.sort_values(['symbol', 'date'])
        
        # Create time index (continuous integer for each symbol)
        df['time_idx'] = df.groupby('symbol').cumcount()
        
        # Create group_id (symbol as categorical)
        df['group_id'] = df['symbol']
        
        return df
    
    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset and handle missing values"""
        logger.info("Cleaning dataset...")
        
        # Fill numerical missing values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['target']:  # Don't fill target variable
                df[col] = df.groupby('symbol')[col].fillna(method='ffill').fillna(method='bfill')
        
        # Fill categorical missing values
        categorical_cols = ['sector', 'industry', 'exchange']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill any remaining NaN values in numerical columns (except target)
        for col in numerical_cols:
            if col not in ['target'] and df[col].isna().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)

        # Drop rows where target is NaN or infinite (e.g. last N rows from forward returns)
        before = len(df)
        df = df.dropna(subset=['target'])
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with NaN/inf target values")

        return df
    
    def _filter_sufficient_data(self, df: pd.DataFrame, min_observations: int = 100) -> pd.DataFrame:
        """Filter out symbols with insufficient data"""
        symbol_counts = df['symbol'].value_counts()
        valid_symbols = symbol_counts[symbol_counts >= min_observations].index
        
        original_symbols = df['symbol'].nunique()
        df_filtered = df[df['symbol'].isin(valid_symbols)]
        
        logger.info(f"Filtered symbols: {original_symbols} -> {df_filtered['symbol'].nunique()}")
        logger.info(f"Removed {original_symbols - df_filtered['symbol'].nunique()} symbols with < {min_observations} observations")
        
        return df_filtered
    
    def get_feature_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get categorized feature columns for TFT model configuration"""
        
        static_categoricals = ['symbol', 'sector', 'industry', 'exchange']
        
        static_reals = ['market_cap', 'pe_ratio', 'eps', 'dividend_yield']
        
        time_varying_known_reals = [
            'time_idx', 'day_sin', 'day_cos', 'week_sin', 'week_cos', 
            'month_sin', 'month_cos', 'is_monday', 'is_friday', 
            'is_month_end', 'is_quarter_end', 'earnings_flag', 'days_to_earnings'
        ]
        
        time_varying_unknown_reals = [
            'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_position', 'volume_ratio', 'obv',
            'price_change', 'high_low_ratio', 'close_open_ratio', 'returns_volatility',
            'sentiment_score', 'sentiment_magnitude', 'news_count'
        ]
        
        # Filter columns that actually exist in the dataframe
        return {
            'static_categoricals': [col for col in static_categoricals if col in df.columns],
            'static_reals': [col for col in static_reals if col in df.columns],
            'time_varying_known_reals': [col for col in time_varying_known_reals if col in df.columns],
            'time_varying_unknown_reals': [col for col in time_varying_unknown_reals if col in df.columns]
        }
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataset quality and completeness"""
        validation_results = {
            'total_records': len(df),
            'symbols': df['symbol'].nunique(),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d'),
                'days': (df['date'].max() - df['date'].min()).days
            },
            'missing_values': {},
            'target_distribution': {},
            'symbol_stats': {}
        }
        
        # Check missing values
        for col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > 0:
                validation_results['missing_values'][col] = f"{missing_pct:.2f}%"
        
        # Target distribution
        if 'target' in df.columns:
            validation_results['target_distribution'] = {
                'mean': df['target'].mean(),
                'std': df['target'].std(),
                'min': df['target'].min(),
                'max': df['target'].max(),
                'missing_pct': (df['target'].isna().sum() / len(df)) * 100
            }
        
        # Symbol-level statistics
        symbol_stats = df.groupby('symbol').agg({
            'date': ['count', 'min', 'max'],
            'target': ['mean', 'std'] if 'target' in df.columns else ['count']
        }).round(4)
        
        validation_results['symbol_stats'] = {
            'avg_observations_per_symbol': symbol_stats.iloc[:, 0].mean(),
            'min_observations_per_symbol': symbol_stats.iloc[:, 0].min(),
            'max_observations_per_symbol': symbol_stats.iloc[:, 0].max()
        }
        
        return validation_results


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Database configuration
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'database': os.getenv('POSTGRES_DB', 'stock_trading_analysis'),
        'user': os.getenv('POSTGRES_USER', 'trading_user'),
        'password': os.environ['POSTGRES_PASSWORD'],
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'schema': os.getenv('POSTGRES_SCHEMA', 'public')
    }
    
    try:
        # Initialize pipeline
        pipeline = PostgresDataPipeline(db_config)
        
        # Test symbols (use available symbols from database)
        loader = PostgresDataLoader(db_config)
        available_symbols = loader.get_available_symbols()
        
        if available_symbols:
            test_symbols = available_symbols[:5]  # Test with first 5 symbols
            
            # Build dataset
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years
            
            print(f"Building dataset for symbols: {test_symbols}")
            print(f"Date range: {start_date} to {end_date}")
            
            df = pipeline.build_dataset(
                symbols=test_symbols,
                start_date=start_date,
                end_date=end_date,
                target_type='returns',
                prediction_horizon=5
            )
            
            # Validate dataset
            validation = pipeline.validate_dataset(df)
            print("\nDataset Validation:")
            print(f"Total records: {validation['total_records']:,}")
            print(f"Symbols: {validation['symbols']}")
            print(f"Date range: {validation['date_range']['start']} to {validation['date_range']['end']}")
            print(f"Target mean: {validation['target_distribution']['mean']:.4f}")
            print(f"Target std: {validation['target_distribution']['std']:.4f}")
            
            # Get feature columns
            features = pipeline.get_feature_columns(df)
            print(f"\nFeature columns:")
            print(f"Static categoricals: {len(features['static_categoricals'])}")
            print(f"Static reals: {len(features['static_reals'])}")
            print(f"Time varying known: {len(features['time_varying_known_reals'])}")
            print(f"Time varying unknown: {len(features['time_varying_unknown_reals'])}")
            
        else:
            print("No symbols available in database")
            
    except Exception as e:
        print(f"Error testing PostgreSQL data pipeline: {e}")
        import traceback
        traceback.print_exc()
