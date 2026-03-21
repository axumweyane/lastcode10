"""
PostgreSQL Data Connector for TFT Stock Prediction System
Handles loading OHLCV, fundamentals, sentiment, and earnings data from PostgreSQL
"""

import psycopg2
from psycopg2 import sql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Optional, Dict, Any
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PostgresDataLoader:
    """PostgreSQL data loader for stock market data"""
    
    def __init__(self, db_config: Dict[str, Any]):
        """
        Initialize PostgreSQL connection
        
        Args:
            db_config: Database configuration dictionary
        """
        self.db_config = db_config
        self.schema = db_config.get('schema', 'public')
        self._test_connection()
        
    def _test_connection(self):
        """Test database connection"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
            logger.info("PostgreSQL connection successful")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.db_config['host'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                port=self.db_config['port']
            )
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def load_ohlcv(self, symbols: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load OHLCV data from PostgreSQL
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        query = sql.SQL("""
            SELECT 
                symbol, date, open, high, low, close, volume,
                adj_open, adj_high, adj_low, adj_close, adj_volume
            FROM {schema}.ohlcv
            WHERE symbol = ANY(%s)
            AND date BETWEEN %s AND %s
            ORDER BY symbol, date
        """).format(schema=sql.Identifier(self.schema))
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (symbols, start_date, end_date))
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
        df = pd.DataFrame(data, columns=columns)
        logger.info(f"Loaded {len(df)} OHLCV records for {len(symbols)} symbols")
        return df
    
    def load_fundamentals(self, symbols: List[str], as_of_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load fundamental data
        
        Args:
            symbols: List of stock symbols
            as_of_date: As of date for fundamentals, defaults to today
            
        Returns:
            DataFrame with fundamental data
        """
        if as_of_date is None:
            as_of_date = datetime.now().strftime('%Y-%m-%d')
            
        query = sql.SQL("""
            WITH latest_fundamentals AS (
                SELECT 
                    symbol,
                    date,
                    market_cap, pe_ratio, eps, dividend_yield,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn
                FROM {schema}.fundamentals
                WHERE symbol = ANY(%s)
                AND date <= %s
            )
            SELECT 
                f.symbol, f.date, 
                f.market_cap, f.pe_ratio, f.eps, f.dividend_yield,
                COALESCE(s.sector, 'Unknown') as sector,
                COALESCE(s.industry, 'Unknown') as industry,
                COALESCE(s.exchange, 'Unknown') as exchange
            FROM latest_fundamentals f
            LEFT JOIN {schema}.symbols s ON f.symbol = s.symbol
            WHERE f.rn = 1
            ORDER BY f.symbol
        """).format(schema=sql.Identifier(self.schema))
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (symbols, as_of_date))
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
        df = pd.DataFrame(data, columns=columns)
        logger.info(f"Loaded fundamentals for {len(df)} symbols")
        return df
    
    def load_sentiment(self, symbols: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load sentiment data
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            DataFrame with sentiment data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        query = sql.SQL("""
            SELECT 
                symbol, date, 
                COALESCE(sentiment_score, 0) as sentiment_score,
                COALESCE(sentiment_magnitude, 0) as sentiment_magnitude,
                COALESCE(news_count, 0) as news_count
            FROM {schema}.sentiment
            WHERE symbol = ANY(%s)
            AND date BETWEEN %s AND %s
            ORDER BY symbol, date
        """).format(schema=sql.Identifier(self.schema))
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (symbols, start_date, end_date))
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
        df = pd.DataFrame(data, columns=columns)
        logger.info(f"Loaded {len(df)} sentiment records")
        return df
    
    def load_earnings_calendar(self, symbols: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load future earnings dates
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to 90 days from now
            
        Returns:
            DataFrame with earnings calendar data
        """
        if end_date is None:
            end_date = (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')
            
        query = sql.SQL("""
            SELECT 
                symbol, earnings_date, 
                earnings_estimate, earnings_actual
            FROM {schema}.earnings
            WHERE symbol = ANY(%s)
            AND earnings_date BETWEEN %s AND %s
            ORDER BY symbol, earnings_date
        """).format(schema=sql.Identifier(self.schema))
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (symbols, start_date, end_date))
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                
        df = pd.DataFrame(data, columns=columns)
        logger.info(f"Loaded {len(df)} earnings calendar records")
        return df
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols in the database"""
        query = sql.SQL("""
            SELECT DISTINCT symbol 
            FROM {schema}.ohlcv 
            ORDER BY symbol
        """).format(schema=sql.Identifier(self.schema))
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                symbols = [row[0] for row in cursor.fetchall()]
                
        logger.info(f"Found {len(symbols)} available symbols")
        return symbols
    
    def get_date_range(self, symbol: str) -> Dict[str, str]:
        """Get available date range for a symbol"""
        query = sql.SQL("""
            SELECT 
                MIN(date) as start_date,
                MAX(date) as end_date,
                COUNT(*) as record_count
            FROM {schema}.ohlcv
            WHERE symbol = %s
        """).format(schema=sql.Identifier(self.schema))
        
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (symbol,))
                result = cursor.fetchone()
                
        return {
            'start_date': result[0].strftime('%Y-%m-%d') if result[0] else None,
            'end_date': result[1].strftime('%Y-%m-%d') if result[1] else None,
            'record_count': result[2] if result[2] else 0
        }
    
    def validate_data_quality(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Validate data quality for given symbols and date range
        
        Returns:
            Dictionary with data quality metrics
        """
        results = {}
        
        for symbol in symbols:
            query = sql.SQL("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN adj_close IS NULL THEN 1 END) as missing_prices,
                    COUNT(CASE WHEN volume = 0 THEN 1 END) as zero_volume_days,
                    MIN(date) as actual_start,
                    MAX(date) as actual_end
                FROM {schema}.ohlcv
                WHERE symbol = %s
                AND date BETWEEN %s AND %s
            """).format(schema=sql.Identifier(self.schema))
            
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (symbol, start_date, end_date))
                    result = cursor.fetchone()
                    
            results[symbol] = {
                'total_records': result[0],
                'missing_prices': result[1],
                'zero_volume_days': result[2],
                'actual_start': result[3].strftime('%Y-%m-%d') if result[3] else None,
                'actual_end': result[4].strftime('%Y-%m-%d') if result[4] else None,
                'data_completeness': 1 - (result[1] / max(result[0], 1))
            }
            
        return results


class TechnicalIndicators:
    """Technical indicator calculations for PostgreSQL data"""
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame, price_col: str = 'adj_close') -> pd.DataFrame:
        """Add moving averages"""
        grouped = df.groupby('symbol')[price_col]
        
        df['sma_5'] = grouped.transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['sma_10'] = grouped.transform(lambda x: x.rolling(10, min_periods=1).mean())
        df['sma_20'] = grouped.transform(lambda x: x.rolling(20, min_periods=1).mean())
        df['sma_50'] = grouped.transform(lambda x: x.rolling(50, min_periods=1).mean())
        
        df['ema_12'] = grouped.transform(lambda x: x.ewm(span=12, min_periods=1).mean())
        df['ema_26'] = grouped.transform(lambda x: x.ewm(span=26, min_periods=1).mean())
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, price_col: str = 'adj_close', period: int = 14) -> pd.DataFrame:
        """Add RSI indicator"""
        def calculate_rsi(prices):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi'] = df.groupby('symbol')[price_col].transform(calculate_rsi)
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, price_col: str = 'adj_close') -> pd.DataFrame:
        """Add MACD indicator"""
        grouped = df.groupby('symbol')[price_col]
        
        ema_12 = grouped.transform(lambda x: x.ewm(span=12, min_periods=1).mean())
        ema_26 = grouped.transform(lambda x: x.ewm(span=26, min_periods=1).mean())
        
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df.groupby('symbol')['macd'].transform(
            lambda x: x.ewm(span=9, min_periods=1).mean()
        )
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, price_col: str = 'adj_close', period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Add Bollinger Bands"""
        grouped = df.groupby('symbol')[price_col]
        
        df['bb_middle'] = grouped.transform(lambda x: x.rolling(period, min_periods=1).mean())
        df['bb_std'] = grouped.transform(lambda x: x.rolling(period, min_periods=1).std())
        
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
        
        df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        grouped_volume = df.groupby('symbol')['adj_volume']
        grouped_price = df.groupby('symbol')['adj_close']
        
        # Volume moving averages
        df['volume_sma_20'] = grouped_volume.transform(lambda x: x.rolling(20, min_periods=1).mean())
        df['volume_ratio'] = df['adj_volume'] / df['volume_sma_20']
        
        # On Balance Volume (OBV)
        df['price_change'] = grouped_price.transform(lambda x: x.diff())
        df['obv'] = df.groupby('symbol').apply(
            lambda x: x['adj_volume'].where(x['price_change'] > 0, 
                                          -x['adj_volume'].where(x['price_change'] < 0, 0)).cumsum()
        ).reset_index(level=0, drop=True)
        
        return df


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
        # Test data loader
        loader = PostgresDataLoader(db_config)
        
        # Get available symbols
        symbols = loader.get_available_symbols()[:5]  # Test with first 5 symbols
        print(f"Testing with symbols: {symbols}")
        
        if symbols:
            # Test data loading
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            ohlcv = loader.load_ohlcv(symbols, start_date, end_date)
            print(f"Loaded OHLCV data: {ohlcv.shape}")
            
            fundamentals = loader.load_fundamentals(symbols)
            print(f"Loaded fundamentals: {fundamentals.shape}")
            
            sentiment = loader.load_sentiment(symbols, start_date, end_date)
            print(f"Loaded sentiment data: {sentiment.shape}")
            
            # Validate data quality
            quality = loader.validate_data_quality(symbols, start_date, end_date)
            print(f"Data quality validation completed for {len(quality)} symbols")
            
    except Exception as e:
        print(f"Error testing PostgreSQL data loader: {e}")
