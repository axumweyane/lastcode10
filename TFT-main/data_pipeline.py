"""
Data Pipeline for Stock Market Data Collection and Processing
Enhanced with Polygon.io integration (replacing yfinance)
"""

import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """
    Comprehensive stock data collection system
    Handles OHLCV data, fundamental data, and sentiment data
    """

    def __init__(
        self, db_path: str = "data/stock_data.db", cache_dir: str = "data/cache"
    ):
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create database connection
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for storing stock data"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Create tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_prices (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_fundamentals (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    market_cap REAL,
                    pe_ratio REAL,
                    pb_ratio REAL,
                    dividend_yield REAL,
                    eps REAL,
                    revenue REAL,
                    debt_to_equity REAL,
                    roe REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    sentiment_score REAL,
                    news_count INTEGER,
                    reddit_mentions INTEGER,
                    twitter_mentions INTEGER,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date, source)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS economic_indicators (
                    id INTEGER PRIMARY KEY,
                    date TEXT NOT NULL,
                    indicator_name TEXT NOT NULL,
                    value REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, indicator_name)
                )
            """)

            # Create indexes
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON stock_prices(symbol, date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fundamentals_symbol_date ON stock_fundamentals(symbol, date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_date ON sentiment_data(symbol, date)"
            )

            conn.commit()

    def get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols from Wikipedia"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_df = tables[0]
            symbols = sp500_df["Symbol"].tolist()

            # Clean symbols (replace dots with dashes for Yahoo Finance)
            symbols = [symbol.replace(".", "-") for symbol in symbols]

            logger.info(f"Retrieved {len(symbols)} S&P 500 symbols")
            return symbols

        except Exception as e:
            logger.error(f"Error fetching S&P 500 symbols: {e}")
            # Fallback to a subset of major stocks
            return [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "TSLA",
                "META",
                "NVDA",
                "BRK-B",
                "UNH",
                "JNJ",
                "JPM",
                "V",
                "PG",
                "HD",
                "MA",
                "PFE",
                "BAC",
                "ABBV",
                "KO",
                "PEP",
                "AVGO",
                "TMO",
                "COST",
                "WMT",
                "DIS",
                "ABT",
                "CRM",
                "VZ",
                "ADBE",
                "NFLX",
                "NKE",
                "DHR",
                "XOM",
                "CVX",
                "LLY",
                "ACN",
            ]

    def fetch_stock_data(
        self,
        symbols: List[str],
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for multiple symbols"""

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(
            f"Fetching data for {len(symbols)} symbols from {start_date} to {end_date}"
        )

        all_data = []
        failed_symbols = []

        def fetch_single_symbol(symbol):
            try:
                # Check cache first
                cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}.csv"

                if cache_file.exists() and self._is_cache_valid(cache_file):
                    logger.info(f"Loading {symbol} from cache")
                    return pd.read_csv(cache_file)

                # Fetch from Polygon.io API
                polygon_api_key = os.getenv("POLYGON_API_KEY")
                if not polygon_api_key:
                    logger.error("POLYGON_API_KEY not found in environment variables")
                    return None

                # Convert dates to proper format for Polygon API
                start_polygon = datetime.strptime(start_date, "%Y-%m-%d").strftime(
                    "%Y-%m-%d"
                )
                end_polygon = datetime.strptime(end_date, "%Y-%m-%d").strftime(
                    "%Y-%m-%d"
                )

                # Polygon.io aggregates endpoint
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_polygon}/{end_polygon}"
                params = {"adjusted": "true", "sort": "asc", "apikey": polygon_api_key}

                response = requests.get(url, params=params)

                if response.status_code != 200:
                    logger.warning(
                        f"Failed to fetch data for {symbol}: {response.status_code}"
                    )
                    return None

                polygon_data = response.json()

                if "results" not in polygon_data or not polygon_data["results"]:
                    logger.warning(f"No data returned for {symbol}")
                    return None

                # Convert Polygon data to DataFrame
                records = []
                for bar in polygon_data["results"]:
                    records.append(
                        {
                            "date": datetime.fromtimestamp(bar["t"] / 1000).strftime(
                                "%Y-%m-%d"
                            ),
                            "symbol": symbol,
                            "open": bar["o"],
                            "high": bar["h"],
                            "low": bar["l"],
                            "close": bar["c"],
                            "volume": bar["v"],
                            "adj_close": bar[
                                "c"
                            ],  # Polygon already provides adjusted data
                            "adj_volume": bar["v"],
                        }
                    )

                data = pd.DataFrame(records)

                if data.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return None

                # Save to cache
                data.to_csv(cache_file, index=False)

                logger.info(f"Fetched {len(data)} records for {symbol}")
                return data

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return None

        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(fetch_single_symbol, symbol): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data is not None:
                        all_data.append(data)
                    else:
                        failed_symbols.append(symbol)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    failed_symbols.append(symbol)

        if failed_symbols:
            logger.warning(
                f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols[:10]}..."
            )

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined dataset shape: {combined_data.shape}")

            # Store in database
            self._store_price_data(combined_data)

            return combined_data
        else:
            logger.error("No data was successfully fetched")
            return pd.DataFrame()

    def fetch_fundamental_data(self, symbols: List[str]) -> pd.DataFrame:
        """Fetch fundamental data for symbols"""
        logger.info(f"Fetching fundamental data for {len(symbols)} symbols")

        fundamental_data = []

        def fetch_fundamentals(symbol):
            try:
                # Use Polygon.io for fundamental data
                polygon_api_key = os.getenv("POLYGON_API_KEY")
                if not polygon_api_key:
                    logger.error("POLYGON_API_KEY not found in environment variables")
                    return None

                # Polygon ticker details endpoint
                url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
                params = {"apikey": polygon_api_key}

                response = requests.get(url, params=params)

                if response.status_code != 200:
                    logger.warning(
                        f"Failed to fetch fundamentals for {symbol}: {response.status_code}"
                    )
                    return None

                ticker_data = response.json()

                if "results" not in ticker_data:
                    logger.warning(f"No fundamental data returned for {symbol}")
                    return None

                info = ticker_data["results"]

                # Extract key fundamental metrics (Polygon structure)
                fundamentals = {
                    "symbol": symbol,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "market_cap": info.get("market_cap"),
                    "pe_ratio": None,  # Not directly available in Polygon basic plan
                    "pb_ratio": None,  # Not directly available in Polygon basic plan
                    "dividend_yield": None,  # Not directly available in Polygon basic plan
                    "eps": None,  # Not directly available in Polygon basic plan
                    "revenue": None,  # Not directly available in Polygon basic plan
                    "debt_to_equity": None,  # Not directly available in Polygon basic plan
                    "roe": None,  # Not directly available in Polygon basic plan
                    "total_employees": info.get("total_employees"),
                    "weighted_shares_outstanding": info.get(
                        "weighted_shares_outstanding"
                    ),
                    "description": info.get("description", "")[
                        :500
                    ],  # Limit description length
                }

                return fundamentals

            except Exception as e:
                logger.error(f"Error fetching fundamentals for {symbol}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(fetch_fundamentals, symbol) for symbol in symbols[:50]
            ]  # Limit to avoid rate limits

            for future in as_completed(futures):
                result = future.result()
                if result:
                    fundamental_data.append(result)

        if fundamental_data:
            df = pd.DataFrame(fundamental_data)
            self._store_fundamental_data(df)
            return df
        else:
            return pd.DataFrame()

    def generate_synthetic_sentiment(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Generate synthetic sentiment data for testing"""
        logger.info("Generating synthetic sentiment data")

        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        sentiment_data = []

        np.random.seed(42)

        for symbol in symbols:
            for date in date_range:
                if date.weekday() < 5:  # Only weekdays
                    sentiment_data.append(
                        {
                            "symbol": symbol,
                            "date": date.strftime("%Y-%m-%d"),
                            "sentiment_score": np.random.normal(
                                0, 0.5
                            ),  # Neutral sentiment with some variance
                            "news_count": np.random.poisson(5),
                            "reddit_mentions": np.random.poisson(10),
                            "twitter_mentions": np.random.poisson(20),
                            "source": "synthetic",
                        }
                    )

        df = pd.DataFrame(sentiment_data)
        self._store_sentiment_data(df)
        return df

    def _is_cache_valid(self, cache_file: Path, max_age_hours: float = 24) -> bool:
        """Check if cache file is still valid"""
        if not cache_file.exists():
            return False

        file_age = time.time() - cache_file.stat().st_mtime
        return file_age < (max_age_hours * 3600)

    def _store_price_data(self, df: pd.DataFrame):
        """Store price data in database"""
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql("stock_prices", conn, if_exists="replace", index=False)
            logger.info(f"Stored {len(df)} price records in database")

    def _store_fundamental_data(self, df: pd.DataFrame):
        """Store fundamental data in database"""
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql("stock_fundamentals", conn, if_exists="replace", index=False)
            logger.info(f"Stored {len(df)} fundamental records in database")

    def _store_sentiment_data(self, df: pd.DataFrame):
        """Store sentiment data in database"""
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql("sentiment_data", conn, if_exists="replace", index=False)
            logger.info(f"Stored {len(df)} sentiment records in database")

    def load_data_from_db(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load data from database"""

        query = """
        SELECT p.*, f.market_cap, f.pe_ratio, f.pb_ratio, f.dividend_yield,
               s.sentiment_score, s.news_count
        FROM stock_prices p
        LEFT JOIN stock_fundamentals f ON p.symbol = f.symbol AND p.date = f.date
        LEFT JOIN sentiment_data s ON p.symbol = s.symbol AND p.date = s.date
        WHERE 1=1
        """

        params = []

        if symbols:
            placeholders = ",".join(["?" for _ in symbols])
            query += f" AND p.symbol IN ({placeholders})"
            params.extend(symbols)

        if start_date:
            query += " AND p.date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND p.date <= ?"
            params.append(end_date)

        query += " ORDER BY p.symbol, p.date"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        # Rename columns to match preprocessor expectations
        df = df.rename(columns={"date": "timestamp", "sentiment_score": "sentiment"})

        logger.info(f"Loaded {len(df)} records from database")
        return df


class DataPipeline:
    """
    Complete data pipeline for TFT model
    """

    def __init__(self, db_path: str = "data/stock_data.db"):
        self.collector = StockDataCollector(db_path)

    def run_daily_update(self, symbols: Optional[List[str]] = None):
        """Run daily data update"""
        logger.info("Starting daily data update...")

        if symbols is None:
            symbols = self.collector.get_sp500_symbols()

        # Get last 7 days of data to ensure we catch any updates
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        # Fetch price data
        price_data = self.collector.fetch_stock_data(
            symbols=symbols, start_date=start_date, end_date=end_date
        )

        # Generate synthetic sentiment (replace with real sentiment API)
        sentiment_data = self.collector.generate_synthetic_sentiment(
            symbols=symbols, start_date=start_date, end_date=end_date
        )

        logger.info("Daily data update completed")

        return len(price_data) if not price_data.empty else 0

    def run_historical_backfill(
        self, symbols: Optional[List[str]] = None, start_date: str = "2020-01-01"
    ):
        """Run historical data backfill"""
        logger.info("Starting historical data backfill...")

        if symbols is None:
            symbols = self.collector.get_sp500_symbols()

        end_date = datetime.now().strftime("%Y-%m-%d")

        # Fetch historical price data
        price_data = self.collector.fetch_stock_data(
            symbols=symbols, start_date=start_date, end_date=end_date
        )

        # Fetch fundamental data
        fundamental_data = self.collector.fetch_fundamental_data(symbols)

        # Generate historical sentiment data
        sentiment_data = self.collector.generate_synthetic_sentiment(
            symbols=symbols, start_date=start_date, end_date=end_date
        )

        logger.info("Historical backfill completed")

        return {
            "price_records": len(price_data) if not price_data.empty else 0,
            "fundamental_records": (
                len(fundamental_data) if not fundamental_data.empty else 0
            ),
            "sentiment_records": len(sentiment_data) if not sentiment_data.empty else 0,
        }


if __name__ == "__main__":
    # Test the data pipeline
    pipeline = DataPipeline()

    # Test with a small set of symbols
    test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

    print("Running historical backfill...")
    results = pipeline.run_historical_backfill(
        symbols=test_symbols, start_date="2023-01-01"
    )

    print(f"Backfill results: {results}")

    # Load data to verify
    df = pipeline.collector.load_data_from_db(
        symbols=test_symbols, start_date="2023-01-01"
    )

    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Symbols: {df['symbol'].unique()}")
