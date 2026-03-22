"""
Enhanced Data Collection Pipeline with Advanced API Integration
Integrates Polygon.io, Alpaca, Reddit, and sentiment analysis
"""

"""
# COPILOT PROMPT: Process Polygon news into trading features:
# 1. Calculate sentiment polarity score per article using TextBlob
# 2. Compute daily sentiment momentum (3-day rolling change)
# 3. Merge with OHLCV using Polygon's news timestamp alignment
# EXPECTED OUTPUT: Comprehensive news sentiment processing pipeline
# POLYGON INTEGRATION: News API, sentiment analysis, temporal alignment
"""


import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import sqlite3
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import praw
from textblob import TextBlob
import websocket
import json
import threading
from queue import Queue
import psycopg2
from psycopg2.extras import RealDictCursor

from config_manager import get_config

logger = logging.getLogger(__name__)


class PolygonWebSocketClient:
    """
    ENHANCED COPILOT PROMPT: Production-grade Polygon WebSocket client
    Your enhanced Copilot should generate:
    1. Connect to Polygon WebSocket with authentication
    2. Subscribe to real-time quotes for 500+ symbols
    3. Handle connection pooling and automatic failover
    4. Implement exponential backoff reconnection logic
    5. Validate data completeness and perform sanity checks
    6. Batch insert to PostgreSQL every 15 seconds
    7. Trigger TFT predictions on volume spikes (>3 sigma)
    8. Monitor market hours and pause processing after hours
    """

    def __init__(self, api_key: str, symbols: List[str], db_config: Dict):
        self.api_key = api_key
        self.symbols = symbols
        self.db_config = db_config
        self.ws = None
        self.data_queue = Queue()
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.base_reconnect_delay = 1  # seconds

    def setup_polygon_websocket(self):
        """
        ENHANCED COPILOT PROMPT: Initialize Polygon WebSocket connection
        Connect to wss://socket.polygon.io with authentication and symbol subscriptions
        """
        try:
            # Polygon WebSocket URL
            ws_url = "wss://socket.polygon.io/stocks"

            def on_message(ws, message):
                """Handle incoming WebSocket messages"""
                try:
                    data = json.loads(message)

                    # Handle different message types
                    if isinstance(data, list):
                        for item in data:
                            self._process_message(item)
                    else:
                        self._process_message(data)

                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")

            def on_error(ws, error):
                """Handle WebSocket errors"""
                logger.error(f"WebSocket error: {error}")
                self._schedule_reconnect()

            def on_close(ws, close_status_code, close_msg):
                """Handle WebSocket connection close"""
                logger.warning(
                    f"WebSocket connection closed: {close_status_code} - {close_msg}"
                )
                if self.is_running:
                    self._schedule_reconnect()

            def on_open(ws):
                """Handle WebSocket connection open"""
                logger.info("✅ Connected to Polygon WebSocket")
                self.reconnect_attempts = 0

                # Authenticate
                auth_message = {"action": "auth", "params": self.api_key}
                ws.send(json.dumps(auth_message))

                # Subscribe to symbols
                subscribe_message = {
                    "action": "subscribe",
                    "params": f"T.{',T.'.join(self.symbols)}",  # Trade data
                }
                ws.send(json.dumps(subscribe_message))

                # Also subscribe to quotes
                quote_message = {
                    "action": "subscribe",
                    "params": f"Q.{',Q.'.join(self.symbols)}",  # Quote data
                }
                ws.send(json.dumps(quote_message))

                logger.info(f"Subscribed to {len(self.symbols)} symbols")

            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
            )

            return self.ws

        except Exception as e:
            logger.error(f"Failed to setup Polygon WebSocket: {e}")
            raise

    def _process_message(self, message: Dict):
        """Process individual WebSocket message"""
        try:
            msg_type = message.get("ev")  # Event type

            if msg_type == "T":  # Trade data
                trade_data = {
                    "symbol": message.get("sym"),
                    "price": message.get("p"),
                    "size": message.get("s"),
                    "timestamp": message.get("t"),
                    "conditions": message.get("c", []),
                    "exchange": message.get("x"),
                    "type": "trade",
                }
                self.data_queue.put(trade_data)

                # Check for volume spike (simplified)
                if trade_data["size"] > 10000:  # Large trade
                    logger.info(
                        f"🚨 Large trade detected: {trade_data['symbol']} - {trade_data['size']} shares"
                    )

            elif msg_type == "Q":  # Quote data
                quote_data = {
                    "symbol": message.get("sym"),
                    "bid_price": message.get("bp"),
                    "bid_size": message.get("bs"),
                    "ask_price": message.get("ap"),
                    "ask_size": message.get("as"),
                    "timestamp": message.get("t"),
                    "exchange": message.get("x"),
                    "type": "quote",
                }
                self.data_queue.put(quote_data)

            elif msg_type == "status":
                status = message.get("status")
                logger.info(f"WebSocket status: {status}")

                if status == "auth_success":
                    logger.info("✅ WebSocket authentication successful")
                elif status == "auth_failed":
                    logger.error("❌ WebSocket authentication failed")

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _schedule_reconnect(self):
        """Schedule reconnection with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached. Stopping.")
            self.is_running = False
            return

        delay = self.base_reconnect_delay * (2**self.reconnect_attempts)
        delay = min(delay, 30)  # Cap at 30 seconds

        logger.info(
            f"Scheduling reconnect in {delay} seconds (attempt {self.reconnect_attempts + 1})"
        )

        def reconnect():
            time.sleep(delay)
            if self.is_running:
                self.reconnect_attempts += 1
                self.start()

        thread = threading.Thread(target=reconnect)
        thread.daemon = True
        thread.start()

    def _batch_insert_to_db(self):
        """Batch insert data to PostgreSQL every 15 seconds"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()

            trades_to_insert = []
            quotes_to_insert = []

            # Process queued data
            while not self.data_queue.empty():
                try:
                    data = self.data_queue.get_nowait()

                    if data["type"] == "trade":
                        trades_to_insert.append(
                            (
                                data["symbol"],
                                data["price"],
                                data["size"],
                                datetime.fromtimestamp(data["timestamp"] / 1000),
                                json.dumps(data["conditions"]),
                                data["exchange"],
                            )
                        )
                    elif data["type"] == "quote":
                        quotes_to_insert.append(
                            (
                                data["symbol"],
                                data["bid_price"],
                                data["bid_size"],
                                data["ask_price"],
                                data["ask_size"],
                                datetime.fromtimestamp(data["timestamp"] / 1000),
                                data["exchange"],
                            )
                        )

                except:
                    break

            # Insert trades
            if trades_to_insert:
                cursor.executemany(
                    """
                    INSERT INTO realtime_trades 
                    (symbol, price, size, timestamp, conditions, exchange)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """,
                    trades_to_insert,
                )

                logger.info(f"Inserted {len(trades_to_insert)} trades to database")

            # Insert quotes
            if quotes_to_insert:
                cursor.executemany(
                    """
                    INSERT INTO realtime_quotes
                    (symbol, bid_price, bid_size, ask_price, ask_size, timestamp, exchange)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """,
                    quotes_to_insert,
                )

                logger.info(f"Inserted {len(quotes_to_insert)} quotes to database")

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Database insert error: {e}")

    def start(self):
        """Start the WebSocket client"""
        try:
            self.is_running = True
            logger.info("🚀 Starting Polygon WebSocket client...")

            # Start database batch insert timer
            def db_timer():
                while self.is_running:
                    time.sleep(15)  # Every 15 seconds
                    if self.is_running:
                        self._batch_insert_to_db()

            db_thread = threading.Thread(target=db_timer)
            db_thread.daemon = True
            db_thread.start()

            # Setup and run WebSocket
            ws = self.setup_polygon_websocket()
            ws.run_forever()

        except Exception as e:
            logger.error(f"WebSocket client error: {e}")
            if self.is_running:
                self._schedule_reconnect()

    def stop(self):
        """Stop the WebSocket client"""
        logger.info("Stopping Polygon WebSocket client...")
        self.is_running = False
        if self.ws:
            self.ws.close()


logger = logging.getLogger(__name__)


class PolygonDataCollector:
    """
    Enhanced data collector using Polygon.io API
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        self.rate_limit_delay = 0.02  # 50 requests per second for paid plans

    def get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timespan: str = "day",
        multiplier: int = 1,
    ) -> pd.DataFrame:
        """
        Get OHLCV data from Polygon.io
        """
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"

        params = {"adjusted": "true", "sort": "asc", "apikey": self.api_key}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if data.get("status") == "OK" and "results" in data:
                results = data["results"]

                df = pd.DataFrame(results)
                df["date"] = pd.to_datetime(df["t"], unit="ms").dt.strftime("%Y-%m-%d")
                df["symbol"] = symbol

                # Rename columns to match our schema
                df = df.rename(
                    columns={
                        "o": "open",
                        "h": "high",
                        "l": "low",
                        "c": "close",
                        "v": "volume",
                        "t": "timestamp",
                    }
                )

                # Calculate additional metrics
                df["vwap"] = df.get("vw", df["close"])  # Volume weighted average price
                df["trade_count"] = df.get("n", 0)  # Number of transactions

                time.sleep(self.rate_limit_delay)
                return df[
                    [
                        "symbol",
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "vwap",
                        "trade_count",
                    ]
                ]

            else:
                logger.warning(
                    f"No data returned for {symbol}: {data.get('message', 'Unknown error')}"
                )
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_market_sentiment_indicators(self, symbol: str) -> Dict[str, Any]:
        """
        Get market sentiment indicators from Polygon
        """
        # Get recent news
        news_url = f"{self.base_url}/v2/reference/news"
        params = {
            "ticker": symbol,
            "published_utc.gte": (datetime.now() - timedelta(days=7)).strftime(
                "%Y-%m-%d"
            ),
            "limit": 50,
            "apikey": self.api_key,
        }

        try:
            response = self.session.get(news_url, params=params)
            response.raise_for_status()

            data = response.json()

            if data.get("status") == "OK" and "results" in data:
                news_items = data["results"]

                # Calculate sentiment from news headlines
                sentiments = []
                for item in news_items:
                    title = item.get("title", "")
                    description = item.get("description", "")
                    text = f"{title} {description}"

                    if text.strip():
                        blob = TextBlob(text)
                        sentiments.append(blob.sentiment.polarity)

                avg_sentiment = np.mean(sentiments) if sentiments else 0.0
                news_count = len(news_items)

                time.sleep(self.rate_limit_delay)

                return {
                    "sentiment_score": avg_sentiment,
                    "news_count": news_count,
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Error fetching sentiment for {symbol}: {e}")

        return {
            "sentiment_score": 0.0,
            "news_count": 0,
            "timestamp": datetime.now().isoformat(),
        }


class RedditSentimentCollector:
    """
    Reddit sentiment collector with advanced WSB analysis
    """

    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.reddit = praw.Reddit(
            client_id=client_id, client_secret=client_secret, user_agent=user_agent
        )
        self.wsb_terms = {
            "bullish": [
                "moon",
                "rocket",
                "diamond hands",
                "hodl",
                "stonks",
                "tendies",
                "calls",
            ],
            "bearish": ["puts", "bear", "crash", "dump", "paper hands", "sell"],
            "neutral": ["sideways", "flat", "theta gang"],
        }

    def get_subreddit_sentiment(
        self,
        subreddit_name: str,
        symbol: str,
        limit: int = 100,
        timeframe: str = "week",
    ) -> Dict[str, Any]:
        """
        Analyze sentiment for a specific symbol in a subreddit
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Get recent posts
            if timeframe == "day":
                posts = subreddit.top(time_filter="day", limit=limit)
            elif timeframe == "week":
                posts = subreddit.top(time_filter="week", limit=limit)
            else:
                posts = subreddit.hot(limit=limit)

            relevant_posts = []
            sentiments = []
            wsb_scores = []

            for post in posts:
                title = post.title.lower()
                selftext = post.selftext.lower()
                combined_text = f"{title} {selftext}"

                # Check if post mentions the symbol
                if (
                    symbol.lower() in combined_text
                    or f"${symbol.lower()}" in combined_text
                ):
                    relevant_posts.append(
                        {
                            "title": post.title,
                            "score": post.score,
                            "upvote_ratio": post.upvote_ratio,
                            "num_comments": post.num_comments,
                            "created_utc": post.created_utc,
                            "url": post.url,
                        }
                    )

                    # Basic sentiment analysis
                    blob = TextBlob(combined_text)
                    sentiments.append(blob.sentiment.polarity)

                    # WSB-specific sentiment scoring
                    wsb_score = self._calculate_wsb_sentiment(combined_text)
                    wsb_scores.append(wsb_score)

            if relevant_posts:
                avg_sentiment = np.mean(sentiments)
                avg_wsb_sentiment = np.mean(wsb_scores)
                total_upvotes = sum(post["score"] for post in relevant_posts)
                total_comments = sum(post["num_comments"] for post in relevant_posts)

                return {
                    "subreddit": subreddit_name,
                    "symbol": symbol,
                    "post_count": len(relevant_posts),
                    "avg_sentiment": avg_sentiment,
                    "wsb_sentiment": avg_wsb_sentiment,
                    "total_upvotes": total_upvotes,
                    "total_comments": total_comments,
                    "posts": relevant_posts,
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(
                f"Error collecting Reddit sentiment for {symbol} in {subreddit_name}: {e}"
            )

        return {
            "subreddit": subreddit_name,
            "symbol": symbol,
            "post_count": 0,
            "avg_sentiment": 0.0,
            "wsb_sentiment": 0.0,
            "total_upvotes": 0,
            "total_comments": 0,
            "posts": [],
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_wsb_sentiment(self, text: str) -> float:
        """
        Calculate WSB-specific sentiment score
        """
        text = text.lower()
        bullish_count = sum(1 for term in self.wsb_terms["bullish"] if term in text)
        bearish_count = sum(1 for term in self.wsb_terms["bearish"] if term in text)

        if bullish_count + bearish_count == 0:
            return 0.0

        return (bullish_count - bearish_count) / (bullish_count + bearish_count)


class VIXDataCollector:
    """
    VIX data collector for market regime analysis
    """

    def __init__(self):
        self.vix_symbol = "^VIX"

    def get_current_vix(self) -> float:
        """Get current VIX level using Polygon.io"""
        try:
            config = get_config()
            polygon_api_key = config.get("POLYGON_API_KEY")

            if not polygon_api_key:
                logger.warning("POLYGON_API_KEY not found, using default VIX value")
                return 20.0

            # Use today's date
            today = datetime.now().strftime("%Y-%m-%d")
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

            # Polygon.io aggregates endpoint for VIX
            url = f"https://api.polygon.io/v2/aggs/ticker/I:VIX/range/1/day/{yesterday}/{today}"
            params = {
                "adjusted": "true",
                "sort": "desc",
                "limit": 1,
                "apikey": polygon_api_key,
            }

            response = requests.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                if "results" in data and data["results"]:
                    return float(data["results"][0]["c"])  # Close price

        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")

        return 20.0  # Default VIX level

    def get_vix_history(self, days: int = 30) -> pd.DataFrame:
        """Get VIX historical data using Polygon.io"""
        try:
            config = get_config()
            polygon_api_key = config.get("POLYGON_API_KEY")

            if not polygon_api_key:
                logger.warning("POLYGON_API_KEY not found, returning empty DataFrame")
                return pd.DataFrame()

            # Calculate date range
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            # Polygon.io aggregates endpoint for VIX
            url = f"https://api.polygon.io/v2/aggs/ticker/I:VIX/range/1/day/{start_date}/{end_date}"
            params = {"adjusted": "true", "sort": "asc", "apikey": polygon_api_key}

            response = requests.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                if "results" in data and data["results"]:
                    records = []
                    for bar in data["results"]:
                        records.append(
                            {
                                "symbol": "VIX",
                                "date": datetime.fromtimestamp(
                                    bar["t"] / 1000
                                ).strftime("%Y-%m-%d"),
                                "open": bar["o"],
                                "high": bar["h"],
                                "low": bar["l"],
                                "close": bar["c"],
                                "volume": bar.get("v", 0),  # VIX might not have volume
                            }
                        )

                    return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"Error fetching VIX history: {e}")

        return pd.DataFrame()

    def calculate_vix_regime(self, current_vix: float, config) -> str:
        """Determine current VIX regime"""
        if current_vix <= config.trading_config.vix_low_regime_max:
            return "low_volatility"
        elif current_vix <= config.trading_config.vix_medium_regime_max:
            return "medium_volatility"
        else:
            return "high_volatility"


class EnhancedDataPipeline:
    """
    Enhanced data pipeline with advanced integrations
    """

    def __init__(self, db_path: str = None):
        self.config = get_config()
        self.db_path = db_path or self.config.data_config.tft_db_path

        # Initialize collectors
        self.polygon_collector = None
        if self.config.data_config.polygon_api_key:
            self.polygon_collector = PolygonDataCollector(
                self.config.data_config.polygon_api_key
            )

        self.reddit_collector = None
        if self.config.data_config.reddit_client_id:
            self.reddit_collector = RedditSentimentCollector(
                self.config.data_config.reddit_client_id,
                self.config.data_config.reddit_client_secret,
                "Kironix_TFT_v1.0",
            )

        self.vix_collector = VIXDataCollector()

        self._init_database()

    def _init_database(self):
        """Initialize enhanced database schema"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Enhanced stock prices table
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
                    vwap REAL,
                    trade_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            """)

            # Enhanced sentiment data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    source TEXT NOT NULL,
                    sentiment_score REAL,
                    wsb_sentiment REAL,
                    news_count INTEGER,
                    reddit_mentions INTEGER,
                    reddit_upvotes INTEGER,
                    reddit_comments INTEGER,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date, source)
                )
            """)

            # VIX data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vix_data (
                    id INTEGER PRIMARY KEY,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    regime TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)

            # Market regime table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_regimes (
                    id INTEGER PRIMARY KEY,
                    date TEXT NOT NULL,
                    vix_level REAL,
                    regime TEXT,
                    bullish_threshold REAL,
                    bearish_threshold REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)

            conn.commit()

    def collect_enhanced_stock_data(
        self, symbols: List[str], start_date: str, end_date: str = None
    ) -> pd.DataFrame:
        """
        Collect enhanced stock data with multiple sources
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Collecting enhanced data for {len(symbols)} symbols")

        all_data = []

        # Collect price data
        if self.polygon_collector:
            logger.info("Using Polygon.io for price data")
            for symbol in symbols:
                try:
                    df = self.polygon_collector.get_stock_data(
                        symbol, start_date, end_date
                    )
                    if not df.empty:
                        all_data.append(df)
                        logger.info(f"Collected {len(df)} records for {symbol}")
                except Exception as e:
                    logger.error(f"Error collecting Polygon data for {symbol}: {e}")
        else:
            # Fallback to Yahoo Finance
            logger.info("Using Yahoo Finance for price data")
            from data_pipeline import StockDataCollector

            fallback_collector = StockDataCollector(self.db_path)
            df = fallback_collector.fetch_stock_data(symbols, start_date, end_date)
            all_data.append(df)

        # Combine price data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self._store_enhanced_price_data(combined_df)
            return combined_df

        return pd.DataFrame()

    def collect_enhanced_sentiment_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        Collect enhanced sentiment data from multiple sources
        """
        logger.info(f"Collecting sentiment data for {len(symbols)} symbols")

        sentiment_data = []

        # Collect Reddit sentiment
        if self.reddit_collector:
            for symbol in symbols:
                for subreddit in self.config.sentiment_config.monitor_subreddits:
                    try:
                        sentiment = self.reddit_collector.get_subreddit_sentiment(
                            subreddit,
                            symbol,
                            limit=self.config.sentiment_config.posts_per_subreddit,
                        )

                        if sentiment["post_count"] > 0:
                            sentiment_data.append(
                                {
                                    "symbol": symbol,
                                    "date": datetime.now().strftime("%Y-%m-%d"),
                                    "source": f"reddit_{subreddit}",
                                    "sentiment_score": sentiment["avg_sentiment"],
                                    "wsb_sentiment": sentiment["wsb_sentiment"],
                                    "news_count": 0,
                                    "reddit_mentions": sentiment["post_count"],
                                    "reddit_upvotes": sentiment["total_upvotes"],
                                    "reddit_comments": sentiment["total_comments"],
                                    "confidence_score": min(
                                        sentiment["post_count"] / 10.0, 1.0
                                    ),
                                }
                            )

                        time.sleep(self.config.sentiment_config.rate_limit_seconds)

                    except Exception as e:
                        logger.error(
                            f"Error collecting Reddit sentiment for {symbol} in {subreddit}: {e}"
                        )

        # Collect news sentiment (Polygon)
        if self.polygon_collector:
            for symbol in symbols:
                try:
                    sentiment = self.polygon_collector.get_market_sentiment_indicators(
                        symbol
                    )

                    sentiment_data.append(
                        {
                            "symbol": symbol,
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "source": "polygon_news",
                            "sentiment_score": sentiment["sentiment_score"],
                            "wsb_sentiment": 0.0,
                            "news_count": sentiment["news_count"],
                            "reddit_mentions": 0,
                            "reddit_upvotes": 0,
                            "reddit_comments": 0,
                            "confidence_score": min(
                                sentiment["news_count"] / 20.0, 1.0
                            ),
                        }
                    )

                except Exception as e:
                    logger.error(f"Error collecting news sentiment for {symbol}: {e}")

        if sentiment_data:
            df = pd.DataFrame(sentiment_data)
            self._store_enhanced_sentiment_data(df)
            return df

        return pd.DataFrame()

    def collect_vix_data(self) -> Dict[str, Any]:
        """
        Collect and analyze VIX data for market regime detection
        """
        logger.info("Collecting VIX data for regime analysis")

        try:
            # Get current VIX
            current_vix = self.vix_collector.get_current_vix()

            # Get VIX history
            vix_history = self.vix_collector.get_vix_history(30)

            # Determine regime
            regime = self.vix_collector.calculate_vix_regime(current_vix, self.config)

            # Get dynamic thresholds
            thresholds = self.config.get_vix_regime_thresholds(current_vix)

            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                # Store VIX history
                if not vix_history.empty:
                    vix_history["regime"] = regime
                    vix_history.to_sql(
                        "vix_data", conn, if_exists="replace", index=False
                    )

                # Store current regime
                regime_data = pd.DataFrame(
                    [
                        {
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "vix_level": current_vix,
                            "regime": regime,
                            "bullish_threshold": thresholds["bullish_threshold"],
                            "bearish_threshold": thresholds["bearish_threshold"],
                        }
                    ]
                )
                regime_data.to_sql(
                    "market_regimes", conn, if_exists="replace", index=False
                )

            return {
                "current_vix": current_vix,
                "regime": regime,
                "thresholds": thresholds,
                "history_records": len(vix_history),
            }

        except Exception as e:
            logger.error(f"Error collecting VIX data: {e}")
            return {
                "current_vix": 20.0,
                "regime": "medium_volatility",
                "thresholds": {},
            }

    def _store_enhanced_price_data(self, df: pd.DataFrame):
        """Store enhanced price data"""
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql("stock_prices", conn, if_exists="replace", index=False)
            logger.info(f"Stored {len(df)} enhanced price records")

    def _store_enhanced_sentiment_data(self, df: pd.DataFrame):
        """Store enhanced sentiment data"""
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql("sentiment_data", conn, if_exists="replace", index=False)
            logger.info(f"Stored {len(df)} enhanced sentiment records")

    def load_enhanced_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load enhanced dataset with sentiment and VIX data"""

        query = """
        SELECT 
            p.*,
            AVG(s.sentiment_score) as avg_sentiment,
            AVG(s.wsb_sentiment) as wsb_sentiment,
            SUM(s.reddit_mentions) as total_mentions,
            SUM(s.reddit_upvotes) as total_upvotes,
            AVG(s.confidence_score) as sentiment_confidence,
            v.close as vix_close,
            v.regime as market_regime,
            mr.bullish_threshold,
            mr.bearish_threshold
        FROM stock_prices p
        LEFT JOIN sentiment_data s ON p.symbol = s.symbol AND p.date = s.date
        LEFT JOIN vix_data v ON p.date = v.date
        LEFT JOIN market_regimes mr ON p.date = mr.date
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

        query += " GROUP BY p.symbol, p.date ORDER BY p.symbol, p.date"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        # Rename for TFT compatibility
        df = df.rename(columns={"date": "timestamp"})

        # Fill missing sentiment values
        df["avg_sentiment"] = df["avg_sentiment"].fillna(0.0)
        df["wsb_sentiment"] = df["wsb_sentiment"].fillna(0.0)

        logger.info(f"Loaded {len(df)} enhanced records")
        return df

    def run_full_collection(
        self, symbols: List[str] = None, start_date: str = "2023-01-01"
    ) -> Dict[str, int]:
        """
        Run full enhanced data collection pipeline
        """
        if symbols is None:
            # Use default symbols or S&P 500
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]

        logger.info(
            f"Starting full enhanced data collection for {len(symbols)} symbols"
        )

        results = {}

        # Collect price data
        price_df = self.collect_enhanced_stock_data(symbols, start_date)
        results["price_records"] = len(price_df)

        # Collect sentiment data
        sentiment_df = self.collect_enhanced_sentiment_data(symbols)
        results["sentiment_records"] = len(sentiment_df)

        # Collect VIX data
        vix_data = self.collect_vix_data()
        results["vix_records"] = vix_data.get("history_records", 0)
        results["current_vix"] = vix_data.get("current_vix", 0)
        results["market_regime"] = vix_data.get("regime", "unknown")

        logger.info(f"Enhanced data collection completed: {results}")
        return results


if __name__ == "__main__":
    # Test enhanced data pipeline
    pipeline = EnhancedDataPipeline()

    test_symbols = ["AAPL", "TSLA", "GME"]  # Include a meme stock for WSB sentiment

    print("🚀 Testing Enhanced Data Pipeline")
    print("=" * 50)

    # Run collection
    results = pipeline.run_full_collection(test_symbols, "2024-01-01")

    print("Collection Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")

    # Load and display sample data
    df = pipeline.load_enhanced_data(test_symbols, "2024-01-01")

    print(f"\nEnhanced Dataset:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    if not df.empty:
        print(f"\nSample sentiment data:")
        sentiment_cols = [
            "avg_sentiment",
            "wsb_sentiment",
            "total_mentions",
            "market_regime",
        ]
        available_cols = [col for col in sentiment_cols if col in df.columns]
        if available_cols:
            print(df[["symbol", "timestamp"] + available_cols].head())

    print("\n✅ Enhanced data pipeline test completed!")
