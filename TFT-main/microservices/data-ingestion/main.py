"""
🚀 DATA INGESTION MICROSERVICE
=============================
Real-time data collection and publishing to Kafka
Handles Polygon.io market data and Reddit sentiment data
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import aiohttp
import redis.asyncio as redis
from kafka import KafkaProducer
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Kafka Topics
KAFKA_TOPICS = {
    "market_data": "market-data",
    "reddit_comments": "reddit-comments",
    "earnings_calendar": "earnings-calendar",
    "system_health": "system-health",
}


class DataIngestionService:
    def __init__(self):
        self.kafka_producer = None
        self.redis_client = None
        self.session = None

    async def initialize(self):
        """Initialize connections to external services"""
        try:
            # Initialize Kafka producer
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                retries=3,
                acks="all",
            )

            # Initialize Redis connection
            self.redis_client = redis.from_url(REDIS_URL)

            # Initialize HTTP session
            self.session = aiohttp.ClientSession()

            logger.info("Data ingestion service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise

    async def cleanup(self):
        """Cleanup connections"""
        if self.kafka_producer:
            self.kafka_producer.close()
        if self.redis_client:
            await self.redis_client.close()
        if self.session:
            await self.session.close()

    async def collect_market_data(self, tickers: List[str]) -> Dict:
        """
        Collect real-time market data from Polygon API
        """
        try:
            results = {}

            for ticker in tickers:
                # Check cache first
                cache_key = f"market_data:{ticker}"
                cached_data = await self.redis_client.get(cache_key)

                if cached_data:
                    results[ticker] = json.loads(cached_data)
                    continue

                # Fetch from Polygon API
                url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
                params = {"apikey": POLYGON_API_KEY}

                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        if data.get("results"):
                            market_data = {
                                "ticker": ticker,
                                "timestamp": datetime.utcnow().isoformat(),
                                "open": data["results"][0]["o"],
                                "high": data["results"][0]["h"],
                                "low": data["results"][0]["l"],
                                "close": data["results"][0]["c"],
                                "volume": data["results"][0]["v"],
                                "vwap": data["results"][0].get("vw"),
                                "transactions": data["results"][0].get("n"),
                            }

                            results[ticker] = market_data

                            # Cache for 1 minute
                            await self.redis_client.setex(
                                cache_key, 60, json.dumps(market_data)
                            )

                            # Publish to Kafka
                            self.kafka_producer.send(
                                KAFKA_TOPICS["market_data"],
                                key=ticker,
                                value=market_data,
                            )

                        await asyncio.sleep(0.1)  # Rate limiting

            self.kafka_producer.flush()
            logger.info(f"Collected market data for {len(results)} tickers")
            return results

        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            raise

    async def collect_reddit_data(self, subreddits: List[str] = None) -> Dict:
        """
        Collect Reddit comments from financial subreddits
        """
        if subreddits is None:
            subreddits = ["stocks", "investing", "SecurityAnalysis", "ValueInvesting"]

        try:
            # Get Reddit access token
            auth_url = "https://www.reddit.com/api/v1/access_token"
            auth_data = {
                "grant_type": "client_credentials",
                "username": "",
                "password": "",
            }
            auth_headers = {"User-Agent": "TFT-DataIngestion/1.0"}

            async with self.session.post(
                auth_url,
                data=auth_data,
                headers=auth_headers,
                auth=aiohttp.BasicAuth(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET),
            ) as auth_response:
                auth_result = await auth_response.json()
                access_token = auth_result.get("access_token")

            if not access_token:
                raise Exception("Failed to get Reddit access token")

            results = []
            headers = {
                "Authorization": f"Bearer {access_token}",
                "User-Agent": "TFT-DataIngestion/1.0",
            }

            for subreddit in subreddits:
                url = f"https://oauth.reddit.com/r/{subreddit}/new"
                params = {"limit": 100}

                async with self.session.get(
                    url, headers=headers, params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        for post in data.get("data", {}).get("children", []):
                            post_data = post["data"]

                            comment_data = {
                                "id": post_data["id"],
                                "subreddit": subreddit,
                                "title": post_data["title"],
                                "body": post_data.get("selftext", ""),
                                "score": post_data["score"],
                                "upvote_ratio": post_data["upvote_ratio"],
                                "num_comments": post_data["num_comments"],
                                "created_utc": post_data["created_utc"],
                                "author": post_data["author"],
                                "timestamp": datetime.utcnow().isoformat(),
                            }

                            results.append(comment_data)

                            # Publish to Kafka
                            self.kafka_producer.send(
                                KAFKA_TOPICS["reddit_comments"],
                                key=post_data["id"],
                                value=comment_data,
                            )

                await asyncio.sleep(1)  # Rate limiting

            self.kafka_producer.flush()
            logger.info(f"Collected {len(results)} Reddit posts")
            return {"posts": results, "count": len(results)}

        except Exception as e:
            logger.error(f"Error collecting Reddit data: {e}")
            raise

    async def collect_earnings_calendar(self, days_ahead: int = 30) -> Dict:
        """
        Collect upcoming earnings calendar from Polygon
        """
        try:
            end_date = (datetime.now() + timedelta(days=days_ahead)).strftime(
                "%Y-%m-%d"
            )

            url = "https://api.polygon.io/v3/reference/financials"
            params = {
                "apikey": POLYGON_API_KEY,
                "filing_date.gte": datetime.now().strftime("%Y-%m-%d"),
                "filing_date.lte": end_date,
                "limit": 1000,
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for result in data.get("results", []):
                        earnings_data = {
                            "ticker": result.get("tickers", [None])[0],
                            "filing_date": result.get("filing_date"),
                            "period_of_report_date": result.get(
                                "period_of_report_date"
                            ),
                            "timeframe": result.get("timeframe"),
                            "fiscal_period": result.get("fiscal_period"),
                            "fiscal_year": result.get("fiscal_year"),
                            "timestamp": datetime.utcnow().isoformat(),
                        }

                        if earnings_data["ticker"]:
                            results.append(earnings_data)

                            # Publish to Kafka
                            self.kafka_producer.send(
                                KAFKA_TOPICS["earnings_calendar"],
                                key=earnings_data["ticker"],
                                value=earnings_data,
                            )

                    self.kafka_producer.flush()
                    logger.info(f"Collected {len(results)} earnings events")
                    return {"events": results, "count": len(results)}

        except Exception as e:
            logger.error(f"Error collecting earnings calendar: {e}")
            raise

    async def health_check(self) -> Dict:
        """Service health check"""
        try:
            health_status = {
                "service": "data-ingestion",
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "kafka_connected": self.kafka_producer is not None,
                "redis_connected": (
                    await self.redis_client.ping() if self.redis_client else False
                ),
            }

            # Publish health status
            self.kafka_producer.send(
                KAFKA_TOPICS["system_health"], key="data-ingestion", value=health_status
            )

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "service": "data-ingestion",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }


# Global service instance
service = DataIngestionService()


# FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await service.initialize()
    yield
    # Shutdown
    await service.cleanup()


# FastAPI app
app = FastAPI(
    title="TFT Data Ingestion Service",
    description="Real-time data collection and publishing",
    version="1.0.0",
    lifespan=lifespan,
)


# Request models
class MarketDataRequest(BaseModel):
    tickers: List[str]


class RedditDataRequest(BaseModel):
    subreddits: Optional[List[str]] = None


# API Endpoints
@app.get("/health")
async def health_endpoint():
    """Service health status"""
    return await service.health_check()


@app.post("/collect/market-data")
async def collect_market_data_endpoint(request: MarketDataRequest):
    """Collect market data for specified tickers"""
    try:
        result = await service.collect_market_data(request.tickers)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collect/reddit-data")
async def collect_reddit_data_endpoint(request: RedditDataRequest):
    """Collect Reddit data from financial subreddits"""
    try:
        result = await service.collect_reddit_data(request.subreddits)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collect/earnings-calendar")
async def collect_earnings_calendar_endpoint(days_ahead: int = 30):
    """Collect upcoming earnings calendar"""
    try:
        result = await service.collect_earnings_calendar(days_ahead)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    # TODO: Implement Prometheus metrics
    return {"metrics": "TODO: Implement Prometheus metrics"}


# Background task for continuous data collection
async def continuous_data_collection():
    """Background task for continuous data collection"""
    tickers = ["NVDA", "TSLA", "AAPL", "GOOGL", "MSFT", "AMZN"]

    while True:
        try:
            # Collect market data every 5 minutes
            await service.collect_market_data(tickers)

            # Collect Reddit data every 10 minutes
            await service.collect_reddit_data()

            # Wait 5 minutes
            await asyncio.sleep(300)

        except Exception as e:
            logger.error(f"Error in continuous data collection: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error


@app.on_event("startup")
async def start_background_tasks():
    """Start background data collection"""
    asyncio.create_task(continuous_data_collection())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True, log_level="info")
