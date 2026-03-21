"""
🚀 SENTIMENT ENGINE MICROSERVICE
===============================
Real-time sentiment analysis with momentum detection
Processes Reddit comments and generates sentiment scores
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import redis.asyncio as redis
from kafka import KafkaConsumer, KafkaProducer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MODEL_NAME = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")

# Kafka Topics
KAFKA_TOPICS = {
    "input": "reddit-comments",
    "output": "sentiment-scores",
    "system_health": "system-health"
}

class SentimentEngine:
    def __init__(self):
        self.kafka_consumer = None
        self.kafka_producer = None
        self.redis_client = None
        self.sentiment_pipeline = None
        self.tokenizer = None
        self.model = None
        
        # Sentiment momentum tracking
        self.sentiment_history = defaultdict(lambda: deque(maxlen=1000))
        self.momentum_window = 100  # Number of comments for momentum calculation
        
    async def initialize(self):
        """Initialize sentiment model and connections"""
        try:
            # Initialize GPU/CPU device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load sentiment model
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            self.model.to(self.device)
            
            # Create sentiment pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            # Initialize Kafka
            self.kafka_consumer = KafkaConsumer(
                KAFKA_TOPICS["input"],
                bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                group_id="sentiment-engine",
                auto_offset_reset="latest",
                enable_auto_commit=False,
            )
            
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retries=3,
                acks='all'
            )
            
            # Initialize Redis
            self.redis_client = redis.from_url(REDIS_URL)
            
            logger.info("Sentiment engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize sentiment engine: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.kafka_consumer:
            self.kafka_consumer.close()
        if self.kafka_producer:
            self.kafka_producer.close()
        if self.redis_client:
            await self.redis_client.close()
    
    def extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text"""
        # Common ticker pattern: $SYMBOL or SYMBOL mentioned
        ticker_pattern = r'\$([A-Z]{1,5})\b|(?:^|\s)([A-Z]{2,5})(?:\s|$)'
        matches = re.findall(ticker_pattern, text.upper())
        
        tickers = []
        for match in matches:
            ticker = match[0] or match[1]
            if ticker and len(ticker) >= 2:
                tickers.append(ticker)
        
        return list(set(tickers))  # Remove duplicates
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of single text"""
        try:
            # Clean text
            cleaned_text = re.sub(r'http\S+', '', text)  # Remove URLs
            cleaned_text = re.sub(r'@\w+', '', cleaned_text)  # Remove mentions
            cleaned_text = cleaned_text.strip()
            
            if not cleaned_text:
                return {"label": "NEUTRAL", "score": 0.0, "confidence": 0.0}
            
            # Truncate text to model max length
            if len(cleaned_text) > 512:
                cleaned_text = cleaned_text[:512]
            
            # Get sentiment prediction
            results = self.sentiment_pipeline(cleaned_text)
            
            # Process results - assuming RoBERTa sentiment labels
            sentiment_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
            
            best_result = max(results[0], key=lambda x: x['score'])
            label = sentiment_map.get(best_result['label'], best_result['label'])
            
            # Convert to numerical score: negative=-1, neutral=0, positive=1
            score_map = {"NEGATIVE": -1.0, "NEUTRAL": 0.0, "POSITIVE": 1.0}
            sentiment_score = score_map.get(label, 0.0) * best_result['score']
            
            return {
                "label": label,
                "score": sentiment_score,
                "confidence": best_result['score'],
                "raw_results": results[0]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"label": "NEUTRAL", "score": 0.0, "confidence": 0.0, "error": str(e)}
    
    def calculate_sentiment_percentages(self, sentiments: List[float]) -> Dict:
        """Calculate bullish/bearish/neutral percentages"""
        if not sentiments:
            return {"bullish_pct": 0.0, "bearish_pct": 0.0, "neutral_pct": 0.0}
        
        sentiments = np.array(sentiments)
        total = len(sentiments)
        
        # Using thresholds from the original prompt
        bullish_count = np.sum(sentiments > 0.3)
        bearish_count = np.sum(sentiments < -0.3)
        neutral_count = total - bullish_count - bearish_count
        
        return {
            "bullish_pct": round((bullish_count / total) * 100, 1),
            "bearish_pct": round((bearish_count / total) * 100, 1),
            "neutral_pct": round((neutral_count / total) * 100, 1)
        }
    
    def calculate_sentiment_momentum(self, ticker: str) -> Dict:
        """Calculate sentiment momentum for a ticker"""
        try:
            history = list(self.sentiment_history[ticker])
            
            if len(history) < 10:
                return {
                    "sentiment_momentum": 0.0,
                    "bullish_acceleration": 0.0,
                    "abnormal_spike": False,
                    "data_points": len(history)
                }
            
            # Extract sentiment scores and timestamps
            scores = [item['sentiment_score'] for item in history]
            timestamps = [item['timestamp'] for item in history]
            
            # Calculate momentum (recent vs older sentiment)
            recent_window = min(20, len(scores) // 2)
            recent_avg = np.mean(scores[-recent_window:])
            older_avg = np.mean(scores[:-recent_window]) if len(scores) > recent_window else 0
            
            sentiment_momentum = recent_avg - older_avg
            
            # Calculate acceleration (change in momentum)
            if len(scores) >= 40:
                mid_point = len(scores) // 2
                early_momentum = np.mean(scores[:mid_point]) - np.mean(scores[:mid_point//2]) if mid_point > 10 else 0
                bullish_acceleration = sentiment_momentum - early_momentum
            else:
                bullish_acceleration = 0.0
            
            # Detect abnormal spikes (3 sigma rule)
            if len(scores) >= 30:
                score_std = np.std(scores)
                score_mean = np.mean(scores)
                recent_score = scores[-1]
                abnormal_spike = abs(recent_score - score_mean) > 3 * score_std
            else:
                abnormal_spike = False
            
            return {
                "sentiment_momentum": round(sentiment_momentum, 4),
                "bullish_acceleration": round(bullish_acceleration, 4), 
                "abnormal_spike": abnormal_spike,
                "data_points": len(history),
                "recent_avg": round(recent_avg, 4),
                "older_avg": round(older_avg, 4)
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum for {ticker}: {e}")
            return {
                "sentiment_momentum": 0.0,
                "bullish_acceleration": 0.0,
                "abnormal_spike": False,
                "error": str(e)
            }
    
    async def process_comment(self, comment_data: Dict) -> Dict:
        """Process single comment and generate sentiment data"""
        try:
            # Extract text content
            text = f"{comment_data.get('title', '')} {comment_data.get('body', '')}"
            
            # Analyze sentiment
            sentiment_result = self.analyze_sentiment(text)
            
            # Extract mentioned tickers
            tickers = self.extract_tickers(text)
            
            # Create sentiment data
            result = {
                "comment_id": comment_data.get("id"),
                "subreddit": comment_data.get("subreddit"),
                "text": text[:200] + "..." if len(text) > 200 else text,
                "tickers": tickers,
                "sentiment_label": sentiment_result["label"],
                "sentiment_score": sentiment_result["score"],
                "confidence": sentiment_result["confidence"],
                "score": comment_data.get("score", 0),
                "timestamp": datetime.utcnow().isoformat(),
                "processed_at": datetime.utcnow().isoformat()
            }
            
            # Update sentiment history for each ticker
            for ticker in tickers:
                sentiment_entry = {
                    "sentiment_score": sentiment_result["score"],
                    "confidence": sentiment_result["confidence"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "comment_score": comment_data.get("score", 0)
                }
                self.sentiment_history[ticker].append(sentiment_entry)
                
                # Calculate momentum for this ticker
                momentum = self.calculate_sentiment_momentum(ticker)
                
                # Create ticker-specific sentiment data
                ticker_sentiment = {
                    "ticker": ticker,
                    "sentiment_score": sentiment_result["score"],
                    "confidence": sentiment_result["confidence"],
                    "momentum": momentum,
                    "comment_id": comment_data.get("id"),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Publish ticker sentiment to Kafka
                self.kafka_producer.send(
                    KAFKA_TOPICS["output"],
                    key=f"{ticker}_{comment_data.get('id')}",
                    value=ticker_sentiment
                )
                
                # Cache in Redis
                cache_key = f"sentiment:{ticker}:latest"
                await self.redis_client.setex(
                    cache_key, 3600, json.dumps(ticker_sentiment)
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing comment: {e}")
            return {"error": str(e), "comment_id": comment_data.get("id")}
    
    async def process_batch(self, comments: List[Dict]) -> Dict:
        """Process batch of comments"""
        try:
            results = []
            ticker_sentiments = defaultdict(list)
            
            for comment in comments:
                result = await self.process_comment(comment)
                results.append(result)
                
                # Aggregate by ticker
                for ticker in result.get("tickers", []):
                    if "sentiment_score" in result:
                        ticker_sentiments[ticker].append(result["sentiment_score"])
            
            # Calculate overall sentiment percentages by ticker
            ticker_summaries = {}
            for ticker, scores in ticker_sentiments.items():
                percentages = self.calculate_sentiment_percentages(scores)
                momentum = self.calculate_sentiment_momentum(ticker)
                
                ticker_summaries[ticker] = {
                    "ticker": ticker,
                    "comment_count": len(scores),
                    "avg_sentiment": round(np.mean(scores), 4),
                    **percentages,
                    "momentum": momentum,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            return {
                "processed_count": len(results),
                "ticker_summaries": ticker_summaries,
                "batch_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise
    
    async def health_check(self) -> Dict:
        """Service health check"""
        try:
            health_status = {
                "service": "sentiment-engine",
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "model_loaded": self.sentiment_pipeline is not None,
                "device": str(self.device),
                "kafka_connected": self.kafka_producer is not None,
                "redis_connected": await self.redis_client.ping() if self.redis_client else False,
                "tracked_tickers": len(self.sentiment_history)
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "service": "sentiment-engine", 
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# Global service instance
service = SentimentEngine()

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
    title="TFT Sentiment Engine Service",
    description="Real-time sentiment analysis and momentum detection",
    version="1.0.0",
    lifespan=lifespan
)

# Request models
class SentimentRequest(BaseModel):
    text: str

class BatchSentimentRequest(BaseModel):
    comments: List[Dict]

# API Endpoints
@app.get("/health")
async def health_endpoint():
    """Service health status"""
    return await service.health_check()

@app.post("/analyze")
async def analyze_sentiment_endpoint(request: SentimentRequest):
    """Analyze sentiment of single text"""
    try:
        result = service.analyze_sentiment(request.text)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_analyze_endpoint(request: BatchSentimentRequest):
    """Batch sentiment analysis"""
    try:
        result = await service.process_batch(request.comments)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sentiment/{ticker}")
async def get_ticker_sentiment(ticker: str):
    """Get current sentiment status for ticker"""
    try:
        cache_key = f"sentiment:{ticker}:latest"
        cached_data = await service.redis_client.get(cache_key)
        
        if cached_data:
            data = json.loads(cached_data)
            return {"status": "success", "data": data}
        else:
            return {"status": "success", "data": None, "message": "No recent sentiment data"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/momentum/{ticker}")
async def get_ticker_momentum(ticker: str):
    """Get sentiment momentum indicators for ticker"""
    try:
        momentum = service.calculate_sentiment_momentum(ticker)
        return {"status": "success", "data": momentum}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background Kafka consumer
async def kafka_consumer_task():
    """Background task to consume messages from Kafka"""
    logger.info("Starting Kafka consumer task")

    while True:
        try:
            for message in service.kafka_consumer:
                comment_data = message.value
                await service.process_comment(comment_data)
                service.kafka_producer.flush()
                service.kafka_consumer.commit()

        except Exception as e:
            logger.error(f"Error in Kafka consumer: {e}")
            await asyncio.sleep(5)

@app.on_event("startup")
async def start_background_tasks():
    """Start background tasks"""
    asyncio.create_task(kafka_consumer_task())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
