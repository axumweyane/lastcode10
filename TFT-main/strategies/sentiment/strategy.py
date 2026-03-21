"""
Sentiment Strategy — contrarian/momentum signals from NLP sentiment.

Compares 5-day sentiment trend vs 5-day price trend per symbol:
  - Divergence (sentiment up, price down): contrarian BUY signal (score * 1.5)
  - Divergence (sentiment down, price up): contrarian SELL signal (score * 1.5)
  - Alignment (both up or both down): momentum confirmation (score * 0.8)
  - Z-scored for ensemble consistency

Uses SentimentModel (#7) via ModelManager. Falls back gracefully if the model
is unavailable or sentiment data is missing.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models.manager import ModelManager
from strategies.base import (
    AlphaScore,
    BaseStrategy,
    SignalDirection,
    StrategyOutput,
    StrategyPerformance,
)
from strategies.config import SentimentConfig

logger = logging.getLogger(__name__)

# Scoring multipliers
DIVERGENCE_MULTIPLIER = 1.5
ALIGNMENT_MULTIPLIER = 0.8

# Trend lookback
TREND_WINDOW = 5


class SentimentStrategy(BaseStrategy):

    def __init__(
        self,
        config: Optional[SentimentConfig] = None,
        manager: Optional[ModelManager] = None,
    ):
        self.config = config or SentimentConfig.from_env()
        self._manager = manager
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._initialized = False

    @property
    def name(self) -> str:
        return "sentiment"

    @property
    def description(self) -> str:
        return "Sentiment divergence/alignment: contrarian on divergence, momentum on alignment"

    def initialize(self, historical_data: pd.DataFrame) -> None:
        self._initialized = True

    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        if not self._initialized:
            self.initialize(data)

        empty = StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=[],
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
        )

        # Get sentiment predictions from model
        sentiment_by_symbol = self._get_sentiment_scores(data)
        if not sentiment_by_symbol:
            logger.info("%s: no sentiment data available, returning empty", self.name)
            return empty

        # Compute 5-day price trends
        price_trends = self._compute_price_trends(data)
        if not price_trends:
            logger.info("%s: no price trend data available, returning empty", self.name)
            return empty

        scores: List[AlphaScore] = []

        for symbol, sentiment_score in sentiment_by_symbol.items():
            if symbol not in price_trends:
                continue

            price_trend = price_trends[symbol]

            # Classify relationship and compute raw score
            raw_score, direction, signal_type = self._classify_signal(
                sentiment_score, price_trend,
            )

            if direction == SignalDirection.NEUTRAL:
                continue

            # Confidence from sentiment strength and article count
            confidence = min(
                abs(sentiment_score) * 0.6 + 0.3,
                1.0,
            )

            scores.append(AlphaScore(
                symbol=symbol,
                score=raw_score,
                raw_score=raw_score,
                confidence=confidence,
                direction=direction,
                metadata={
                    "sentiment_score": round(sentiment_score, 4),
                    "price_trend": round(price_trend, 4),
                    "signal_type": signal_type,
                    "strategy_type": "sentiment",
                },
            ))

        # Z-score normalize
        if scores:
            raw_values = np.array([s.score for s in scores])
            mean = np.mean(raw_values)
            std = np.std(raw_values)
            if std > 1e-10:
                for s in scores:
                    s.score = (s.score - mean) / std

        # Trim to max positions
        scores.sort(key=lambda s: abs(s.score), reverse=True)
        max_pos = self.config.max_positions_per_side
        longs = [s for s in scores if s.direction == SignalDirection.LONG][:max_pos]
        shorts = [s for s in scores if s.direction == SignalDirection.SHORT][:max_pos]
        scores = longs + shorts

        logger.info("%s: %d signals (%d long, %d short)",
                    self.name, len(scores), len(longs), len(shorts))

        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=scores,
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
        )

    def _get_sentiment_scores(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get per-symbol sentiment scores from SentimentModel via ModelManager."""
        if self._manager is None:
            return {}

        try:
            predictions = self._manager.predict_sentiment(data)
            return {
                p.symbol: p.predicted_value
                for p in predictions
                if p.predicted_value is not None
            }
        except Exception as e:
            logger.warning("Sentiment model prediction failed: %s", e)
            return {}

    def _compute_price_trends(self, data: pd.DataFrame) -> Dict[str, float]:
        """Compute 5-day return per symbol as a trend indicator."""
        if data.empty or "close" not in data.columns or "symbol" not in data.columns:
            return {}

        trends = {}
        ts_col = "timestamp" if "timestamp" in data.columns else None

        for symbol in data["symbol"].unique():
            sym_data = data[data["symbol"] == symbol]
            if ts_col:
                sym_data = sym_data.sort_values(ts_col)

            closes = sym_data["close"].dropna().values
            if len(closes) < TREND_WINDOW + 1:
                continue

            # 5-day return
            recent = closes[-1]
            past = closes[-(TREND_WINDOW + 1)]
            if past > 0:
                trends[symbol] = (recent - past) / past

        return trends

    @staticmethod
    def _classify_signal(
        sentiment_score: float,
        price_trend: float,
    ) -> tuple:
        """
        Classify the sentiment-price relationship.

        Returns:
            (raw_score, direction, signal_type)
        """
        sentiment_up = sentiment_score > 0
        price_up = price_trend > 0

        divergence = abs(sentiment_score - price_trend)
        alignment = abs(sentiment_score + price_trend) / 2

        if sentiment_up and not price_up:
            # Sentiment positive, price falling -> contrarian BUY
            raw_score = divergence * DIVERGENCE_MULTIPLIER
            return raw_score, SignalDirection.LONG, "contrarian_buy"

        elif not sentiment_up and price_up:
            # Sentiment negative, price rising -> contrarian SELL
            raw_score = -divergence * DIVERGENCE_MULTIPLIER
            return raw_score, SignalDirection.SHORT, "contrarian_sell"

        elif sentiment_up and price_up:
            # Both positive -> momentum confirmation LONG
            raw_score = alignment * ALIGNMENT_MULTIPLIER
            return raw_score, SignalDirection.LONG, "momentum_long"

        elif not sentiment_up and not price_up:
            # Both negative -> momentum confirmation SHORT
            raw_score = -alignment * ALIGNMENT_MULTIPLIER
            return raw_score, SignalDirection.SHORT, "momentum_short"

        return 0.0, SignalDirection.NEUTRAL, "neutral"

    def get_performance(self) -> StrategyPerformance:
        return self._performance
