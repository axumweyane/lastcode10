"""
Strategy 5: Earnings Plays — use TFT + sentiment to trade options around earnings.

Edge: combining TFT's 5-day directional forecast with Reddit/news sentiment
creates an information advantage on the expected earnings reaction.

Logic:
  - High confidence directional (TFT + sentiment agree): buy bull/bear call/put spread
  - Low confidence (signals disagree or neutral): sell iron condor (premium capture)
  - No play if IV rank < 40% (not enough premium to justify risk)

Risk: max 2% of portfolio per earnings play (defined risk via spreads)
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base import (
    AlphaScore,
    BaseStrategy,
    SignalDirection,
    StrategyOutput,
    StrategyPerformance,
)
from strategies.options.config import EarningsPlayConfig
from strategies.options.infrastructure.vol_monitor import VolMonitor

logger = logging.getLogger(__name__)


class EarningsPlays(BaseStrategy):

    def __init__(self, config: Optional[EarningsPlayConfig] = None):
        self.config = config or EarningsPlayConfig.from_env()
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._vol_monitor = VolMonitor()
        self._initialized = False

    @property
    def name(self) -> str:
        return "earnings_plays"

    @property
    def description(self) -> str:
        return "Trade options around earnings using TFT + sentiment signals"

    def initialize(self, historical_data: pd.DataFrame) -> None:
        self._initialized = True

    def generate_signals(
        self,
        data: pd.DataFrame,
        tft_predictions: Optional[Dict[str, float]] = None,
        sentiment_scores: Optional[Dict[str, float]] = None,
        earnings_dates: Optional[Dict[str, datetime]] = None,
    ) -> StrategyOutput:
        """
        Args:
            data: price data
            tft_predictions: {symbol: predicted_5d_return} from TFT model
            sentiment_scores: {symbol: sentiment_score} from sentiment engine (-1 to +1)
            earnings_dates: {symbol: next_earnings_date}
        """
        if not self._initialized:
            self.initialize(data)

        tft_preds = tft_predictions or {}
        sentiments = sentiment_scores or {}
        earnings = earnings_dates or {}

        scores: List[AlphaScore] = []
        today = datetime.now().date()

        for symbol in data["symbol"].unique():
            # Check if earnings is within entry window
            earn_date = earnings.get(symbol)
            if earn_date is None:
                continue

            if hasattr(earn_date, "date"):
                earn_date = earn_date.date()
            days_to_earnings = (earn_date - today).days

            if days_to_earnings < 0 or days_to_earnings > self.config.entry_days_before:
                continue

            sym_data = data[data["symbol"] == symbol].sort_values("timestamp")
            if len(sym_data) < 30:
                continue

            close = sym_data["close"]
            spot = float(close.iloc[-1])
            returns = close.pct_change().dropna()
            rv = float(returns.tail(21).std() * np.sqrt(252))
            est_iv = rv * 1.30  # earnings IV crush premium

            metrics = self._vol_monitor.compute(symbol, sym_data, est_iv)

            if metrics.iv_rank < self.config.min_iv_rank:
                continue

            # Get TFT and sentiment signals
            tft_signal = tft_preds.get(symbol, 0.0)
            sent_signal = sentiments.get(symbol, 0.0)

            # Combine: average of TFT and sentiment direction
            combined_signal = 0.6 * tft_signal + 0.4 * sent_signal
            signal_confidence = abs(combined_signal)

            # Determine play type
            if signal_confidence >= self.config.min_confidence:
                # Directional play
                if combined_signal > 0:
                    play_type = "bull_call_spread"
                    direction = SignalDirection.LONG
                    raw_score = combined_signal
                else:
                    play_type = "bear_put_spread"
                    direction = SignalDirection.SHORT
                    raw_score = combined_signal
            else:
                # Neutral play — sell premium
                play_type = "iron_condor"
                direction = SignalDirection.NEUTRAL
                raw_score = metrics.iv_rank / 100.0 * 0.5  # moderate positive score

            # Estimate risk/reward
            spread_width = self.config.directional_spread_width
            max_risk = spread_width * 100  # per contract
            max_reward = max_risk * 2  # rough 2:1 on directional

            scores.append(
                AlphaScore(
                    symbol=symbol,
                    score=raw_score,
                    raw_score=raw_score,
                    confidence=min(signal_confidence + 0.2, 0.95),
                    direction=direction,
                    metadata={
                        "strategy_type": "earnings_play",
                        "play_type": play_type,
                        "earnings_date": str(earn_date),
                        "days_to_earnings": days_to_earnings,
                        "tft_signal": round(tft_signal, 4),
                        "sentiment_signal": round(sent_signal, 4),
                        "combined_signal": round(combined_signal, 4),
                        "iv_rank": round(metrics.iv_rank, 1),
                        "spot": round(spot, 2),
                        "spread_width": spread_width,
                        "max_risk_per_contract": max_risk,
                    },
                )
            )

        logger.info("%s: %d earnings plays", self.name, len(scores))

        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=scores,
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
        )

    def get_performance(self) -> StrategyPerformance:
        return self._performance
