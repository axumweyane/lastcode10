"""
FX Momentum Strategy — time-series momentum on currency pairs.

Alpha source: weighted composite of multi-lookback returns (1m, 3m, 6m, 12m).
Complement to FX Carry (value): carry captures rate differentials,
momentum captures sustained trends. Correlation ~0.2.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from strategies.base import (
    AlphaScore,
    BaseStrategy,
    SignalDirection,
    StrategyOutput,
    StrategyPerformance,
)
from strategies.config import FXMomentumConfig

logger = logging.getLogger(__name__)

# Lookback periods and weights (shorter = more weight)
DEFAULT_LOOKBACKS = {
    21: 0.4,  # 1 month
    63: 0.3,  # 3 months
    126: 0.2,  # 6 months
    252: 0.1,  # 12 months
}


class FXMomentumStrategy(BaseStrategy):

    def __init__(self, config: Optional[FXMomentumConfig] = None):
        self.config = config or FXMomentumConfig.from_env()
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._initialized = False

    @property
    def name(self) -> str:
        return "fx_momentum"

    @property
    def description(self) -> str:
        return "Multi-lookback time-series momentum on FX pairs with trend confirmation"

    def initialize(self, historical_data: pd.DataFrame) -> None:
        self._initialized = True

    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        if not self._initialized:
            self.initialize(data)

        scores: List[AlphaScore] = []
        symbols = data["symbol"].unique()

        trend_scores = {}
        for symbol in symbols:
            sym_data = data[data["symbol"] == symbol].sort_values(
                "timestamp" if "timestamp" in data.columns else data.columns[0]
            )

            close = sym_data["close"].dropna()
            if len(close) < self.config.min_lookback_days:
                continue

            # Compute weighted composite momentum
            composite = 0.0
            total_weight = 0.0
            lookback_details = {}

            for lookback, weight in DEFAULT_LOOKBACKS.items():
                if len(close) < lookback + 1:
                    continue
                ret = float(close.iloc[-1] / close.iloc[-lookback] - 1)
                composite += ret * weight
                total_weight += weight
                lookback_details[f"ret_{lookback}d"] = round(ret, 6)

            if total_weight < 0.3:  # need at least some lookback data
                continue

            composite /= total_weight

            # Trend confirmation: directional consistency (ADX-like)
            if len(close) >= 21:
                returns = close.pct_change().dropna().tail(21)
                positive_days = (returns > 0).sum()
                directional_ratio = float(positive_days) / len(returns)
                trend_consistency = (
                    abs(directional_ratio - 0.5) * 2.0
                )  # 0 = no trend, 1 = all same direction
            else:
                trend_consistency = 0.5

            trend_scores[symbol] = {
                "composite": composite,
                "trend_consistency": trend_consistency,
                "lookbacks": lookback_details,
            }

        if not trend_scores:
            return StrategyOutput(
                strategy_name=self.name,
                timestamp=datetime.now(timezone.utc),
                scores=[],
            )

        # Cross-sectional z-score
        composites = np.array([v["composite"] for v in trend_scores.values()])
        mean = np.mean(composites)
        std = np.std(composites)

        for symbol, info in trend_scores.items():
            if std > 1e-10:
                z_score = (info["composite"] - mean) / std
            else:
                z_score = 0.0

            if abs(z_score) < self.config.signal_threshold:
                continue

            direction = SignalDirection.LONG if z_score > 0 else SignalDirection.SHORT
            confidence = min(
                abs(z_score) / 2.0 * 0.6 + info["trend_consistency"] * 0.4,
                0.95,
            )

            scores.append(
                AlphaScore(
                    symbol=symbol,
                    score=z_score,
                    raw_score=info["composite"],
                    confidence=confidence,
                    direction=direction,
                    metadata={
                        "strategy_type": "fx_momentum",
                        "trend_consistency": round(info["trend_consistency"], 4),
                        **info["lookbacks"],
                    },
                )
            )

        scores.sort(key=lambda s: abs(s.score), reverse=True)
        max_long = self.config.max_pairs_long
        max_short = self.config.max_pairs_short
        longs = [s for s in scores if s.direction == SignalDirection.LONG][:max_long]
        shorts = [s for s in scores if s.direction == SignalDirection.SHORT][:max_short]
        scores = longs + shorts

        logger.info(
            "%s: %d signals (%d long, %d short)",
            self.name,
            len(scores),
            len(longs),
            len(shorts),
        )

        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=scores,
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
        )

    def get_performance(self) -> StrategyPerformance:
        return self._performance
