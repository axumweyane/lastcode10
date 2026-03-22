"""
FX Volatility Breakout Strategy — Bollinger squeeze to expansion.

Alpha source: breakouts from low-volatility compression. When Bollinger
bandwidth reaches a 6-month low and vol forecast is rising, position
for directional breakout.

Uncorrelated with other strategies: low signal frequency (only during
squeezes), event-driven.
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
from strategies.config import FXVolBreakoutConfig

logger = logging.getLogger(__name__)


class FXVolBreakoutStrategy(BaseStrategy):

    def __init__(self, config: Optional[FXVolBreakoutConfig] = None):
        self.config = config or FXVolBreakoutConfig.from_env()
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._initialized = False

    @property
    def name(self) -> str:
        return "fx_vol_breakout"

    @property
    def description(self) -> str:
        return "Bollinger squeeze breakout: position for expansion when bandwidth hits 6-month lows"

    def initialize(self, historical_data: pd.DataFrame) -> None:
        self._initialized = True

    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        if not self._initialized:
            self.initialize(data)

        scores: List[AlphaScore] = []
        symbols = data["symbol"].unique()

        for symbol in symbols:
            sym_data = data[data["symbol"] == symbol].sort_values(
                "timestamp" if "timestamp" in data.columns else data.columns[0]
            )

            close = sym_data["close"].dropna()
            if len(close) < self.config.lookback_days:
                continue

            close_arr = close.values.astype(float)

            # Bollinger Bands
            bb_window = self.config.bb_window
            if len(close_arr) < bb_window:
                continue

            ma = np.convolve(close_arr, np.ones(bb_window) / bb_window, mode="valid")
            std = pd.Series(close_arr).rolling(bb_window).std().dropna().values

            if len(ma) < 2 or len(std) < 2:
                continue

            # Bandwidth = (upper - lower) / middle = 2 * std / ma
            current_ma = ma[-1]
            current_std = std[-1]
            if current_ma <= 0 or current_std <= 0:
                continue

            bandwidth = 2.0 * current_std / current_ma

            # Historical bandwidth for percentile ranking
            bandwidths = []
            for i in range(min(len(ma), len(std))):
                if ma[i] > 0:
                    bandwidths.append(2.0 * std[i] / ma[i])

            if len(bandwidths) < self.config.squeeze_lookback:
                continue

            # Squeeze detection: bandwidth at percentile low
            recent_bw = bandwidths[-self.config.squeeze_lookback :]
            bandwidth_percentile = sum(1 for b in recent_bw if b <= bandwidth) / len(
                recent_bw
            )

            is_squeeze = bandwidth_percentile <= self.config.squeeze_percentile

            if not is_squeeze:
                continue

            # Squeeze intensity: how tight is the squeeze relative to history
            squeeze_intensity = 1.0 - bandwidth_percentile

            # Volatility forecast: is vol expected to expand?
            # Simple: compare recent realized vol to longer-term
            recent_vol = float(
                pd.Series(close_arr[-10:]).pct_change().dropna().std()
            ) * np.sqrt(252)
            longer_vol = float(
                pd.Series(close_arr[-63:]).pct_change().dropna().std()
            ) * np.sqrt(252)

            vol_expanding = recent_vol > longer_vol * 0.8  # vol starting to pick up

            if not vol_expanding and squeeze_intensity < 0.8:
                # Need either vol expansion signal or very tight squeeze
                continue

            vol_forecast_change = (
                (recent_vol / longer_vol - 1.0) if longer_vol > 0 else 0.0
            )

            # Direction: price momentum during squeeze
            momentum_window = min(self.config.momentum_window, len(close_arr) - 1)
            price_momentum = (
                (close_arr[-1] / close_arr[-momentum_window] - 1)
                if momentum_window > 0
                else 0.0
            )
            directional_bias = np.sign(price_momentum)

            if directional_bias == 0:
                continue

            # Score = squeeze_intensity * vol_forecast_change * directional_bias
            raw_score = (
                squeeze_intensity
                * (1.0 + max(vol_forecast_change, 0.0))
                * directional_bias
            )

            direction = SignalDirection.LONG if raw_score > 0 else SignalDirection.SHORT
            confidence = min(
                squeeze_intensity * 0.6 + abs(vol_forecast_change) * 0.4, 0.9
            )

            scores.append(
                AlphaScore(
                    symbol=symbol,
                    score=raw_score,
                    raw_score=raw_score,
                    confidence=confidence,
                    direction=direction,
                    metadata={
                        "strategy_type": "fx_vol_breakout",
                        "bandwidth": round(bandwidth, 6),
                        "bandwidth_percentile": round(bandwidth_percentile, 4),
                        "squeeze_intensity": round(squeeze_intensity, 4),
                        "vol_forecast_change": round(vol_forecast_change, 4),
                        "price_momentum": round(price_momentum, 6),
                        "recent_vol": round(recent_vol, 4),
                        "longer_vol": round(longer_vol, 4),
                    },
                )
            )

        # Z-score normalize
        if scores:
            raw_values = np.array([s.score for s in scores])
            mean = np.mean(raw_values)
            std_val = np.std(raw_values)
            if std_val > 1e-10:
                for s in scores:
                    s.score = (s.score - mean) / std_val

        scores.sort(key=lambda s: abs(s.score), reverse=True)

        logger.info("%s: %d breakout signals", self.name, len(scores))

        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=scores,
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
        )

    def get_performance(self) -> StrategyPerformance:
        return self._performance
