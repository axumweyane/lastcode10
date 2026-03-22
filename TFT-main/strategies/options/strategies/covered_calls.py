"""
Strategy 1: Covered Calls — sell OTM calls on existing long equity positions.

Edge: captures the volatility risk premium (IV > RV ~85% of the time).
The premium received enhances returns on positions you already hold.

Entry: sell call at delta 0.20-0.30, 25-45 DTE, when IV rank > 20%
Exit:  close at 50% of max profit OR auto-roll at 7 DTE
Kill:  if strategy drawdown > 15% or Sharpe < -1.0

This overlays on top of the momentum/equity strategies — it does NOT
require additional capital, just existing long positions.
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
from strategies.options.config import CoveredCallConfig
from strategies.options.infrastructure.vol_monitor import VolMonitor, VolMetrics

logger = logging.getLogger(__name__)


class CoveredCalls(BaseStrategy):

    def __init__(self, config: Optional[CoveredCallConfig] = None):
        self.config = config or CoveredCallConfig.from_env()
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._vol_monitor = VolMonitor()
        self._initialized = False

    @property
    def name(self) -> str:
        return "covered_calls"

    @property
    def description(self) -> str:
        return "Sell OTM covered calls on long positions to capture vol risk premium"

    def initialize(self, historical_data: pd.DataFrame) -> None:
        self._initialized = True

    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        if not self._initialized:
            self.initialize(data)

        scores: List[AlphaScore] = []
        symbols = data["symbol"].unique()

        for symbol in symbols:
            sym_data = data[data["symbol"] == symbol].sort_values("timestamp")
            if len(sym_data) < 63:
                continue

            # Estimate current IV from recent realized vol * 1.15 (avg premium)
            returns = sym_data["close"].pct_change().dropna()
            rv_21d = float(returns.tail(21).std() * np.sqrt(252))
            estimated_iv = rv_21d * 1.15  # rough IV estimate without chain data

            metrics = self._vol_monitor.compute(symbol, sym_data, estimated_iv)

            # Only sell calls when IV is rich enough
            if metrics.iv_rank < self.config.min_iv_rank:
                continue

            # Signal: higher IV rank → stronger sell signal
            # Score is positive because selling calls = income on long positions
            raw_score = (metrics.iv_rank - 50) / 50.0  # -1 to +1 centered at 50%
            premium_estimate = (
                estimated_iv * np.sqrt(30 / 365) * sym_data["close"].iloc[-1]
            )
            premium_pct = premium_estimate / sym_data["close"].iloc[-1] * 100

            confidence = min(metrics.iv_rank / 100.0, 0.95)

            scores.append(
                AlphaScore(
                    symbol=symbol,
                    score=max(raw_score, 0.01),  # always slightly positive (income)
                    raw_score=raw_score,
                    confidence=confidence,
                    direction=SignalDirection.LONG,  # overlays on existing longs
                    metadata={
                        "strategy_type": "covered_call",
                        "iv_rank": round(metrics.iv_rank, 1),
                        "estimated_iv": round(estimated_iv, 4),
                        "rv_21d": round(rv_21d, 4),
                        "target_delta": self.config.target_delta,
                        "target_dte": (self.config.min_dte + self.config.max_dte) // 2,
                        "premium_estimate_pct": round(premium_pct, 2),
                        "vol_regime": metrics.vol_regime,
                    },
                )
            )

        scores.sort(key=lambda s: s.score, reverse=True)

        logger.info(
            "%s: %d signals from %d symbols (min IV rank: %.0f%%)",
            self.name,
            len(scores),
            len(symbols),
            self.config.min_iv_rank,
        )

        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=scores,
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
            metadata={"signals": len(scores), "symbols_scanned": len(symbols)},
        )

    def get_performance(self) -> StrategyPerformance:
        return self._performance
