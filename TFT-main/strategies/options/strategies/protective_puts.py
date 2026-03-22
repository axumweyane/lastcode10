"""
Strategy 3: Protective Puts — portfolio insurance in volatile regimes.

Edge: negative expected return (you're paying for insurance), but the
portfolio-level Sharpe improves because the variance reduction from
cutting tail risk more than compensates for the premium cost.

This strategy ONLY activates in VOLATILE regimes (detected by the
regime detector). In calm markets, it sits idle.

Entry: regime = volatile_trending OR volatile_choppy, buy OTM puts at ~20 delta
Exit:  close when regime returns to calm OR puts expire
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
from strategies.options.config import ProtectivePutConfig
from strategies.options.infrastructure.vol_monitor import VolMonitor

logger = logging.getLogger(__name__)


class ProtectivePuts(BaseStrategy):

    def __init__(self, config: Optional[ProtectivePutConfig] = None):
        self.config = config or ProtectivePutConfig.from_env()
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._vol_monitor = VolMonitor()
        self._is_volatile_regime = False
        self._initialized = False

    @property
    def name(self) -> str:
        return "protective_puts"

    @property
    def description(self) -> str:
        return "Buy OTM puts for portfolio insurance during volatile regimes"

    def initialize(self, historical_data: pd.DataFrame) -> None:
        self._initialized = True

    def set_regime(self, is_volatile: bool) -> None:
        """Called by the regime detector to toggle this strategy."""
        self._is_volatile_regime = is_volatile

    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        if not self._initialized:
            self.initialize(data)

        scores: List[AlphaScore] = []

        # Check regime gate
        if self.config.only_volatile_regime and not self._is_volatile_regime:
            # Auto-detect from data if regime detector hasn't set it
            spy_data = data[data["symbol"].isin(["SPY", "QQQ"])]
            if not spy_data.empty:
                sym = spy_data["symbol"].iloc[0]
                sym_data = spy_data[spy_data["symbol"] == sym].sort_values("timestamp")
                if len(sym_data) >= 21:
                    rv = float(
                        sym_data["close"].pct_change().tail(21).std() * np.sqrt(252)
                    )
                    self._is_volatile_regime = rv > 0.20  # >20% annualized vol

            if not self._is_volatile_regime:
                return self._empty_output()

        # Generate put signals for the largest/most liquid positions
        symbols = data["symbol"].unique()

        for symbol in symbols:
            sym_data = data[data["symbol"] == symbol].sort_values("timestamp")
            if len(sym_data) < 30:
                continue

            close = sym_data["close"]
            spot = float(close.iloc[-1])
            returns = close.pct_change().dropna()
            rv_21d = float(returns.tail(21).std() * np.sqrt(252))

            # Put strike at target delta OTM
            target_delta = abs(self.config.target_delta)
            dte = (self.config.min_dte + self.config.max_dte) // 2
            dte_years = dte / 365.0
            put_strike = spot * (1 - rv_21d * np.sqrt(dte_years) * 0.84)  # ~20 delta

            # Estimate premium
            premium_estimate = rv_21d * np.sqrt(dte_years) * spot * 0.10  # rough
            premium_pct = premium_estimate / spot * 100

            # Cap total premium expenditure
            if premium_pct > self.config.max_premium_pct:
                continue

            # Score: higher volatility → more urgent need for protection
            urgency = rv_21d / 0.20 - 1.0  # 0 at 20% vol, positive above
            confidence = min(rv_21d / 0.30, 0.90)

            scores.append(
                AlphaScore(
                    symbol=symbol,
                    score=max(urgency, 0.01),
                    raw_score=urgency,
                    confidence=confidence,
                    direction=SignalDirection.LONG,  # buying puts = long puts
                    metadata={
                        "strategy_type": "protective_put",
                        "put_strike": round(put_strike, 2),
                        "spot": round(spot, 2),
                        "rv_21d": round(rv_21d, 4),
                        "premium_estimate_pct": round(premium_pct, 2),
                        "hedge_ratio": self.config.hedge_ratio,
                        "target_dte": dte,
                    },
                )
            )

        scores.sort(key=lambda s: s.score, reverse=True)
        logger.info(
            "%s: %d signals (volatile_regime=%s)",
            self.name,
            len(scores),
            self._is_volatile_regime,
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

    def _empty_output(self) -> StrategyOutput:
        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=[],
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
            metadata={"reason": "calm_regime_no_puts_needed"},
        )
