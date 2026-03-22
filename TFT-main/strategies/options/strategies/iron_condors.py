"""
Strategy 2: Iron Condors — sell OTM put spread + OTM call spread on index ETFs.

Edge: index options (SPY, QQQ) are systematically overpriced because
institutional hedgers pay for downside protection. When IV rank > 50%,
the premium collected exceeds the expected move ~65% of the time.

Structure: sell put at -1 SD, buy put at -1.5 SD, sell call at +1 SD, buy call at +1.5 SD
Entry: IV rank > 50%, regime is calm (not volatile trending)
Exit:  close at 50% profit or 200% of credit received (stop loss)
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
from strategies.options.config import IronCondorConfig
from strategies.options.infrastructure.vol_monitor import VolMonitor

logger = logging.getLogger(__name__)


class IronCondors(BaseStrategy):

    def __init__(self, config: Optional[IronCondorConfig] = None):
        self.config = config or IronCondorConfig.from_env()
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._vol_monitor = VolMonitor()
        self._initialized = False

    @property
    def name(self) -> str:
        return "iron_condors"

    @property
    def description(self) -> str:
        return "Sell iron condors on SPY/QQQ when IV rank is elevated"

    def initialize(self, historical_data: pd.DataFrame) -> None:
        self._initialized = True

    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        if not self._initialized:
            self.initialize(data)

        scores: List[AlphaScore] = []

        for underlying in self.config.underlyings:
            sym_data = data[data["symbol"] == underlying].sort_values("timestamp")
            if len(sym_data) < 63:
                continue

            close = sym_data["close"]
            spot = float(close.iloc[-1])
            returns = close.pct_change().dropna()
            rv_21d = float(returns.tail(21).std() * np.sqrt(252))
            estimated_iv = rv_21d * 1.20  # index options carry higher premium

            metrics = self._vol_monitor.compute(underlying, sym_data, estimated_iv)

            # Only sell condors when IV is elevated
            if metrics.iv_rank < self.config.min_iv_rank:
                continue

            # Calculate expected strikes (1 SD move)
            dte = (self.config.min_dte + self.config.max_dte) // 2
            dte_years = dte / 365.0
            one_sd = spot * estimated_iv * np.sqrt(dte_years)
            wing_width = self.config.wing_width_std

            short_put = spot - one_sd * wing_width
            long_put = spot - one_sd * (wing_width + 0.5)
            short_call = spot + one_sd * wing_width
            long_call = spot + one_sd * (wing_width + 0.5)

            # Estimate credit received (rough: ~30% of wing width)
            wing_dollar = short_put - long_put
            credit_estimate = wing_dollar * 0.30
            max_loss = wing_dollar - credit_estimate

            # Score: higher IV rank → more premium → stronger signal
            raw_score = (metrics.iv_rank - self.config.min_iv_rank) / 50.0
            confidence = min(metrics.iv_rank / 100.0, 0.90)

            scores.append(
                AlphaScore(
                    symbol=underlying,
                    score=max(raw_score, 0.01),
                    raw_score=raw_score,
                    confidence=confidence,
                    direction=SignalDirection.NEUTRAL,  # market-neutral structure
                    metadata={
                        "strategy_type": "iron_condor",
                        "iv_rank": round(metrics.iv_rank, 1),
                        "iv_rv_spread": round(metrics.iv_rv_spread, 4),
                        "spot": round(spot, 2),
                        "short_put": round(short_put, 2),
                        "long_put": round(long_put, 2),
                        "short_call": round(short_call, 2),
                        "long_call": round(long_call, 2),
                        "credit_estimate": round(credit_estimate, 2),
                        "max_loss": round(max_loss, 2),
                        "target_dte": dte,
                        "vol_regime": metrics.vol_regime,
                    },
                )
            )

        logger.info("%s: %d signals", self.name, len(scores))

        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=scores,
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
        )

    def get_performance(self) -> StrategyPerformance:
        return self._performance
