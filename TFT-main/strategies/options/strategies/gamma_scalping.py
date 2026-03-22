"""
Strategy 6: Gamma Scalping — buy ATM straddles + delta hedge with stock.

Edge: when realized volatility exceeds implied volatility (RV > IV),
the gamma P&L from delta hedging exceeds the theta decay cost.

This is a pure vol trade: you're long gamma (profit from moves) and
short theta (pay time decay). Net P&L = gamma_pnl - theta_cost.

Entry: GARCH forecasts RV > IV by threshold, volatile trending regime
Exit:  close when RV drops below IV, or at DTE <= 5
Hedge: delta hedge every 4 hours (configurable)
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
from strategies.options.config import GammaScalpConfig
from strategies.options.infrastructure.vol_monitor import VolMonitor

logger = logging.getLogger(__name__)


class GammaScalping(BaseStrategy):

    def __init__(self, config: Optional[GammaScalpConfig] = None):
        self.config = config or GammaScalpConfig.from_env()
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._vol_monitor = VolMonitor()
        self._initialized = False

    @property
    def name(self) -> str:
        return "gamma_scalping"

    @property
    def description(self) -> str:
        return "Buy ATM straddles and delta hedge to capture RV > IV spread"

    def initialize(self, historical_data: pd.DataFrame) -> None:
        self._initialized = True

    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        if not self._initialized:
            self.initialize(data)

        scores: List[AlphaScore] = []
        symbols = data["symbol"].unique()

        for symbol in symbols:
            sym_data = data[data["symbol"] == symbol].sort_values("timestamp")
            if len(sym_data) < 100:
                continue

            close = sym_data["close"]
            spot = float(close.iloc[-1])
            returns = close.pct_change().dropna()

            # Compute vol metrics
            rv_21d = float(returns.tail(21).std() * np.sqrt(252))
            estimated_iv = rv_21d * 1.10  # slight premium
            metrics = self._vol_monitor.compute(symbol, sym_data, estimated_iv)

            garch_rv = metrics.garch_forecast
            rv_iv_spread = garch_rv - estimated_iv

            # Only enter when RV is forecasted to exceed IV
            if rv_iv_spread < self.config.rv_iv_threshold:
                continue

            # Estimate gamma P&L vs theta cost
            dte = (self.config.min_dte + self.config.max_dte) // 2
            dte_years = dte / 365.0
            sqrt_t = np.sqrt(dte_years)

            # ATM straddle: gamma ≈ N'(0) / (S * vol * sqrt(T))
            # N'(0) = 0.3989
            gamma_estimate = 0.3989 / (spot * estimated_iv * sqrt_t)

            # Daily gamma P&L ≈ 0.5 * gamma * (daily_move)^2
            expected_daily_move = spot * garch_rv / np.sqrt(252)
            daily_gamma_pnl = 0.5 * gamma_estimate * expected_daily_move**2

            # Daily theta ≈ -S * N'(0) * vol / (2 * sqrt(T)) / 365
            daily_theta = -spot * 0.3989 * estimated_iv / (2 * sqrt_t) / 365

            # Net expected daily P&L (per share equivalent)
            net_daily = daily_gamma_pnl + daily_theta  # theta is negative

            if net_daily <= 0:
                continue  # theta would eat the gamma — skip

            # Straddle premium estimate
            straddle_premium = (
                spot * estimated_iv * sqrt_t * 0.80
            )  # ~80% of theoretical

            # Score: higher RV-IV spread → stronger signal
            raw_score = rv_iv_spread / 0.10  # normalize: 10 vol pts → score 1.0
            confidence = min(rv_iv_spread / 0.15, 0.90)

            scores.append(
                AlphaScore(
                    symbol=symbol,
                    score=raw_score,
                    raw_score=rv_iv_spread,
                    confidence=confidence,
                    direction=SignalDirection.LONG,  # buying straddle = long vol
                    metadata={
                        "strategy_type": "gamma_scalp",
                        "estimated_iv": round(estimated_iv, 4),
                        "rv_21d": round(rv_21d, 4),
                        "garch_rv_forecast": round(garch_rv, 4),
                        "rv_iv_spread": round(rv_iv_spread, 4),
                        "gamma_estimate": round(gamma_estimate, 6),
                        "daily_gamma_pnl": round(daily_gamma_pnl, 4),
                        "daily_theta": round(daily_theta, 4),
                        "net_daily_estimate": round(net_daily, 4),
                        "straddle_premium": round(straddle_premium, 2),
                        "spot": round(spot, 2),
                        "hedge_freq_hours": self.config.hedge_frequency_hours,
                    },
                )
            )

        scores.sort(key=lambda s: s.score, reverse=True)
        logger.info(
            "%s: %d signals from %d symbols", self.name, len(scores), len(symbols)
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
