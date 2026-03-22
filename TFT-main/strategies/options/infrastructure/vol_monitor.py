"""
Volatility monitoring: IV rank, IV percentile, and IV-RV spread.

These are the core signals that drive options strategy timing:
  - IV Rank > 50% → sell premium (covered calls, iron condors)
  - IV Rank < 20% → buy premium (protective puts, gamma scalping)
  - IV >> RV → vol is overpriced → sell (vol arb)
  - IV << RV → vol is underpriced → buy (gamma scalping)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VolMetrics:
    """Volatility metrics for a single underlying."""

    symbol: str
    current_iv: float  # current ATM implied vol
    iv_rank: float  # (current - 52w low) / (52w high - 52w low) * 100
    iv_percentile: float  # % of days IV was below current level
    iv_52w_high: float
    iv_52w_low: float
    realized_vol_21d: float  # 21-day realized vol
    realized_vol_63d: float  # 63-day realized vol
    iv_rv_spread: float  # current_iv - realized_vol_21d
    iv_rv_ratio: float  # current_iv / realized_vol_21d
    garch_forecast: float  # GARCH(1,1) 1-step ahead vol forecast
    vol_regime: str  # "low", "normal", "elevated", "extreme"

    @property
    def is_iv_elevated(self) -> bool:
        return self.iv_rank > 50.0

    @property
    def is_iv_cheap(self) -> bool:
        return self.iv_rank < 25.0

    @property
    def iv_overpriced(self) -> bool:
        return self.iv_rv_spread > 0.03  # IV > RV by 3+ vol points

    @property
    def iv_underpriced(self) -> bool:
        return self.iv_rv_spread < -0.03


class VolMonitor:
    """
    Tracks implied and realized volatility metrics over time.

    Usage:
        monitor = VolMonitor(lookback_days=252)
        metrics = monitor.compute(symbol="AAPL", price_history=df, current_iv=0.28)
    """

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self._iv_history: Dict[str, List[float]] = {}

    def compute(
        self,
        symbol: str,
        price_history: pd.DataFrame,
        current_iv: float,
    ) -> VolMetrics:
        """
        Compute all vol metrics for a symbol.

        Args:
            symbol: Stock ticker.
            price_history: DataFrame with [timestamp, close] sorted ascending.
            current_iv: Current ATM implied volatility (from options chain).
        """
        close = price_history["close"].values

        # Realized vol
        returns = np.diff(np.log(close))
        rv_21d = (
            float(np.std(returns[-21:]) * np.sqrt(252)) if len(returns) >= 21 else 0.20
        )
        rv_63d = (
            float(np.std(returns[-63:]) * np.sqrt(252)) if len(returns) >= 63 else 0.20
        )

        # IV history tracking
        self._record_iv(symbol, current_iv)
        iv_hist = self._iv_history.get(symbol, [current_iv])

        # IV Rank
        if len(iv_hist) >= 20:
            iv_high = max(iv_hist[-self.lookback_days :])
            iv_low = min(iv_hist[-self.lookback_days :])
            iv_range = iv_high - iv_low
            iv_rank = (
                ((current_iv - iv_low) / iv_range * 100) if iv_range > 0.001 else 50.0
            )
        else:
            iv_high = current_iv * 1.3
            iv_low = current_iv * 0.7
            iv_rank = 50.0

        # IV Percentile
        if len(iv_hist) >= 20:
            below = sum(1 for v in iv_hist[-self.lookback_days :] if v < current_iv)
            iv_percentile = below / len(iv_hist[-self.lookback_days :]) * 100
        else:
            iv_percentile = 50.0

        # IV-RV spread
        iv_rv_spread = current_iv - rv_21d
        iv_rv_ratio = current_iv / rv_21d if rv_21d > 0.01 else 1.0

        # GARCH forecast
        garch_forecast = self._garch_forecast(returns)

        # Vol regime classification
        vol_regime = self._classify_regime(current_iv, rv_21d, iv_rank)

        return VolMetrics(
            symbol=symbol,
            current_iv=current_iv,
            iv_rank=min(max(iv_rank, 0), 100),
            iv_percentile=min(max(iv_percentile, 0), 100),
            iv_52w_high=iv_high,
            iv_52w_low=iv_low,
            realized_vol_21d=rv_21d,
            realized_vol_63d=rv_63d,
            iv_rv_spread=iv_rv_spread,
            iv_rv_ratio=iv_rv_ratio,
            garch_forecast=garch_forecast,
            vol_regime=vol_regime,
        )

    def _record_iv(self, symbol: str, iv: float) -> None:
        if symbol not in self._iv_history:
            self._iv_history[symbol] = []
        self._iv_history[symbol].append(iv)
        # Keep last 2 years
        if len(self._iv_history[symbol]) > 504:
            self._iv_history[symbol] = self._iv_history[symbol][-504:]

    def _garch_forecast(self, returns: np.ndarray) -> float:
        """
        GARCH(1,1) volatility forecast using the arch library.

        Returns 1-step-ahead annualized volatility forecast.
        """
        if len(returns) < 50:
            return (
                float(np.std(returns[-21:]) * np.sqrt(252))
                if len(returns) >= 21
                else 0.20
            )

        try:
            from arch import arch_model

            # Scale to percentage returns for numerical stability
            scaled = returns[-252:] * 100

            am = arch_model(
                scaled, vol="Garch", p=1, q=1, mean="Constant", rescale=False
            )
            res = am.fit(disp="off", show_warning=False)

            # 1-step forecast
            forecast = res.forecast(horizon=1)
            daily_var = forecast.variance.values[-1, 0]
            # Convert back from percentage^2 to decimal annualized
            daily_vol = np.sqrt(daily_var) / 100
            annual_vol = daily_vol * np.sqrt(252)

            return float(min(max(annual_vol, 0.05), 2.0))

        except Exception as e:
            logger.debug("GARCH forecast failed: %s, using ewma fallback", e)
            return self._ewma_vol(returns)

    @staticmethod
    def _ewma_vol(returns: np.ndarray, span: int = 21) -> float:
        """Exponentially weighted moving average volatility as GARCH fallback."""
        if len(returns) < 5:
            return 0.20
        weights = np.exp(-np.arange(min(len(returns), span)) / (span / 2))
        weights = weights[::-1]
        recent = returns[-len(weights) :]
        weighted_var = np.average(recent**2, weights=weights)
        return float(np.sqrt(weighted_var * 252))

    @staticmethod
    def _classify_regime(current_iv: float, rv_21d: float, iv_rank: float) -> str:
        if iv_rank >= 80 or current_iv > 0.40:
            return "extreme"
        elif iv_rank >= 50 or current_iv > 0.25:
            return "elevated"
        elif iv_rank >= 20:
            return "normal"
        else:
            return "low"
