"""
FX Carry + Trend Following Strategy.

Two independent signals combined:

1. CARRY: Currencies with higher interest rates tend to appreciate.
   This is the "carry trade" — borrow in low-yield currencies, invest in
   high-yield currencies. The edge comes from the forward rate bias:
   high-yield currencies depreciate less than the interest rate differential
   implies, so carry traders earn excess returns.

2. TREND: Currencies that have been trending continue in that direction.
   3-month price momentum on FX pairs. The edge comes from central bank
   policy persistence — rate hike/cut cycles last months, creating
   sustained trends.

These two signals have low correlation (~0.2) because carry is a
mean-reversion/value signal while trend is a momentum signal.

Pairs: 6 majors — EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF.

Interest rate data: Updated monthly from central bank rates.
Since we can't fetch live rates in backtesting, we use a static table
that gets updated in production via config.
"""

import logging
from dataclasses import dataclass, field
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
from strategies.config import FXConfig

logger = logging.getLogger(__name__)

# Central bank policy rates (approximate, as of early 2025)
# Updated periodically — in production, fetch from an API or config
DEFAULT_RATES: Dict[str, float] = {
    "USD": 4.50,
    "EUR": 2.75,
    "GBP": 4.50,
    "JPY": 0.50,
    "AUD": 4.10,
    "CAD": 3.25,
    "CHF": 0.50,
}

# Map pair names to (base_currency, quote_currency)
PAIR_CURRENCIES: Dict[str, tuple] = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "AUDUSD": ("AUD", "USD"),
    "USDCAD": ("USD", "CAD"),
    "USDCHF": ("USD", "CHF"),
}


@dataclass
class FXSignal:
    """Signal for a single FX pair."""

    pair: str
    carry_score: float  # interest rate differential (annualized %)
    trend_score: float  # price momentum z-score
    combined_score: float  # weighted combination
    direction: str  # "long" or "short" the pair


class FXCarryTrend(BaseStrategy):
    """
    FX Carry + Trend Following strategy.

    Produces alpha scores for FX pairs. The ensemble combiner treats these
    alongside stock strategy scores — the near-zero correlation with equity
    strategies is the key diversification benefit.
    """

    def __init__(
        self,
        config: Optional[FXConfig] = None,
        interest_rates: Optional[Dict[str, float]] = None,
    ):
        self.config = config or FXConfig.from_env()
        self.rates = interest_rates or DEFAULT_RATES
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._initialized = False

    @property
    def name(self) -> str:
        return "fx_carry_trend"

    @property
    def description(self) -> str:
        return (
            "FX carry + trend following on 6 major currency pairs. "
            "Carry exploits interest rate differentials; trend exploits "
            "central bank policy persistence."
        )

    def initialize(self, historical_data: pd.DataFrame) -> None:
        """
        Warm up with historical FX price data.

        Args:
            historical_data: DataFrame with [symbol, timestamp, close]
                            where symbol is pair name (e.g., "EURUSD").
        """
        self._validate_data(historical_data)
        self._initialized = True
        logger.info(
            "%s initialized with %d pairs",
            self.name,
            historical_data["symbol"].nunique(),
        )

    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        """
        Generate carry + trend signals for all configured FX pairs.

        Args:
            data: DataFrame with [symbol, timestamp, close] for FX pairs.

        Returns:
            StrategyOutput with per-pair AlphaScores.
        """
        if not self._initialized:
            self.initialize(data)

        scores: List[AlphaScore] = []
        fx_signals: List[FXSignal] = []

        for pair in self.config.pairs:
            pair_data = data[data["symbol"] == pair].sort_values("timestamp")

            if len(pair_data) < self.config.trend_lookback_days:
                logger.debug(
                    "Skipping %s: insufficient data (%d days)", pair, len(pair_data)
                )
                continue

            # Compute carry score
            carry = self._compute_carry(pair)

            # Compute trend score
            trend = self._compute_trend(pair_data)

            # Combine
            combined = (
                self.config.carry_weight * carry + self.config.trend_weight * trend
            )

            direction = "long" if combined > 0 else "short"

            fx_signals.append(
                FXSignal(
                    pair=pair,
                    carry_score=carry,
                    trend_score=trend,
                    combined_score=combined,
                    direction=direction,
                )
            )

        if not fx_signals:
            return self._empty_output()

        # Z-score the combined signals across pairs
        raw_scores = np.array([s.combined_score for s in fx_signals])
        mean = np.mean(raw_scores)
        std = np.std(raw_scores)

        for sig in fx_signals:
            if std > 1e-10:
                z = (sig.combined_score - mean) / std
            else:
                z = 0.0

            confidence = min(abs(z) / 2.0, 1.0)

            if z > 0.3:
                direction = SignalDirection.LONG
            elif z < -0.3:
                direction = SignalDirection.SHORT
            else:
                direction = SignalDirection.NEUTRAL

            scores.append(
                AlphaScore(
                    symbol=sig.pair,
                    score=z,
                    raw_score=sig.combined_score,
                    confidence=confidence,
                    direction=direction,
                    metadata={
                        "carry_score": round(sig.carry_score, 4),
                        "trend_score": round(sig.trend_score, 4),
                        "combined_raw": round(sig.combined_score, 4),
                    },
                )
            )

        # Keep top N long and short
        longs = sorted(
            [s for s in scores if s.direction == SignalDirection.LONG],
            key=lambda x: x.score,
            reverse=True,
        )[: self.config.max_pairs_long]

        shorts = sorted(
            [s for s in scores if s.direction == SignalDirection.SHORT],
            key=lambda x: x.score,
        )[: self.config.max_pairs_short]

        final_scores = longs + shorts

        logger.info(
            "%s: %d signals (%d long, %d short)",
            self.name,
            len(final_scores),
            len(longs),
            len(shorts),
        )

        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=final_scores,
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
            metadata={
                "pairs_evaluated": len(fx_signals),
                "long_count": len(longs),
                "short_count": len(shorts),
                "carry_weight": self.config.carry_weight,
                "trend_weight": self.config.trend_weight,
            },
        )

    def get_performance(self) -> StrategyPerformance:
        return self._performance

    def update_interest_rates(self, rates: Dict[str, float]) -> None:
        """Update interest rates (call monthly or when central banks act)."""
        self.rates = rates
        logger.info("Interest rates updated: %s", rates)

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_carry(self, pair: str) -> float:
        """
        Carry score = interest rate differential.

        For EURUSD: carry = EUR rate - USD rate.
        Positive carry means the base currency yields more → long signal.

        Normalized to roughly [-1, 1] range by dividing by 5
        (max realistic rate differential).
        """
        currencies = PAIR_CURRENCIES.get(pair)
        if currencies is None:
            return 0.0

        base_ccy, quote_ccy = currencies
        base_rate = self.rates.get(base_ccy, 0.0)
        quote_rate = self.rates.get(quote_ccy, 0.0)

        # Rate differential in percent
        diff = base_rate - quote_rate

        # Normalize: 5% diff → score of 1.0
        normalized = diff / 5.0

        return normalized

    def _compute_trend(self, pair_data: pd.DataFrame) -> float:
        """
        Trend score = price momentum over lookback period, z-scored against
        its own history.

        Uses the ratio: current price / lookback-period-ago price - 1
        Then z-scores against the rolling distribution of this ratio.
        """
        close = pair_data["close"].values
        lookback = self.config.trend_lookback_days

        if len(close) < lookback + 20:
            return 0.0

        # Current momentum
        current_momentum = (close[-1] / close[-lookback]) - 1.0

        # Rolling momentum series for z-scoring
        momentum_series = []
        for i in range(lookback, len(close)):
            m = (close[i] / close[i - lookback]) - 1.0
            momentum_series.append(m)

        if len(momentum_series) < 10:
            return 0.0

        arr = np.array(momentum_series)
        mean = np.mean(arr)
        std = np.std(arr)

        if std > 1e-10:
            z = (current_momentum - mean) / std
        else:
            z = 0.0

        # Clip to [-3, 3] to avoid extreme values
        return float(np.clip(z, -3.0, 3.0))

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Check data has the required columns and pairs."""
        required = {"symbol", "timestamp", "close"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        available = set(data["symbol"].unique())
        configured = set(self.config.pairs)
        found = available & configured

        if not found:
            logger.warning(
                "No configured FX pairs found in data. "
                "Available: %s, Configured: %s",
                available,
                configured,
            )

    def _empty_output(self) -> StrategyOutput:
        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=[],
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
        )
