"""
Market regime classifier — 4-state model.

States:
    CALM_TRENDING      — low vol, broad participation → momentum thrives
    CALM_CHOPPY        — low vol, narrow breadth      → mean reversion thrives
    VOLATILE_TRENDING  — high vol, directional move   → reduced size, favor trend
    VOLATILE_CHOPPY    — high vol, no direction        → defensive, favor pairs

Inputs:
    1. VIX level (or realized vol proxy if VIX unavailable)
    2. Market breadth: % of stocks above their 50-day MA
    3. Realized volatility of the broad market (e.g., SPY)

The regime drives two things:
    - Strategy weight allocation in the ensemble combiner
    - Gross exposure scaling (reduce in volatile regimes)

Design choice: deterministic thresholds, not hidden Markov. HMMs are elegant
but add latency, require fitting, and produce ambiguous states during
transitions. Threshold-based is transparent, instant, and easier to debug
in production.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.config import RegimeConfig

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    CALM_TRENDING = "calm_trending"
    CALM_CHOPPY = "calm_choppy"
    VOLATILE_TRENDING = "volatile_trending"
    VOLATILE_CHOPPY = "volatile_choppy"


@dataclass
class RegimeState:
    """Current regime classification with diagnostics."""
    regime: MarketRegime
    vix_level: float
    market_breadth: float        # fraction of stocks above 50-day MA
    realized_vol: float          # annualized realized vol of market proxy
    is_volatile: bool
    is_trending: bool
    confidence: float            # 0-1, how clearly we're in this regime
    strategy_weights: Dict[str, float]  # recommended weights per strategy
    exposure_scalar: float       # 0-1, scale gross exposure in volatile regimes

    def __str__(self) -> str:
        return (
            f"Regime: {self.regime.value} | VIX={self.vix_level:.1f} "
            f"Breadth={self.market_breadth:.1%} RVol={self.realized_vol:.1%} "
            f"Exposure={self.exposure_scalar:.0%}"
        )


class RegimeDetector:
    """
    Classifies the current market regime and outputs strategy weights.

    Usage:
        detector = RegimeDetector(config)
        state = detector.detect(market_data)
        weights = state.strategy_weights  # feed into ensemble combiner
    """

    # Strategy key ordering for weight arrays in config:
    # [momentum, mean_reversion, pairs, tft]
    STRATEGY_KEYS = ["momentum", "mean_reversion", "pairs", "tft"]

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig.from_env()
        self._history: List[RegimeState] = []

    def detect(
        self,
        market_data: pd.DataFrame,
        vix_value: Optional[float] = None,
    ) -> RegimeState:
        """
        Classify the current market regime.

        Args:
            market_data: DataFrame with [symbol, timestamp, close, volume].
                         Must include a broad set of stocks for breadth calc.
                         If a 'vix' column exists, it will be used.
            vix_value: Explicit VIX level. Overrides column-based lookup.

        Returns:
            RegimeState with classification and recommended strategy weights.
        """
        # 1. Determine VIX level
        vix = self._get_vix(market_data, vix_value)

        # 2. Compute market breadth
        breadth = self._compute_breadth(market_data)

        # 3. Compute realized volatility of market proxy
        realized_vol = self._compute_realized_vol(market_data)

        # 4. Classify
        is_volatile = vix > self.config.vix_high_threshold or realized_vol > 0.25
        is_trending = breadth > self.config.breadth_trending_threshold

        if not is_volatile and is_trending:
            regime = MarketRegime.CALM_TRENDING
        elif not is_volatile and not is_trending:
            regime = MarketRegime.CALM_CHOPPY
        elif is_volatile and is_trending:
            regime = MarketRegime.VOLATILE_TRENDING
        else:
            regime = MarketRegime.VOLATILE_CHOPPY

        # 5. Confidence: distance from thresholds (further = more confident)
        confidence = self._compute_confidence(vix, breadth, realized_vol)

        # 6. Strategy weights from config
        strategy_weights = self._get_weights(regime)

        # 7. Exposure scalar: reduce in volatile regimes
        exposure_scalar = self._compute_exposure_scalar(vix, realized_vol)

        state = RegimeState(
            regime=regime,
            vix_level=vix,
            market_breadth=breadth,
            realized_vol=realized_vol,
            is_volatile=is_volatile,
            is_trending=is_trending,
            confidence=confidence,
            strategy_weights=strategy_weights,
            exposure_scalar=exposure_scalar,
        )

        self._history.append(state)

        # Log regime changes
        if len(self._history) >= 2 and self._history[-2].regime != regime:
            logger.info(
                "REGIME CHANGE: %s -> %s",
                self._history[-2].regime.value, regime.value,
            )
        logger.info(str(state))

        return state

    def get_regime_history(self, n: int = 30) -> List[RegimeState]:
        """Return recent regime history for analysis."""
        return self._history[-n:]

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def _get_vix(
        self, data: pd.DataFrame, explicit_vix: Optional[float],
    ) -> float:
        """
        Get VIX level from: explicit value > column in data > realized vol proxy.
        """
        if explicit_vix is not None:
            return explicit_vix

        # Check for VIX column in data
        if "vix" in data.columns:
            vix_series = data["vix"].dropna()
            if not vix_series.empty:
                return float(vix_series.iloc[-1])

        # Check for ^VIX or VIX symbol in data
        for vix_sym in ["^VIX", "VIX", "VIXY"]:
            vix_data = data[data["symbol"] == vix_sym]
            if not vix_data.empty:
                return float(vix_data.sort_values("timestamp")["close"].iloc[-1])

        # Fallback: estimate from realized vol (very rough: rvol * 100 ≈ VIX)
        rvol = self._compute_realized_vol(data)
        estimated_vix = rvol * 100
        logger.debug("VIX unavailable, estimated from realized vol: %.1f", estimated_vix)
        return estimated_vix

    def _compute_breadth(self, data: pd.DataFrame) -> float:
        """
        Market breadth: fraction of stocks trading above their 50-day MA.

        High breadth (>60%) = broad participation = trending market.
        Low breadth (<40%) = narrow/divergent = choppy market.
        """
        lookback = self.config.breadth_lookback_days
        above_count = 0
        total_count = 0

        for symbol, group in data.groupby("symbol"):
            group = group.sort_values("timestamp")
            close = group["close"]

            if len(close) < lookback:
                continue

            ma = close.rolling(lookback, min_periods=lookback).mean()
            latest_close = close.iloc[-1]
            latest_ma = ma.iloc[-1]

            if pd.notna(latest_ma):
                total_count += 1
                if latest_close > latest_ma:
                    above_count += 1

        if total_count == 0:
            return 0.5  # neutral default

        breadth = above_count / total_count
        return breadth

    def _compute_realized_vol(self, data: pd.DataFrame) -> float:
        """
        Realized volatility of the market proxy.

        Uses the equal-weighted average of all stocks' realized vol as a proxy.
        If SPY is in the universe, uses that preferentially.
        """
        window = self.config.realized_vol_window

        # Try SPY first
        for proxy in ["SPY", "^GSPC", "QQQ"]:
            proxy_data = data[data["symbol"] == proxy]
            if len(proxy_data) >= window:
                returns = proxy_data.sort_values("timestamp")["close"].pct_change()
                rvol = returns.tail(window).std() * np.sqrt(252)
                if pd.notna(rvol):
                    return float(rvol)

        # Fallback: equal-weighted average across all symbols
        rvols = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_values("timestamp")
            if len(group) < window:
                continue
            ret = group["close"].pct_change()
            rv = ret.tail(window).std() * np.sqrt(252)
            if pd.notna(rv):
                rvols.append(rv)

        if rvols:
            return float(np.median(rvols))

        return 0.15  # default moderate vol

    def _compute_confidence(
        self, vix: float, breadth: float, realized_vol: float,
    ) -> float:
        """
        Regime confidence: how clearly we're in the classified state.

        High confidence = far from regime boundaries.
        Low confidence = near a threshold crossover.
        """
        # Distance from VIX threshold
        vix_mid = (self.config.vix_low_threshold + self.config.vix_high_threshold) / 2
        vix_dist = abs(vix - vix_mid) / vix_mid

        # Distance from breadth midpoint
        breadth_mid = (
            self.config.breadth_trending_threshold
            + self.config.breadth_choppy_threshold
        ) / 2
        breadth_dist = abs(breadth - breadth_mid) / max(breadth_mid, 0.01)

        # Combined confidence (0-1 range)
        raw = (vix_dist + breadth_dist) / 2
        return min(max(raw, 0.0), 1.0)

    def _get_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Map regime to strategy weight dict."""
        weight_map = {
            MarketRegime.CALM_TRENDING: self.config.calm_trending_weights,
            MarketRegime.CALM_CHOPPY: self.config.calm_choppy_weights,
            MarketRegime.VOLATILE_TRENDING: self.config.volatile_trending_weights,
            MarketRegime.VOLATILE_CHOPPY: self.config.volatile_choppy_weights,
        }
        weights = weight_map[regime]
        return dict(zip(self.STRATEGY_KEYS, weights))

    def _compute_exposure_scalar(
        self, vix: float, realized_vol: float,
    ) -> float:
        """
        Scale gross exposure inversely with volatility.

        In calm markets: full exposure (1.0).
        In volatile markets: reduce to 0.5-0.7 to control portfolio vol.

        Uses inverse vol targeting: scalar = target_vol / realized_vol,
        capped at [0.3, 1.0].
        """
        target_vol = 0.15  # 15% annualized target

        if realized_vol > 0.01:
            scalar = target_vol / realized_vol
        else:
            scalar = 1.0

        # VIX penalty: additional reduction when VIX is elevated
        if vix > self.config.vix_high_threshold:
            vix_penalty = 1.0 - min(
                (vix - self.config.vix_high_threshold) / 30.0, 0.3
            )
            scalar *= vix_penalty

        return min(max(scalar, 0.3), 1.0)
