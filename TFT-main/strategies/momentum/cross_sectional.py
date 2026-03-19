"""
Cross-Sectional Momentum + Mean Reversion Strategy.

Combines three orthogonal factors into a single composite alpha score:
  - 12-1 Momentum  (captures underreaction / trend continuation)
  - 5-day Reversal  (captures overreaction / mean reversion)
  - Quality          (captures profitability premium)

The relative weighting between momentum and mean-reversion adapts based on
the current volatility regime — momentum gets more weight in calm trending
markets, mean-reversion gets more weight in choppy/volatile markets.

Academic basis:
  - Jegadeesh & Titman (1993): Momentum profits
  - Jegadeesh (1990): Short-term reversal
  - Asness, Frazzini & Pedersen (2013): Quality minus Junk
  - Daniel & Moskowitz (2016): Momentum crashes and regime dependence
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from strategies.base import (
    AlphaScore,
    BaseStrategy,
    SignalDirection,
    StrategyOutput,
    StrategyPerformance,
)
from strategies.config import MomentumConfig
from strategies.momentum.features import compute_all_factors

logger = logging.getLogger(__name__)


class CrossSectionalMomentum(BaseStrategy):
    """
    Cross-sectional momentum + mean reversion factor strategy.

    Produces a per-symbol composite alpha z-score by combining momentum,
    mean-reversion, and quality factors. The ensemble combiner consumes
    these scores alongside TFT predictions and other strategy outputs.
    """

    def __init__(self, config: Optional[MomentumConfig] = None):
        self.config = config or MomentumConfig.from_env()
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._factor_data: Optional[pd.DataFrame] = None
        self._initialized = False

        # Regime-adaptive weights (can be overridden by regime detector)
        self._momentum_weight = self.config.momentum_weight
        self._meanrev_weight = self.config.mean_reversion_weight
        self._quality_weight = self.config.quality_weight

    @property
    def name(self) -> str:
        return "cross_sectional_momentum"

    @property
    def description(self) -> str:
        return (
            "Cross-sectional momentum + mean reversion + quality factor combination. "
            "Ranks stocks by composite z-score, goes long top decile, short bottom decile."
        )

    def initialize(self, historical_data: pd.DataFrame) -> None:
        """
        Compute all factors on the full historical dataset.

        Args:
            historical_data: DataFrame with [symbol, timestamp, open, high, low,
                            close, volume]. Must have >= min_history_days per symbol.
        """
        logger.info("Initializing %s with %d rows", self.name, len(historical_data))

        # Validate minimum data requirements
        symbol_counts = historical_data.groupby("symbol")["timestamp"].count()
        valid_symbols = symbol_counts[
            symbol_counts >= self.config.min_history_days
        ].index.tolist()

        if not valid_symbols:
            logger.warning(
                "No symbols have >= %d days of history. Strategy will produce no signals.",
                self.config.min_history_days,
            )
            self._initialized = True
            return

        filtered = historical_data[historical_data["symbol"].isin(valid_symbols)].copy()
        logger.info(
            "Computing factors for %d symbols (%d dropped for insufficient history)",
            len(valid_symbols),
            historical_data["symbol"].nunique() - len(valid_symbols),
        )

        self._factor_data = compute_all_factors(
            prices=filtered,
            momentum_lookback=self.config.momentum_lookback_days,
            momentum_skip=self.config.momentum_skip_days,
            meanrev_lookback=self.config.mean_reversion_lookback_days,
        )

        self._initialized = True
        logger.info("%s initialized successfully", self.name)

    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        """
        Generate alpha scores for the latest period.

        If called without initialize(), computes factors on the fly using `data`.
        For production use, call initialize() first with full history, then
        generate_signals() with the latest slice.

        Args:
            data: DataFrame with [symbol, timestamp, open, high, low, close, volume].

        Returns:
            StrategyOutput with per-symbol AlphaScores.
        """
        if not self._initialized:
            self.initialize(data)

        # Compute factors on the latest data
        factor_df = compute_all_factors(
            prices=data,
            momentum_lookback=self.config.momentum_lookback_days,
            momentum_skip=self.config.momentum_skip_days,
            meanrev_lookback=self.config.mean_reversion_lookback_days,
        )

        # Get the latest date's cross-section
        latest_date = factor_df["timestamp"].max()
        latest = factor_df[factor_df["timestamp"] == latest_date].copy()

        if latest.empty:
            logger.warning("No data for latest date")
            return self._empty_output()

        # Liquidity filter
        if "avg_dollar_volume" in latest.columns:
            latest = latest[
                latest["avg_dollar_volume"] >= self.config.min_avg_dollar_volume
            ].copy()

        if latest.empty:
            logger.warning("No symbols pass liquidity filter")
            return self._empty_output()

        # Compute composite score
        latest = self._compute_composite_score(latest)

        # Generate alpha scores
        scores = []
        for _, row in latest.iterrows():
            composite = row["composite_zscore"]

            if np.isnan(composite):
                continue

            # Determine direction
            if composite >= self.config.long_threshold_zscore:
                direction = SignalDirection.LONG
            elif composite <= self.config.short_threshold_zscore:
                direction = SignalDirection.SHORT
            else:
                direction = SignalDirection.NEUTRAL

            # Confidence: higher absolute z-score = higher confidence, capped at 1.0
            confidence = min(abs(composite) / 3.0, 1.0)

            scores.append(AlphaScore(
                symbol=row["symbol"],
                score=composite,
                raw_score=composite,
                confidence=confidence,
                direction=direction,
                metadata={
                    "momentum_z": _safe_float(row.get("momentum_zscore")),
                    "meanrev_z": _safe_float(row.get("meanrev_zscore")),
                    "quality_z": _safe_float(row.get("quality_zscore")),
                    "realized_vol": _safe_float(row.get("realized_vol")),
                    "avg_dollar_volume": _safe_float(row.get("avg_dollar_volume")),
                },
            ))

        # Sort by absolute score descending, keep top N per side
        longs = sorted(
            [s for s in scores if s.direction == SignalDirection.LONG],
            key=lambda x: x.score, reverse=True,
        )[:self.config.max_positions_per_side]

        shorts = sorted(
            [s for s in scores if s.direction == SignalDirection.SHORT],
            key=lambda x: x.score,
        )[:self.config.max_positions_per_side]

        final_scores = longs + shorts

        logger.info(
            "%s generated %d signals (%d long, %d short) for %s",
            self.name, len(final_scores), len(longs), len(shorts),
            latest_date.date() if hasattr(latest_date, "date") else latest_date,
        )

        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=final_scores,
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
            metadata={
                "date": str(latest_date),
                "symbols_evaluated": len(latest),
                "long_count": len(longs),
                "short_count": len(shorts),
                "momentum_weight": self._momentum_weight,
                "meanrev_weight": self._meanrev_weight,
                "quality_weight": self._quality_weight,
            },
        )

    def get_performance(self) -> StrategyPerformance:
        return self._performance

    def set_regime_weights(
        self,
        momentum_weight: float,
        meanrev_weight: float,
        quality_weight: float,
    ) -> None:
        """
        Allow the regime detector to dynamically adjust factor weights.

        Weights are normalized to sum to 1.0.
        """
        total = momentum_weight + meanrev_weight + quality_weight
        if total <= 0:
            logger.warning("Invalid regime weights (sum <= 0), keeping current weights")
            return

        self._momentum_weight = momentum_weight / total
        self._meanrev_weight = meanrev_weight / total
        self._quality_weight = quality_weight / total

        logger.info(
            "Regime weights updated: mom=%.2f, mr=%.2f, qual=%.2f",
            self._momentum_weight, self._meanrev_weight, self._quality_weight,
        )

    def _compute_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Weighted combination of factor z-scores into a single composite.

        The composite score is itself z-scored to ensure consistent scale
        for the downstream ensemble combiner.
        """
        df = df.copy()

        mom_z = df.get("momentum_zscore", pd.Series(0.0, index=df.index))
        mr_z = df.get("meanrev_zscore", pd.Series(0.0, index=df.index))
        qual_z = df.get("quality_zscore", pd.Series(0.0, index=df.index))

        # Fill NaN with 0 (neutral score)
        mom_z = mom_z.fillna(0.0)
        mr_z = mr_z.fillna(0.0)
        qual_z = qual_z.fillna(0.0)

        # Weighted sum
        composite = (
            self._momentum_weight * mom_z
            + self._meanrev_weight * mr_z
            + self._quality_weight * qual_z
        )

        # Z-score the composite itself for consistent scale
        mean = composite.mean()
        std = composite.std()
        if std > 1e-10:
            df["composite_zscore"] = (composite - mean) / std
        else:
            df["composite_zscore"] = 0.0

        return df

    def _empty_output(self) -> StrategyOutput:
        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=[],
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
        )


def _safe_float(val) -> float:
    """Safely convert to float for metadata, handling NaN/None."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if np.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return 0.0
