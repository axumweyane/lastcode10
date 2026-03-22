"""
Bayesian signal combiner — the brain of the APEX multi-strategy system.

Combines alpha scores from all active strategies (TFT, momentum, pairs, etc.)
into a single per-symbol composite alpha using regime-adaptive Bayesian
weighting.

Weighting methods:
    "equal"    — each strategy gets 1/N weight (baseline)
    "sharpe"   — weight proportional to rolling Sharpe ratio
    "bayesian" — exponentially-weighted rolling Sharpe with prior, clamped
                 to [min_weight, max_weight], adjusted by regime detector

The key insight: if five strategies each have Sharpe 0.8 and pairwise
correlation ~0.2, the combined Sharpe is ~2.0. This module captures that
diversification benefit.

TFT integration:
    The existing TFT predictions (from stock_ranking.py) are adapted into
    the same StrategyOutput format via TFTAdapter, so the TFT model
    participates in the ensemble as just another alpha source.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base import (
    AlphaScore,
    SignalDirection,
    StrategyOutput,
    StrategyPerformance,
)
from strategies.config import EnsembleConfig
from strategies.ensemble.bayesian_updater import BayesianWeightUpdater
from strategies.regime.detector import RegimeDetector, RegimeState

logger = logging.getLogger(__name__)


@dataclass
class StrategyWeight:
    """Weight for one strategy in the ensemble."""

    strategy_name: str
    raw_weight: float  # from Sharpe-based calculation
    regime_weight: float  # from regime detector
    final_weight: float  # blended and clamped
    sharpe_63d: float
    sharpe_21d: float


@dataclass
class CombinedSignal:
    """Final per-symbol signal after combining all strategies."""

    symbol: str
    combined_score: float
    confidence: float
    direction: SignalDirection
    contributing_strategies: Dict[str, float]  # {strategy_name: weighted_score}


class EnsembleCombiner:
    """
    Combines multiple strategy outputs into unified alpha scores.

    Usage:
        combiner = EnsembleCombiner(config, regime_detector)
        combined = combiner.combine([tft_output, momentum_output, pairs_output])
        # combined is a list of CombinedSignal
    """

    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
        regime_detector: Optional[RegimeDetector] = None,
        bayesian_updater: Optional[BayesianWeightUpdater] = None,
    ):
        self.config = config or EnsembleConfig.from_env()
        self.regime_detector = regime_detector
        self.bayesian_updater = bayesian_updater
        self._performance_history: Dict[str, StrategyPerformance] = {}
        self._weight_history: List[Dict[str, StrategyWeight]] = []

    def combine(
        self,
        strategy_outputs: List[StrategyOutput],
        regime_state: Optional[RegimeState] = None,
    ) -> List[CombinedSignal]:
        """
        Combine alpha scores from all strategies.

        Args:
            strategy_outputs: List of StrategyOutput from each active strategy.
            regime_state: Current market regime (if None, uses equal weighting
                         for the regime component).

        Returns:
            List of CombinedSignal, sorted by absolute combined score descending.
        """
        if not strategy_outputs:
            return []

        # 1. Compute strategy weights
        weights = self._compute_weights(strategy_outputs, regime_state)
        self._weight_history.append(weights)

        # Log weights
        for name, w in weights.items():
            logger.info(
                "Strategy weight: %s = %.3f (sharpe_63d=%.2f, regime=%.3f)",
                name,
                w.final_weight,
                w.sharpe_63d,
                w.regime_weight,
            )

        # 2. Collect all symbols across all strategies
        all_symbols = set()
        strategy_scores: Dict[str, pd.DataFrame] = {}

        for output in strategy_outputs:
            df = output.to_dataframe()
            if df.empty:
                continue
            strategy_scores[output.strategy_name] = df
            all_symbols.update(df["symbol"].tolist())

        if not all_symbols:
            return []

        # 3. Combine scores per symbol
        combined: List[CombinedSignal] = []

        for symbol in all_symbols:
            weighted_score = 0.0
            total_weight = 0.0
            contributions: Dict[str, float] = {}

            for strat_name, df in strategy_scores.items():
                sym_row = df[df["symbol"] == symbol]
                if sym_row.empty:
                    continue

                score = float(sym_row["score"].iloc[0])
                confidence = float(sym_row["confidence"].iloc[0])

                w = weights.get(strat_name)
                if w is None:
                    continue

                # Weight = strategy weight * per-signal confidence
                effective_weight = w.final_weight * confidence
                weighted_contribution = score * effective_weight

                weighted_score += weighted_contribution
                total_weight += effective_weight
                contributions[strat_name] = round(weighted_contribution, 6)

            # Normalize by total weight to keep scale consistent
            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = 0.0

            # Direction and confidence
            if abs(final_score) < 0.1:
                direction = SignalDirection.NEUTRAL
            elif final_score > 0:
                direction = SignalDirection.LONG
            else:
                direction = SignalDirection.SHORT

            combined_confidence = min(abs(final_score) / 2.0, 1.0)

            combined.append(
                CombinedSignal(
                    symbol=symbol,
                    combined_score=final_score,
                    confidence=combined_confidence,
                    direction=direction,
                    contributing_strategies=contributions,
                )
            )

        # Sort by absolute score descending
        combined.sort(key=lambda s: abs(s.combined_score), reverse=True)

        # Cap at max positions
        max_pos = self.config.max_total_positions
        longs = [s for s in combined if s.direction == SignalDirection.LONG][
            : max_pos // 2
        ]
        shorts = [s for s in combined if s.direction == SignalDirection.SHORT][
            : max_pos // 2
        ]

        result = longs + shorts
        result.sort(key=lambda s: abs(s.combined_score), reverse=True)

        logger.info(
            "Ensemble: %d symbols evaluated, %d signals (%d long, %d short)",
            len(all_symbols),
            len(result),
            len(longs),
            len(shorts),
        )

        return result

    def _compute_weights(
        self,
        strategy_outputs: List[StrategyOutput],
        regime_state: Optional[RegimeState],
    ) -> Dict[str, StrategyWeight]:
        """
        Compute final weight per strategy.

        Blends performance-based weight (from rolling Sharpe) with
        regime-based weight, then clamps to [min_weight, max_weight].
        """
        method = self.config.weighting_method
        strategies = {o.strategy_name: o for o in strategy_outputs}

        if method == "equal":
            return self._equal_weights(strategies, regime_state)
        elif method == "sharpe":
            return self._sharpe_weights(strategies, regime_state)
        elif method == "bayesian":
            return self._bayesian_weights(strategies, regime_state)
        else:
            logger.warning("Unknown weighting method '%s', using equal", method)
            return self._equal_weights(strategies, regime_state)

    def _equal_weights(
        self,
        strategies: Dict[str, StrategyOutput],
        regime_state: Optional[RegimeState],
    ) -> Dict[str, StrategyWeight]:
        n = len(strategies)
        if n == 0:
            return {}
        equal_w = 1.0 / n
        result = {}
        for name, output in strategies.items():
            regime_w = self._get_regime_weight(name, regime_state)
            result[name] = StrategyWeight(
                strategy_name=name,
                raw_weight=equal_w,
                regime_weight=regime_w,
                final_weight=equal_w,
                sharpe_63d=output.strategy_sharpe_63d,
                sharpe_21d=output.strategy_sharpe_21d,
            )
        return result

    def _sharpe_weights(
        self,
        strategies: Dict[str, StrategyOutput],
        regime_state: Optional[RegimeState],
    ) -> Dict[str, StrategyWeight]:
        """Weight proportional to max(rolling Sharpe, 0)."""
        raw_sharpes = {}
        for name, output in strategies.items():
            # Use 63-day Sharpe, floor at 0 (don't give negative-Sharpe
            # strategies negative weight — just minimize them)
            raw_sharpes[name] = max(output.strategy_sharpe_63d, 0.0)

        return self._normalize_weights(strategies, raw_sharpes, regime_state)

    def _bayesian_weights(
        self,
        strategies: Dict[str, StrategyOutput],
        regime_state: Optional[RegimeState],
    ) -> Dict[str, StrategyWeight]:
        """
        Bayesian weighting: prior = equal weight, updated by rolling Sharpe.

        The prior prevents new strategies from being over/under-weighted
        before we have enough data. As observations accumulate, the
        posterior converges toward pure Sharpe weighting.

        Weight_i ∝ prior_i + alpha * max(sharpe_63d_i, 0)

        alpha controls how fast we deviate from equal weights.
        With 63 days of data, alpha ≈ 1.0 (moderate update).
        """
        n = len(strategies)
        if n == 0:
            return {}

        prior = 1.0 / n  # uninformative equal prior
        alpha = 1.0  # update strength

        raw_weights = {}
        for name, output in strategies.items():
            sharpe = max(output.strategy_sharpe_63d, 0.0)
            raw_weights[name] = prior + alpha * sharpe

        return self._normalize_weights(strategies, raw_weights, regime_state)

    def _normalize_weights(
        self,
        strategies: Dict[str, StrategyOutput],
        raw_weights: Dict[str, float],
        regime_state: Optional[RegimeState],
    ) -> Dict[str, StrategyWeight]:
        """
        Normalize raw weights, blend with regime weights, clamp to bounds.

        When use_bayesian_updater is enabled and a bayesian_updater is provided,
        the 60% performance component uses Beta-Binomial weights instead of
        Sharpe-based weights.
        """
        total_raw = sum(raw_weights.values())
        if total_raw <= 0:
            total_raw = 1.0

        # If Bayesian updater is active, use its weights for the performance component
        use_updater = (
            self.config.use_bayesian_updater and self.bayesian_updater is not None
        )
        bayesian_weights = self.bayesian_updater.get_weights() if use_updater else {}

        result = {}
        for name, output in strategies.items():
            if use_updater and name in bayesian_weights:
                raw_w = bayesian_weights[name]
            else:
                raw_w = raw_weights.get(name, 0.0) / total_raw

            regime_w = self._get_regime_weight(name, regime_state)

            # Blend: 60% performance-based, 40% regime-based
            if regime_state is not None:
                blended = 0.6 * raw_w + 0.4 * regime_w
            else:
                blended = raw_w

            # Clamp
            clamped = max(self.config.min_weight, min(self.config.max_weight, blended))

            result[name] = StrategyWeight(
                strategy_name=name,
                raw_weight=raw_w,
                regime_weight=regime_w,
                final_weight=clamped,
                sharpe_63d=output.strategy_sharpe_63d,
                sharpe_21d=output.strategy_sharpe_21d,
            )

        # Re-normalize so weights sum to 1.0 after clamping
        total_final = sum(w.final_weight for w in result.values())
        if total_final > 0:
            for w in result.values():
                w.final_weight /= total_final

        return result

    def _get_regime_weight(
        self,
        strategy_name: str,
        regime_state: Optional[RegimeState],
    ) -> float:
        """Look up regime-recommended weight for a strategy."""
        if regime_state is None:
            return 0.25  # neutral default

        weights = regime_state.strategy_weights

        # Map strategy names to regime weight keys
        # Regime weight arrays are 4-dimensional: [momentum, mean_reversion, pairs, tft]
        key_map = {
            # Stocks
            "cross_sectional_momentum": "momentum",
            "momentum": "momentum",
            "pairs_trading": "pairs",
            "pairs": "pairs",
            "mean_reversion": "mean_reversion",
            "sector_rotation": "momentum",  # macro-driven, correlates with trend
            # Forex
            "fx_carry_trend": "tft",  # value-like strategy
            "fx_momentum": "momentum",  # trend-following
            "fx_vol_breakout": "pairs",  # event-driven, market-neutral-like
            # Options
            "deep_surrogates": "pairs",  # market-neutral
            "tdgf": "pairs",  # options pricing, market-neutral
            "vol_arb": "pairs",  # market-neutral
            # Cross-asset
            "kronos": "tft",  # forecasting model, weight like TFT
            "sentiment": "tft",  # sentiment-driven, weight like TFT
            # Adapters
            "tft": "tft",
            "tft_adapter": "tft",
        }

        regime_key = key_map.get(strategy_name, None)
        if regime_key and regime_key in weights:
            return weights[regime_key]

        # Unknown strategy: give average of existing weights
        if weights:
            return sum(weights.values()) / len(weights)
        return 0.25

    def get_weight_history(self, n: int = 30) -> List[Dict[str, StrategyWeight]]:
        """Return recent weight history for analysis."""
        return self._weight_history[-n:]


class TFTAdapter:
    """
    Adapts existing TFT predictions (from stock_ranking.py) into
    StrategyOutput format for the ensemble combiner.

    This bridges the existing APEX prediction pipeline with the new
    multi-strategy system without modifying any existing code.
    """

    STRATEGY_NAME = "tft_adapter"

    def __init__(self):
        self._performance = StrategyPerformance(strategy_name=self.STRATEGY_NAME)

    def adapt(
        self,
        predictions_df: pd.DataFrame,
        prediction_type: str = "quantile",
    ) -> StrategyOutput:
        """
        Convert TFT prediction DataFrame to StrategyOutput.

        Args:
            predictions_df: DataFrame from StockRankingSystem.process_predictions()
                           with columns: symbol, predicted_return, confidence
                           (and optionally lower_bound, upper_bound for quantile).
            prediction_type: "quantile", "point", or "classification".

        Returns:
            StrategyOutput compatible with the ensemble combiner.
        """
        scores: List[AlphaScore] = []

        for _, row in predictions_df.iterrows():
            symbol = row["symbol"]
            predicted_return = float(row["predicted_return"])

            # Confidence from quantile spread or default
            if "confidence" in row and pd.notna(row["confidence"]):
                raw_confidence = float(row["confidence"])
            else:
                raw_confidence = 0.5

            # Z-score the predicted return across the cross-section
            # (done below after collecting all)
            scores.append(
                AlphaScore(
                    symbol=symbol,
                    score=predicted_return,  # will be z-scored below
                    raw_score=predicted_return,
                    confidence=min(max(raw_confidence, 0.0), 1.0),
                    direction=SignalDirection.NEUTRAL,  # set below
                    metadata={
                        "prediction_type": prediction_type,
                        "lower_bound": float(row.get("lower_bound", 0)),
                        "upper_bound": float(row.get("upper_bound", 0)),
                    },
                )
            )

        # Z-score the raw predictions cross-sectionally
        if scores:
            raw_values = np.array([s.score for s in scores])
            mean = np.mean(raw_values)
            std = np.std(raw_values)

            for s in scores:
                if std > 1e-10:
                    s.score = (s.score - mean) / std
                else:
                    s.score = 0.0

                # Set direction based on z-score
                if s.score > 0.5:
                    s.direction = SignalDirection.LONG
                elif s.score < -0.5:
                    s.direction = SignalDirection.SHORT
                else:
                    s.direction = SignalDirection.NEUTRAL

                # Adjust confidence by z-score magnitude
                s.confidence = min(abs(s.score) / 3.0 + s.confidence * 0.5, 1.0)

        return StrategyOutput(
            strategy_name=self.STRATEGY_NAME,
            timestamp=datetime.now(timezone.utc),
            scores=scores,
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
            metadata={
                "prediction_type": prediction_type,
                "symbol_count": len(scores),
            },
        )

    def get_performance(self) -> StrategyPerformance:
        return self._performance
