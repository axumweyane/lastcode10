"""
TDGF Strategy — American option pricing via Time Deep Gradient Flow PDE solver.

Strategy #14: Generates alpha from option mispricing — the difference between
the TDGF-computed theoretical price and the observed market price.

Alpha logic:
  - Positive mispricing (model > market) → option underpriced → LONG
  - Negative mispricing (model < market) → option overpriced → SHORT

Confidence is derived from two sources:
  1. PDE solution quality (lower residual = more trustworthy price)
  2. Prediction interval width (tighter bounds = higher confidence)

When both TDGF and Deep Surrogates agree on direction for the same
underlying, the signal is stronger. When they disagree, the Risk
Validator should flag it for review.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models.base import ModelPrediction
from models.manager import ModelManager
from strategies.base import (
    AlphaScore,
    BaseStrategy,
    SignalDirection,
    StrategyOutput,
    StrategyPerformance,
)
from strategies.config import TDGFConfig

logger = logging.getLogger(__name__)


class TDGFStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "tdgf"

    @property
    def description(self) -> str:
        return "American option pricing via TDGF PDE solver"

    def __init__(
        self,
        config: Optional[TDGFConfig] = None,
        manager: Optional[ModelManager] = None,
    ):
        self.config = config or TDGFConfig.from_env()
        self._manager = manager
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._initialized = False
        self._latest_mispricings: Dict[str, float] = {}

    def initialize(self, historical_data: pd.DataFrame) -> None:
        if self._manager is not None and self._manager.is_model_loaded("tdgf"):
            self._initialized = True
            logger.info("TDGFStrategy initialized (pde_model=%s)", self.config.pde_model)
        else:
            logger.info("TDGF model not loaded — strategy will return empty signals")

    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        if not self._initialized:
            self.initialize(data)

        if self._manager is None or not self._manager.is_model_loaded("tdgf"):
            return self._empty_output()

        if data.empty:
            return self._empty_output()

        try:
            predictions = self._manager.predict_tdgf(data)
            if not predictions:
                return self._empty_output()

            scores: List[AlphaScore] = []

            for pred in predictions:
                tdgf_price = pred.metadata.get("tdgf_price", 0.0)
                market_price = pred.metadata.get("market_price", 0.0)

                # Mispricing is already computed by the model as predicted_value
                raw_score = pred.predicted_value
                self._latest_mispricings[pred.symbol] = raw_score

                # Confidence from two sources:
                # 1. PDE residual quality (lower = better)
                pde_residual = pred.metadata.get("pde_residual",
                                                  pred.metadata.get("pde_residual", 0.5))
                pde_confidence = max(0.1, 1.0 - pde_residual)

                # 2. Prediction bounds width (tighter = better)
                spread = abs(pred.upper_bound - pred.lower_bound)
                spread_confidence = max(0.1, min(0.95, 1.0 - spread * 5))

                # Geometric mean of both confidence sources
                confidence = min(0.95, (pde_confidence * spread_confidence) ** 0.5)

                # Collect Greeks for downstream consumers
                greeks = {
                    k: v for k, v in pred.metadata.items()
                    if k.startswith("greek_")
                }

                scores.append(AlphaScore(
                    symbol=pred.symbol,
                    score=raw_score,
                    raw_score=raw_score,
                    confidence=confidence,
                    direction=SignalDirection.NEUTRAL,  # set after z-scoring
                    metadata={
                        "tdgf_price": tdgf_price,
                        "market_price": market_price,
                        "mispricing_pct": raw_score,
                        "pde_model": pred.metadata.get("pde_model", self.config.pde_model),
                        "asset_class": "options",
                        **greeks,
                    },
                ))

            # Z-score cross-sectionally, assign direction post-normalization
            if scores:
                raw_values = np.array([s.score for s in scores])
                mean = float(np.mean(raw_values))
                std = float(np.std(raw_values))

                for s in scores:
                    if std > 1e-6:
                        s.score = (s.score - mean) / std
                    else:
                        s.score = 0.0

                    if s.score > 0.5:
                        s.direction = SignalDirection.LONG
                    elif s.score < -0.5:
                        s.direction = SignalDirection.SHORT
                    else:
                        s.direction = SignalDirection.NEUTRAL

                    # Blend z-magnitude with model-derived confidence
                    z_conf = min(abs(s.score) / 3.0, 0.5)
                    s.confidence = min(z_conf + s.confidence * 0.5, 1.0)

            # Summary stats for diagnostics
            long_count = sum(1 for s in scores if s.direction == SignalDirection.LONG)
            short_count = sum(1 for s in scores if s.direction == SignalDirection.SHORT)

            logger.info(
                "TDGF: %d signals (%d long, %d short, pde=%s)",
                len(scores), long_count, short_count, self.config.pde_model,
            )

            return StrategyOutput(
                strategy_name=self.name,
                timestamp=datetime.now(timezone.utc),
                scores=scores,
                strategy_sharpe_63d=self._performance.sharpe_63d,
                strategy_sharpe_21d=self._performance.sharpe_21d,
                metadata={
                    "pde_model": self.config.pde_model,
                    "long_count": long_count,
                    "short_count": short_count,
                    "avg_abs_mispricing": float(np.mean(np.abs(
                        [s.raw_score for s in scores]
                    ))) if scores else 0.0,
                },
            )

        except Exception as e:
            logger.error("TDGFStrategy.generate_signals failed: %s", e, exc_info=True)
            return self._empty_output()

    def get_mispricing(self, symbol: str) -> Optional[float]:
        """Return latest mispricing for a symbol (for cross-validation)."""
        return self._latest_mispricings.get(symbol)

    def get_performance(self) -> StrategyPerformance:
        return self._performance

    def _empty_output(self) -> StrategyOutput:
        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=[],
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
        )
