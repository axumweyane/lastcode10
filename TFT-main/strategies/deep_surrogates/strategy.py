"""
Deep Surrogates Strategy — Neural surrogate option pricing with tail risk index.

Strategy #13: Converts DeepSurrogateModel outputs into ensemble-compatible
alpha scores. Produces two independent signal types:

  1. Tail risk signals: High tail risk from Heston parameter evolution
     maps to SHORT alpha (crash protection). Score = -tail_risk_index.

  2. IV surface signals: ATM IV anomalies map to mean-reversion alpha.
     High IV → SHORT (sell vol), low IV → LONG (buy vol).

Both signal types are z-scored cross-sectionally before delivery to
the ensemble combiner.

Also exposes `get_tail_risk_index()` for the Risk Validator to consume
directly (bypassing the alpha path) for portfolio-level crash warnings.
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
from strategies.config import DeepSurrogateConfig

logger = logging.getLogger(__name__)


def _zscore_scores(scores: List[AlphaScore]) -> None:
    """Z-score alpha scores in place and assign direction + blended confidence."""
    if not scores:
        return

    raw_values = np.array([s.score for s in scores])
    mean = float(np.mean(raw_values))
    std = float(np.std(raw_values))

    for s in scores:
        if std > 1e-6:
            s.score = (s.score - mean) / std
        else:
            s.score = 0.0

        # Direction from z-scored value (post normalization, not pre)
        if s.score > 0.5:
            s.direction = SignalDirection.LONG
        elif s.score < -0.5:
            s.direction = SignalDirection.SHORT
        else:
            s.direction = SignalDirection.NEUTRAL

        # Blend z-magnitude confidence with model confidence
        z_conf = min(abs(s.score) / 3.0, 0.5)
        model_conf = s.confidence * 0.5
        s.confidence = min(z_conf + model_conf, 1.0)


class DeepSurrogateStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "deep_surrogates"

    @property
    def description(self) -> str:
        return "Neural surrogate option pricing with tail risk index"

    def __init__(
        self,
        config: Optional[DeepSurrogateConfig] = None,
        manager: Optional[ModelManager] = None,
    ):
        self.config = config or DeepSurrogateConfig.from_env()
        self._manager = manager
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._initialized = False
        self._latest_tail_risk: Dict[str, float] = {}
        self._composite_tail_risk: float = 0.0

    def initialize(self, historical_data: pd.DataFrame) -> None:
        if self._manager is not None and self._manager.is_model_loaded(
            "deep_surrogates"
        ):
            self._initialized = True
            logger.info("DeepSurrogateStrategy initialized")
        else:
            logger.info(
                "DeepSurrogate model not loaded — strategy will return empty signals"
            )

    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        if not self._initialized:
            self.initialize(data)

        if self._manager is None or not self._manager.is_model_loaded(
            "deep_surrogates"
        ):
            return self._empty_output()

        if data.empty:
            return self._empty_output()

        try:
            predictions = self._manager.predict_deep_surrogates(data)
            if not predictions:
                return self._empty_output()

            # Separate signal types for cleaner processing
            tail_risk_preds: List[ModelPrediction] = []
            iv_surface_preds: List[ModelPrediction] = []

            for pred in predictions:
                sig_type = pred.metadata.get("signal_type", "")
                if sig_type == "tail_risk":
                    tail_risk_preds.append(pred)
                elif sig_type == "iv_surface":
                    iv_surface_preds.append(pred)
                else:
                    # Unknown signal type — treat as tail risk if it has the field
                    if pred.metadata.get("tail_risk_index") is not None:
                        tail_risk_preds.append(pred)
                    else:
                        iv_surface_preds.append(pred)

            scores: List[AlphaScore] = []

            # Process tail risk signals
            for pred in tail_risk_preds:
                tail_risk = pred.metadata.get("tail_risk_index", 0.0)
                self._latest_tail_risk[pred.symbol] = tail_risk

                # Higher tail risk = more crash risk = bearish alpha
                raw_score = -tail_risk
                scores.append(
                    AlphaScore(
                        symbol=pred.symbol,
                        score=raw_score,
                        raw_score=raw_score,
                        confidence=min(max(pred.confidence, 0.0), 1.0),
                        direction=SignalDirection.NEUTRAL,  # set after z-scoring
                        metadata={
                            "signal_type": "tail_risk",
                            "tail_risk_index": tail_risk,
                            "heston_sigma": pred.metadata.get("heston_sigma", 0),
                            "heston_rho": pred.metadata.get("heston_rho", 0),
                            "asset_class": "options",
                        },
                    )
                )

            # Process IV surface signals
            for pred in iv_surface_preds:
                raw_score = pred.predicted_value  # already directional from model
                scores.append(
                    AlphaScore(
                        symbol=pred.symbol,
                        score=raw_score,
                        raw_score=raw_score,
                        confidence=min(max(pred.confidence, 0.0), 1.0),
                        direction=SignalDirection.NEUTRAL,
                        metadata={
                            "signal_type": "iv_surface",
                            "atm_iv": pred.metadata.get("atm_iv", 0),
                            "asset_class": "options",
                        },
                    )
                )

            # Z-score cross-sectionally (direction assigned AFTER normalization)
            _zscore_scores(scores)

            # Update composite tail risk for risk validator
            if self._latest_tail_risk:
                self._composite_tail_risk = float(
                    np.mean(list(self._latest_tail_risk.values()))
                )

            # Check alert threshold
            if (
                self.config.tail_risk_enabled
                and self._composite_tail_risk >= self.config.tail_risk_alert_threshold
            ):
                logger.warning(
                    "TAIL RISK ALERT: composite=%.3f (threshold=%.2f) — %s",
                    self._composite_tail_risk,
                    self.config.tail_risk_alert_threshold,
                    self._latest_tail_risk,
                )

            logger.info(
                "DeepSurrogates: %d signals (%d tail_risk, %d iv_surface), "
                "composite_risk=%.3f",
                len(scores),
                len(tail_risk_preds),
                len(iv_surface_preds),
                self._composite_tail_risk,
            )

            return StrategyOutput(
                strategy_name=self.name,
                timestamp=datetime.now(timezone.utc),
                scores=scores,
                strategy_sharpe_63d=self._performance.sharpe_63d,
                strategy_sharpe_21d=self._performance.sharpe_21d,
                metadata={
                    "composite_tail_risk": self._composite_tail_risk,
                    "tail_risk_symbols": len(tail_risk_preds),
                    "iv_surface_symbols": len(iv_surface_preds),
                },
            )

        except Exception as e:
            logger.error(
                "DeepSurrogateStrategy.generate_signals failed: %s", e, exc_info=True
            )
            return self._empty_output()

    def get_tail_risk_index(self, symbol: str) -> Optional[float]:
        """Return the latest tail risk index for a symbol (for Risk Validator)."""
        return self._latest_tail_risk.get(symbol)

    def get_composite_tail_risk(self) -> float:
        """Portfolio-level composite tail risk (mean across all underlyings)."""
        return self._composite_tail_risk

    def get_all_tail_risk(self) -> Dict[str, float]:
        """All per-symbol tail risk values (for SignalPublisher)."""
        return dict(self._latest_tail_risk)

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
