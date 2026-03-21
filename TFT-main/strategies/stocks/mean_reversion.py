"""
Mean Reversion Strategy — trade deviations from OU equilibrium.

Alpha source: Ornstein-Uhlenbeck deviation z-score, weighted by Hurst < 0.5
confidence. Anti-correlated with momentum (~-0.4) by design for maximum
ensemble diversification benefit.

Uses MeanReversionModel (#8) for regime detection and optionally
MicrostructureModel (#10) for timing.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from models.manager import ModelManager
from strategies.base import (
    AlphaScore,
    BaseStrategy,
    SignalDirection,
    StrategyOutput,
    StrategyPerformance,
)
from strategies.config import MeanReversionConfig

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):

    def __init__(
        self,
        config: Optional[MeanReversionConfig] = None,
        manager: Optional[ModelManager] = None,
    ):
        self.config = config or MeanReversionConfig.from_env()
        self._manager = manager
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._initialized = False

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def description(self) -> str:
        return "OU-based mean reversion: trade deviations from equilibrium weighted by Hurst exponent"

    def initialize(self, historical_data: pd.DataFrame) -> None:
        self._initialized = True

    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        if not self._initialized:
            self.initialize(data)

        scores: List[AlphaScore] = []

        # Get mean reversion model predictions
        mr_predictions = {}
        if self._manager is not None:
            try:
                preds = self._manager.predict_mean_reversion(data)
                mr_predictions = {p.symbol: p for p in preds}
            except Exception as e:
                logger.warning("MeanReversion model failed: %s", e)

        # Get microstructure predictions for timing (optional)
        micro_predictions = {}
        if self._manager is not None:
            try:
                preds = self._manager.predict_microstructure(data)
                micro_predictions = {p.symbol: p.metadata for p in preds}
            except Exception as e:
                logger.debug("Microstructure model not available: %s", e)

        if not mr_predictions:
            # Fallback: compute basic mean reversion signals directly
            mr_predictions = self._compute_basic_signals(data)

        for symbol, pred in mr_predictions.items():
            meta = pred.metadata if hasattr(pred, "metadata") else pred
            hurst = meta.get("hurst_exponent", 0.5)
            half_life = meta.get("half_life", float("inf"))
            deviation_zscore = meta.get("deviation_zscore", 0.0)

            # Filter: only trade when Hurst confirms mean reversion
            if hurst >= self.config.hurst_threshold:
                continue

            # Filter: half-life must be in tradeable range
            if not (self.config.min_half_life <= half_life <= self.config.max_half_life):
                continue

            # Filter: deviation must be significant
            if abs(deviation_zscore) < self.config.entry_zscore:
                continue

            # Signal: negative deviation_zscore → below equilibrium → LONG
            if deviation_zscore < -self.config.entry_zscore:
                direction = SignalDirection.LONG
                raw_score = -deviation_zscore  # positive for long
            elif deviation_zscore > self.config.entry_zscore:
                direction = SignalDirection.SHORT
                raw_score = -deviation_zscore  # negative for short
            else:
                continue

            # Confidence from Hurst quality and half-life
            hurst_quality = max(0.0, (0.5 - hurst) / 0.5)  # 0 at H=0.5, 1 at H=0
            half_life_quality = 1.0 if 2 <= half_life <= 30 else 0.5
            confidence = min(hurst_quality * half_life_quality * 0.8 + 0.2, 1.0)

            # Microstructure timing boost
            micro = micro_predictions.get(symbol, {})
            if micro:
                # If volume confirms (abnormal volume in reversion direction), boost confidence
                buying_pressure = micro.get("buying_pressure", 0.0)
                if (direction == SignalDirection.LONG and buying_pressure > 0.3) or \
                   (direction == SignalDirection.SHORT and buying_pressure < -0.3):
                    confidence = min(confidence + 0.1, 1.0)

            scores.append(AlphaScore(
                symbol=symbol,
                score=raw_score,
                raw_score=float(deviation_zscore),
                confidence=confidence,
                direction=direction,
                metadata={
                    "hurst_exponent": hurst,
                    "half_life": half_life,
                    "deviation_zscore": deviation_zscore,
                    "strategy_type": "mean_reversion",
                },
            ))

        # Z-score normalize
        if scores:
            raw_values = np.array([s.score for s in scores])
            mean = np.mean(raw_values)
            std = np.std(raw_values)
            if std > 1e-10:
                for s in scores:
                    s.score = (s.score - mean) / std

        scores.sort(key=lambda s: abs(s.score), reverse=True)
        max_pos = self.config.max_positions_per_side
        longs = [s for s in scores if s.direction == SignalDirection.LONG][:max_pos]
        shorts = [s for s in scores if s.direction == SignalDirection.SHORT][:max_pos]
        scores = longs + shorts

        logger.info("%s: %d signals (%d long, %d short)",
                    self.name, len(scores), len(longs), len(shorts))

        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=scores,
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
        )

    def _compute_basic_signals(self, data: pd.DataFrame) -> dict:
        """Fallback: compute Hurst + OU from raw price data."""
        from models.mean_reversion_model import _compute_hurst, _fit_ou_params

        result = {}
        for symbol in data["symbol"].unique():
            sym_data = data[data["symbol"] == symbol].sort_values(
                "timestamp" if "timestamp" in data.columns else data.columns[0]
            )
            prices = sym_data["close"].dropna().values
            if len(prices) < 100:
                continue

            hurst = _compute_hurst(prices)
            ou = _fit_ou_params(prices)

            current_log_price = np.log(prices[-1])
            deviation = current_log_price - ou["mu"]
            price_std = np.std(np.log(prices[-63:])) if len(prices) >= 63 else np.std(np.log(prices))
            deviation_zscore = deviation / price_std if price_std > 1e-10 else 0.0

            result[symbol] = type("Pred", (), {
                "metadata": {
                    "hurst_exponent": hurst,
                    "half_life": ou["half_life"],
                    "deviation_zscore": deviation_zscore,
                },
            })()

        return result

    def get_performance(self) -> StrategyPerformance:
        return self._performance
