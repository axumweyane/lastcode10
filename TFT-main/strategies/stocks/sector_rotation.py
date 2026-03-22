"""
Sector Rotation Strategy — macro regime drives sector allocation.

Alpha source: yield curve regime + rate trends mapped to sector overweights.
Uses MacroRegimeModel (#9) for regime classification.
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
from strategies.config import SectorRotationConfig

logger = logging.getLogger(__name__)

# Default sector map if model not available
DEFAULT_SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "META": "Technology",
    "NVDA": "Technology",
    "AMD": "Technology",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS": "Financials",
    "XOM": "Energy",
    "CVX": "Energy",
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "PFE": "Healthcare",
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "CAT": "Industrials",
    "BA": "Industrials",
    "NEE": "Utilities",
    "DUK": "Utilities",
}


class SectorRotationStrategy(BaseStrategy):

    def __init__(
        self,
        config: Optional[SectorRotationConfig] = None,
        manager: Optional[ModelManager] = None,
    ):
        self.config = config or SectorRotationConfig.from_env()
        self._manager = manager
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._initialized = False

    @property
    def name(self) -> str:
        return "sector_rotation"

    @property
    def description(self) -> str:
        return "Macro regime driven sector rotation: overweight cyclicals in expansion, defensives in contraction"

    def initialize(self, historical_data: pd.DataFrame) -> None:
        self._initialized = True

    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        if not self._initialized:
            self.initialize(data)

        scores: List[AlphaScore] = []

        # Get macro regime predictions
        macro_predictions = {}
        if self._manager is not None:
            try:
                preds = self._manager.predict_macro(data)
                macro_predictions = {p.symbol: p for p in preds}
            except Exception as e:
                logger.warning("MacroRegime model failed: %s", e)

        symbols = data["symbol"].unique() if "symbol" in data.columns else []

        if macro_predictions:
            # Use model predictions (include sector tilts)
            for symbol in symbols:
                pred = macro_predictions.get(symbol)
                if pred is None:
                    continue

                tilt = pred.predicted_value
                meta = pred.metadata
                confidence = pred.confidence

                if abs(tilt) < self.config.min_tilt_threshold:
                    continue

                direction = SignalDirection.LONG if tilt > 0 else SignalDirection.SHORT

                scores.append(
                    AlphaScore(
                        symbol=symbol,
                        score=tilt,
                        raw_score=tilt,
                        confidence=confidence,
                        direction=direction,
                        metadata={
                            "strategy_type": "sector_rotation",
                            "sector": meta.get("sector", "Unknown"),
                            "curve_regime": meta.get("curve_regime", "unknown"),
                            "yield_spread": meta.get("yield_spread", 0.0),
                        },
                    )
                )
        else:
            # Fallback: use simple momentum-based sector rotation
            scores = self._fallback_sector_rotation(data, symbols)

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

        logger.info(
            "%s: %d signals (%d long, %d short)",
            self.name,
            len(scores),
            len(longs),
            len(shorts),
        )

        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=scores,
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
        )

    def _fallback_sector_rotation(
        self, data: pd.DataFrame, symbols
    ) -> List[AlphaScore]:
        """Simple relative strength rotation when macro model is unavailable."""
        scores = []
        sector_returns = {}

        for symbol in symbols:
            sym_data = data[data["symbol"] == symbol].sort_values(
                "timestamp" if "timestamp" in data.columns else data.columns[0]
            )
            if len(sym_data) < 63:
                continue

            close = sym_data["close"].values
            ret_3m = (close[-1] / close[-63] - 1) if close[-63] > 0 else 0.0
            sector = DEFAULT_SECTOR_MAP.get(symbol, "Unknown")

            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append((symbol, ret_3m))

        # Rank sectors by average return
        sector_avg = {
            s: np.mean([r for _, r in rets]) for s, rets in sector_returns.items()
        }
        if not sector_avg:
            return []

        avg_values = list(sector_avg.values())
        overall_mean = np.mean(avg_values)
        overall_std = np.std(avg_values) if len(avg_values) > 1 else 1.0

        for sector, rets in sector_returns.items():
            sector_z = (
                (sector_avg[sector] - overall_mean) / overall_std
                if overall_std > 1e-10
                else 0.0
            )
            for symbol, _ in rets:
                if abs(sector_z) < self.config.min_tilt_threshold:
                    continue
                direction = (
                    SignalDirection.LONG if sector_z > 0 else SignalDirection.SHORT
                )
                scores.append(
                    AlphaScore(
                        symbol=symbol,
                        score=sector_z,
                        raw_score=sector_z,
                        confidence=min(abs(sector_z) / 2.0, 0.8),
                        direction=direction,
                        metadata={
                            "strategy_type": "sector_rotation",
                            "sector": sector,
                            "method": "relative_strength_fallback",
                        },
                    )
                )

        return scores

    def get_performance(self) -> StrategyPerformance:
        return self._performance
