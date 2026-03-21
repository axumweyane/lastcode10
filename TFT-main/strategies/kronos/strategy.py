"""
Kronos Strategy — Foundation model forecasting for stocks and forex.

Strategy #12: Wraps KronosModel predictions into the BaseStrategy interface.
Uses TFTAdapter for prediction-to-alpha conversion with z-scoring.

Kronos produces probabilistic multi-step forecasts via Monte Carlo sampling
of future K-lines. It handles both equities and FX — asset class is detected
from the symbol and routed accordingly through the model manager.

Graceful degradation: if Kronos repo is not cloned or HuggingFace model
not downloaded, returns empty StrategyOutput and the ensemble combiner
simply skips it in the weighted combination.
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
from strategies.config import KronosConfig
from strategies.ensemble.combiner import TFTAdapter

logger = logging.getLogger(__name__)

FX_SYMBOLS = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"}


class KronosStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "kronos"

    @property
    def description(self) -> str:
        return "Kronos foundation model forecasting for stocks and forex"

    def __init__(
        self,
        config: Optional[KronosConfig] = None,
        manager: Optional[ModelManager] = None,
    ):
        self.config = config or KronosConfig.from_env()
        self._manager = manager
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._adapter = TFTAdapter()
        self._initialized = False
        self._prediction_counts: Dict[str, int] = {}  # track per-symbol counts

    def initialize(self, historical_data: pd.DataFrame) -> None:
        if self._manager is not None and self._manager.is_model_loaded("kronos"):
            self._initialized = True
            symbols = (
                historical_data["symbol"].unique().tolist()
                if "symbol" in historical_data.columns
                else []
            )
            logger.info(
                "KronosStrategy initialized (%d symbols available)", len(symbols)
            )
        else:
            logger.info("Kronos model not loaded — strategy will return empty signals")

    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        if not self._initialized:
            self.initialize(data)

        if self._manager is None or not self._manager.is_model_loaded("kronos"):
            return self._empty_output()

        if data.empty or "symbol" not in data.columns:
            logger.warning("Kronos: empty or malformed input data")
            return self._empty_output()

        try:
            # Split by asset class — Kronos handles both
            stock_syms = [s for s in data["symbol"].unique() if s not in FX_SYMBOLS]
            fx_syms = [s for s in data["symbol"].unique() if s in FX_SYMBOLS]

            all_predictions: List[ModelPrediction] = []

            if stock_syms:
                stock_data = data[data["symbol"].isin(stock_syms)]
                all_predictions.extend(self._manager.predict_kronos(stock_data))

            if fx_syms:
                fx_data = data[data["symbol"].isin(fx_syms)]
                all_predictions.extend(self._manager.predict_kronos(fx_data))

            if not all_predictions:
                return self._empty_output()

            # Convert through TFTAdapter (z-scores and assigns direction)
            predictions_df = self._manager.predictions_to_dataframe(all_predictions)
            output = self._adapter.adapt(predictions_df)

            # Stamp as kronos (TFTAdapter defaults to "tft_adapter")
            output.strategy_name = self.name
            output.strategy_sharpe_63d = self._performance.sharpe_63d
            output.strategy_sharpe_21d = self._performance.sharpe_21d

            # Enrich metadata with asset class tags
            pred_map = {p.symbol: p for p in all_predictions}
            for score in output.scores:
                pred = pred_map.get(score.symbol)
                if pred:
                    score.metadata["asset_class"] = pred.metadata.get(
                        "asset_class", "stocks"
                    )
                    score.metadata["kronos_model"] = pred.metadata.get("model", "")
                    score.metadata["num_samples"] = pred.metadata.get("num_samples", 0)

            # Track prediction counts for diagnostics
            for score in output.scores:
                self._prediction_counts[score.symbol] = (
                    self._prediction_counts.get(score.symbol, 0) + 1
                )

            logger.info(
                "Kronos: %d signals (%d stock, %d fx)",
                len(output.scores),
                sum(1 for s in output.scores
                    if s.metadata.get("asset_class") == "stocks"),
                sum(1 for s in output.scores
                    if s.metadata.get("asset_class") == "forex"),
            )
            return output

        except Exception as e:
            logger.error("KronosStrategy.generate_signals failed: %s", e, exc_info=True)
            return self._empty_output()

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
