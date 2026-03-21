"""
Model Manager — loads all TFT and external models and provides unified predictions.

Key design: graceful fallback. If a model isn't trained yet, strategies
receive empty predictions and use their non-TFT signals instead.
This means you can deploy the system immediately and add models
incrementally as they're trained.

Model registry (10 models):
  - tft_stocks:       models/tft_model.pth        (existing)
  - tft_forex:        models/tft_forex.pth         (existing)
  - tft_volatility:   models/tft_volatility.pth    (existing)
  - kronos:           pre-trained (HuggingFace)    (Strategy #12)
  - deep_surrogates:  pre-trained (repo)           (Strategy #13)
  - tdgf:             models/tdgf_model.pth        (Strategy #14)
  - sentiment:        pre-trained (FinBERT)         (Model #7)
  - mean_reversion:   statistical estimation        (Model #8)
  - macro_regime:     rule-based                    (Model #9)
  - microstructure:   statistical                   (Model #10)
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from models.base import BaseTFTModel, ModelInfo, ModelPrediction
from models.stocks_adapter import TFTStocksAdapter
from models.forex_model import TFTForexModel
from models.volatility_model import TFTVolatilityModel
from models.kronos_model import KronosModel
from models.deep_surrogate_model import DeepSurrogateModel
from models.tdgf_model import TDGFModel
from models.sentiment_model import SentimentModel
from models.mean_reversion_model import MeanReversionModel
from models.macro_model import MacroRegimeModel
from models.microstructure_model import MicrostructureModel

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PATHS = {
    "tft_stocks": os.path.join(_BASE_DIR, "models", "tft_model.pth"),
    "tft_forex": os.path.join(_BASE_DIR, "models", "tft_forex.pth"),
    "tft_volatility": os.path.join(_BASE_DIR, "models", "tft_volatility.pth"),
    "kronos": "",  # pre-trained, loaded from HuggingFace
    "deep_surrogates": "",  # pre-trained, loaded from repo
    "tdgf": os.path.join(_BASE_DIR, "models", "tdgf_model.pth"),
    "sentiment": "",  # pre-trained (FinBERT / VADER)
    "mean_reversion": "",  # statistical, no weights
    "macro_regime": "",  # rule-based, no weights
    "microstructure": "",  # statistical, no weights
}


@dataclass
class ManagerStatus:
    """Status of all managed models."""
    models_registered: int
    models_loaded: int
    models_failed: int
    details: List[ModelInfo] = field(default_factory=list)


class ModelManager:
    """
    Unified model management for all TFT variants.

    Usage:
        manager = ModelManager()
        manager.load_all()

        stock_preds = manager.predict_stocks(stock_data)
        fx_preds = manager.predict_forex(fx_data)
        vol_preds = manager.predict_volatility(stock_data)

        # Or get all predictions at once
        all_preds = manager.predict_all(stock_data, fx_data)
    """

    def __init__(self, model_paths: Optional[Dict[str, str]] = None):
        paths = model_paths or DEFAULT_PATHS

        self._models: Dict[str, BaseTFTModel] = {
            "tft_stocks": TFTStocksAdapter(paths.get("tft_stocks", DEFAULT_PATHS["tft_stocks"])),
            "tft_forex": TFTForexModel(),
            "tft_volatility": TFTVolatilityModel(),
            "kronos": KronosModel(),
            "deep_surrogates": DeepSurrogateModel(),
            "tdgf": TDGFModel(),
            "sentiment": SentimentModel(),
            "mean_reversion": MeanReversionModel(),
            "macro_regime": MacroRegimeModel(),
            "microstructure": MicrostructureModel(),
        }
        self._paths = paths

    def load_all(self) -> ManagerStatus:
        """Attempt to load all models. Log which succeed and which fail."""
        loaded = 0
        failed = 0
        details = []

        for name, model in self._models.items():
            path = self._paths.get(name, DEFAULT_PATHS.get(name, ""))
            try:
                success = model.load(path)
                if success:
                    loaded += 1
                    logger.info("Model %s loaded from %s", name, path)
                else:
                    failed += 1
                    logger.info("Model %s not available (strategies will use fallback)", name)
            except Exception as e:
                failed += 1
                logger.warning("Model %s load error: %s", name, e)

            details.append(model.get_info())

        status = ManagerStatus(
            models_registered=len(self._models),
            models_loaded=loaded,
            models_failed=failed,
            details=details,
        )

        logger.info(
            "ModelManager: %d/%d models loaded",
            loaded, len(self._models),
        )
        return status

    def predict_stocks(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Get stock return predictions. Empty list if model not loaded."""
        return self._models["tft_stocks"].predict(data)

    def predict_forex(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Get FX return predictions. Empty list if model not loaded."""
        return self._models["tft_forex"].predict(data)

    def predict_volatility(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Get vol forecasts. Empty list if model not loaded."""
        return self._models["tft_volatility"].predict(data)

    def predict_kronos(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Get Kronos forecasts (stocks + forex). Empty list if not loaded."""
        return self._models["kronos"].predict(data)

    def predict_deep_surrogates(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Get DeepSurrogate options/risk signals. Empty list if not loaded."""
        return self._models["deep_surrogates"].predict(data)

    def predict_tdgf(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Get TDGF American option prices. Empty list if not loaded/trained."""
        return self._models["tdgf"].predict(data)

    def predict_sentiment(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Get sentiment scores. Empty list if not loaded."""
        return self._models["sentiment"].predict(data)

    def predict_mean_reversion(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Get mean reversion signals. Empty list if not loaded."""
        return self._models["mean_reversion"].predict(data)

    def predict_macro(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Get macro regime predictions. Empty list if not loaded."""
        return self._models["macro_regime"].predict(data)

    def predict_microstructure(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Get microstructure signals. Empty list if not loaded."""
        return self._models["microstructure"].predict(data)

    def predict_all(
        self,
        stock_data: Optional[pd.DataFrame] = None,
        fx_data: Optional[pd.DataFrame] = None,
        options_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, List[ModelPrediction]]:
        """Get predictions from all available models."""
        results = {}

        if stock_data is not None:
            results["stocks"] = self.predict_stocks(stock_data)
            results["volatility"] = self.predict_volatility(stock_data)
            results["kronos_stocks"] = self.predict_kronos(stock_data)

        if fx_data is not None:
            results["forex"] = self.predict_forex(fx_data)
            results["kronos_forex"] = self.predict_kronos(fx_data)

        if options_data is not None:
            results["deep_surrogates"] = self.predict_deep_surrogates(options_data)
            results["tdgf"] = self.predict_tdgf(options_data)

        total = sum(len(v) for v in results.values())
        logger.info("ModelManager predictions: %d total (%s)",
                     total, {k: len(v) for k, v in results.items()})
        return results

    def get_model(self, name: str) -> Optional[BaseTFTModel]:
        return self._models.get(name)

    def is_model_loaded(self, name: str) -> bool:
        model = self._models.get(name)
        if model is None:
            return False
        return model.get_info().is_loaded

    def get_status(self) -> ManagerStatus:
        details = [m.get_info() for m in self._models.values()]
        loaded = sum(1 for d in details if d.is_loaded)
        return ManagerStatus(
            models_registered=len(self._models),
            models_loaded=loaded,
            models_failed=len(self._models) - loaded,
            details=details,
        )

    def predictions_to_dict(
        self, predictions: List[ModelPrediction],
    ) -> Dict[str, float]:
        """Convert prediction list to {symbol: predicted_value} for strategy consumption."""
        return {p.symbol: p.predicted_value for p in predictions}

    def predictions_to_dataframe(
        self, predictions: List[ModelPrediction],
    ) -> pd.DataFrame:
        """Convert predictions to DataFrame for ensemble combiner TFTAdapter."""
        if not predictions:
            return pd.DataFrame(columns=["symbol", "predicted_return", "confidence",
                                          "lower_bound", "upper_bound"])
        rows = []
        for p in predictions:
            rows.append({
                "symbol": p.symbol,
                "predicted_return": p.predicted_value,
                "confidence": p.confidence,
                "lower_bound": p.lower_bound,
                "upper_bound": p.upper_bound,
            })
        return pd.DataFrame(rows)
