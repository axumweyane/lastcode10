"""
TFT-Stocks Adapter — wraps the existing EnhancedTFTModel for ensemble use.

This does NOT duplicate the existing model. It wraps it to provide
the standardized BaseTFTModel interface so the ModelManager can
treat stocks predictions the same as forex and volatility predictions.

Used by: momentum, pairs, covered calls, earnings plays.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from models.base import BaseTFTModel, ModelInfo, ModelPrediction

logger = logging.getLogger(__name__)


class TFTStocksAdapter(BaseTFTModel):
    """
    Adapter for the existing EnhancedTFTModel (tft_model.py).

    If a trained model exists at models/tft_model.pth, loads and uses it.
    Otherwise, returns empty predictions (strategies fall back to non-TFT signals).
    """

    def __init__(self, model_path: str = "models/tft_model.pth"):
        self._model_path = model_path
        self._model = None
        self._is_loaded = False
        self._trained_at: Optional[datetime] = None
        self._symbols: List[str] = []

    @property
    def name(self) -> str:
        return "tft_stocks"

    @property
    def asset_class(self) -> str:
        return "stocks"

    def prepare_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Delegates to the existing StockDataPreprocessor pipeline.
        This adapter doesn't reimplement feature engineering.
        """
        return raw_data  # existing pipeline handles this

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Training is handled by the existing train.py / train_postgres.py.
        This adapter only loads pre-trained models.
        """
        logger.info(
            "TFT-Stocks training should use existing train.py or train_postgres.py"
        )
        return {"status": "use_existing_training_scripts"}

    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """
        Generate stock return predictions.

        If no trained model is loaded, returns empty list (fallback).
        If loaded, produces 5-day return forecasts per symbol.
        """
        if not self._is_loaded:
            logger.debug("TFT-Stocks model not loaded, returning empty predictions")
            return []

        try:
            from tft_model import EnhancedTFTModel
            from data_preprocessing import StockDataPreprocessor

            # Preprocess
            preprocessor = StockDataPreprocessor()
            processed = preprocessor.fit_transform(data, target_type="returns")

            # Create dataset and predict
            _, val_ds = self._model.create_datasets(processed, validation_split=0.01)
            raw_preds = self._model.predict(val_ds)

            if raw_preds is None:
                return []

            predictions = []
            symbols = processed["symbol"].unique()

            # Map predictions back to symbols
            preds_array = (
                raw_preds.numpy()
                if hasattr(raw_preds, "numpy")
                else np.array(raw_preds)
            )

            for i, symbol in enumerate(symbols):
                if i >= len(preds_array):
                    break

                pred = preds_array[i]
                if len(pred.shape) > 1 and pred.shape[-1] >= 3:
                    lower, median, upper = pred[-1, 0], pred[-1, 1], pred[-1, 2]
                elif len(pred.shape) == 1 and len(pred) >= 3:
                    lower, median, upper = pred[0], pred[1], pred[2]
                else:
                    median = float(pred.flatten()[-1])
                    lower, upper = median * 0.8, median * 1.2

                spread = abs(upper - lower)
                confidence = max(0.1, 1.0 - spread * 5)

                predictions.append(
                    ModelPrediction(
                        symbol=symbol,
                        predicted_value=float(median),
                        lower_bound=float(lower),
                        upper_bound=float(upper),
                        confidence=min(confidence, 0.95),
                        horizon_days=5,
                        model_name=self.name,
                        metadata={"asset_class": "stocks"},
                    )
                )

            logger.info("TFT-Stocks: %d predictions", len(predictions))
            return predictions

        except Exception as e:
            logger.error("TFT-Stocks prediction failed: %s", e)
            return []

    def save(self, path: str) -> None:
        logger.info("TFT-Stocks uses existing save mechanism in tft_model.py")

    def load(self, path: str = None) -> bool:
        """Load the existing trained TFT stock model.

        Supports two checkpoint formats:
        - Legacy (EnhancedTFTModel): config with loss_type, uses create_model()
        - Postgres (TFTPostgresModel): config with target_type, uses from_dataset()
        """
        load_path = path or self._model_path

        if not Path(load_path).exists():
            logger.info(
                "TFT-Stocks model not found at %s (strategies will use fallback)",
                load_path,
            )
            return False

        try:
            checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)

            config = checkpoint.get("config", {})
            state_dict = checkpoint.get("model_state_dict")
            training_dataset = checkpoint.get("training_dataset")

            if state_dict is None or training_dataset is None:
                raise ValueError(
                    "checkpoint missing model_state_dict or training_dataset"
                )

            if "loss_type" in config:
                # Legacy format — use EnhancedTFTModel
                from tft_model import EnhancedTFTModel

                self._model = EnhancedTFTModel()
                self._model.config = config
                self._model.training_dataset = training_dataset
                self._model.model = self._model.create_model(training_dataset)
                self._model.model.load_state_dict(state_dict)
            else:
                # Postgres format — reconstruct TFT directly
                from pytorch_forecasting import TemporalFusionTransformer
                from pytorch_forecasting.metrics import QuantileLoss

                quantiles = config.get("quantiles", [0.1, 0.5, 0.9])
                tft = TemporalFusionTransformer.from_dataset(
                    training_dataset,
                    learning_rate=config.get("learning_rate", 0.001),
                    hidden_size=config.get("hidden_size", 64),
                    attention_head_size=config.get("attention_head_size", 4),
                    dropout=config.get("dropout", 0.2),
                    hidden_continuous_size=config.get("hidden_continuous_size", 32),
                    lstm_layers=config.get("lstm_layers", 2),
                    loss=QuantileLoss(quantiles=quantiles),
                    output_size=len(quantiles),
                    reduce_on_plateau_patience=4,
                    optimizer="adamw",
                )
                tft.load_state_dict(state_dict)
                self._model = tft
                self._config = config
                self._training_dataset = training_dataset

            self._is_loaded = True
            self._trained_at = datetime.fromtimestamp(Path(load_path).stat().st_mtime)
            logger.info("TFT-Stocks model loaded from %s", load_path)
            return True
        except Exception as e:
            logger.warning("Failed to load TFT-Stocks model: %s", e)
            self._is_loaded = False
            return False

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            asset_class=self.asset_class,
            version="1.0",
            trained_at=self._trained_at,
            symbols=self._symbols,
            model_path=self._model_path,
            is_loaded=self._is_loaded,
        )
