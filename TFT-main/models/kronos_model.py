"""
Kronos — Pre-trained financial time series foundation model.

Strategy #12: Stocks + Forex price forecasting via Monte Carlo sampling.
Uses HuggingFace pre-trained models (NeoQuasar/Kronos-*).
NO TRAINING NEEDED — download and run inference.

Repository: https://github.com/shiyu-coder/Kronos
Models: NeoQuasar/Kronos-mini (4.1M), NeoQuasar/Kronos-small (24.7M),
        NeoQuasar/Kronos-base (102.3M)
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from models.base import BaseTFTModel, ModelInfo, ModelPrediction

logger = logging.getLogger(__name__)

KRONOS_REPO_PATH = os.getenv("KRONOS_REPO_PATH", "/opt/kronos")
KRONOS_MODEL_NAME = os.getenv("KRONOS_MODEL_NAME", "NeoQuasar/Kronos-base")
KRONOS_TOKENIZER_NAME = os.getenv(
    "KRONOS_TOKENIZER_NAME", "NeoQuasar/Kronos-Tokenizer-base"
)
KRONOS_MAX_CONTEXT = int(os.getenv("KRONOS_MAX_CONTEXT", "512"))
KRONOS_NUM_SAMPLES = int(os.getenv("KRONOS_NUM_SAMPLES", "100"))
KRONOS_PREDICTION_LENGTH = int(os.getenv("KRONOS_PREDICTION_LENGTH", "5"))

FX_PAIRS = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"}


class KronosModel(BaseTFTModel):
    """
    Wraps the Kronos pre-trained foundation model for financial forecasting.

    Produces probabilistic multi-step forecasts via Monte Carlo sampling
    of future K-lines. Supports both stocks and forex.
    """

    def __init__(self):
        self._predictor = None
        self._is_loaded = False
        self._symbols: List[str] = []
        self._model_name = KRONOS_MODEL_NAME
        self._tokenizer_name = KRONOS_TOKENIZER_NAME
        self._max_context = KRONOS_MAX_CONTEXT
        self._num_samples = KRONOS_NUM_SAMPLES
        self._prediction_length = KRONOS_PREDICTION_LENGTH

    @property
    def name(self) -> str:
        return "kronos"

    @property
    def asset_class(self) -> str:
        return "stocks"  # also supports forex

    def prepare_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Kronos expects OHLCV with columns: open, high, low, close."""
        df = raw_data.copy()
        df.columns = [c.lower() for c in df.columns]
        required = ["open", "high", "low", "close"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Kronos is pre-trained. No training required."""
        logger.info("Kronos is a pre-trained foundation model. No training needed.")
        return {"status": "pre_trained", "val_loss": 0.0}

    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """Generate forecasts via Monte Carlo sampling."""
        if not self._is_loaded:
            logger.debug("Kronos model not loaded, returning empty predictions")
            return []

        predictions = []
        symbols = (
            data["symbol"].unique() if "symbol" in data.columns else ["UNKNOWN"]
        )

        for symbol in symbols:
            try:
                sym_data = (
                    data[data["symbol"] == symbol]
                    if "symbol" in data.columns
                    else data
                )
                pred = self._predict_symbol(symbol, sym_data)
                if pred is not None:
                    predictions.append(pred)
            except Exception as e:
                logger.error("Kronos prediction failed for %s: %s", symbol, e)

        logger.info("Kronos generated %d predictions", len(predictions))
        return predictions

    def _predict_symbol(
        self, symbol: str, sym_data: pd.DataFrame
    ) -> Optional[ModelPrediction]:
        """Run Kronos inference for a single symbol."""
        df = self.prepare_features(sym_data)

        # Truncate to max context length
        if len(df) > self._max_context:
            df = df.tail(self._max_context)

        if len(df) < 10:
            logger.warning("Kronos: insufficient data for %s (%d rows)", symbol, len(df))
            return None

        # Build OHLCV input
        ohlcv = df[["open", "high", "low", "close"]].copy()
        if "volume" in df.columns:
            ohlcv["volume"] = df["volume"]
        if "amount" in df.columns:
            ohlcv["amount"] = df["amount"]

        # Run Kronos predictor
        try:
            result = self._predictor.predict(
                ohlcv,
                prediction_length=self._prediction_length,
                num_samples=self._num_samples,
            )
        except Exception as e:
            logger.error("Kronos predict() failed for %s: %s", symbol, e)
            return None

        # Extract close price forecasts from Monte Carlo samples
        close_samples = self._extract_close_samples(result)
        if close_samples is None or len(close_samples) == 0:
            return None

        # Compute statistics from samples
        current_close = float(df["close"].iloc[-1])

        # Use the final prediction step
        if close_samples.ndim > 1:
            final_step_samples = close_samples[:, -1]
        else:
            final_step_samples = close_samples

        median_price = float(np.median(final_step_samples))
        lower_price = float(np.percentile(final_step_samples, 10))
        upper_price = float(np.percentile(final_step_samples, 90))

        # Convert to returns
        if current_close <= 0:
            return None
        predicted_return = (median_price - current_close) / current_close
        lower_return = (lower_price - current_close) / current_close
        upper_return = (upper_price - current_close) / current_close

        # Confidence from spread (tighter spread = higher confidence)
        spread = abs(upper_return - lower_return)
        confidence = max(0.1, min(0.95, 1.0 - spread * 5))

        detected_class = "forex" if symbol.upper() in FX_PAIRS else "stocks"

        return ModelPrediction(
            symbol=symbol,
            predicted_value=predicted_return,
            lower_bound=lower_return,
            upper_bound=upper_return,
            confidence=confidence,
            horizon_days=self._prediction_length,
            model_name=self.name,
            metadata={
                "asset_class": detected_class,
                "median_price": median_price,
                "current_price": current_close,
                "num_samples": self._num_samples,
                "model": self._model_name,
            },
        )

    def _extract_close_samples(self, result: Any) -> Optional[np.ndarray]:
        """Extract close price samples from Kronos output."""
        try:
            if isinstance(result, dict):
                if "close" in result:
                    return np.array(result["close"])
                if "samples" in result:
                    samples = np.array(result["samples"])
                    # Shape (num_samples, pred_len, 4) — close is index 3
                    if samples.ndim == 3 and samples.shape[2] >= 4:
                        return samples[:, :, 3]
                    return samples
            elif isinstance(result, np.ndarray):
                return result
            elif hasattr(result, "close"):
                return np.array(result.close)
            elif hasattr(result, "samples"):
                return np.array(result.samples)

            logger.warning("Unexpected Kronos output format: %s", type(result))
            return None
        except Exception as e:
            logger.error("Failed to extract close samples: %s", e)
            return None

    def save(self, path: str) -> None:
        """Kronos is pre-trained — nothing to save."""
        logger.info("Kronos is pre-trained; no local model to save")

    def load(self, path: str = None) -> bool:
        """Load Kronos from HuggingFace via the cloned repo."""
        try:
            kronos_path = Path(KRONOS_REPO_PATH)
            if kronos_path.exists() and str(kronos_path) not in sys.path:
                sys.path.insert(0, str(kronos_path))

            from kronos import KronosPredictor  # type: ignore

            self._predictor = KronosPredictor(
                model_name=self._model_name,
                tokenizer_name=self._tokenizer_name,
            )
            self._is_loaded = True
            logger.info("Kronos model loaded: %s", self._model_name)
            return True
        except ImportError as e:
            logger.warning(
                "Kronos not available (clone repo to %s): %s",
                KRONOS_REPO_PATH,
                e,
            )
            return False
        except Exception as e:
            logger.error("Failed to load Kronos: %s", e)
            return False

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            asset_class=self.asset_class,
            version="1.0",
            symbols=self._symbols,
            model_path=KRONOS_REPO_PATH,
            is_loaded=self._is_loaded,
        )
