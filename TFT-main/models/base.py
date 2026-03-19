"""
Base class for all APEX TFT model wrappers.

Each asset-class model (stocks, forex, volatility) extends this base
to provide consistent prediction interface for the model manager.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Standardized prediction output from any TFT model."""
    symbol: str
    predicted_value: float          # median (50th percentile) forecast
    lower_bound: float              # 10th percentile
    upper_bound: float              # 90th percentile
    confidence: float               # derived from quantile spread
    horizon_days: int
    model_name: str
    prediction_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Metadata about a trained model."""
    name: str
    asset_class: str                # "stocks", "forex", "volatility"
    version: str
    trained_at: Optional[datetime] = None
    symbols: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    model_path: Optional[str] = None
    is_loaded: bool = False


class BaseTFTModel(ABC):
    """
    Abstract base for all TFT model wrappers.

    Lifecycle:
        1. __init__(config) — set parameters
        2. train(data) — train on historical data
        3. save(path) / load(path) — persist / restore
        4. predict(data) — generate predictions
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier."""

    @property
    @abstractmethod
    def asset_class(self) -> str:
        """Asset class: 'stocks', 'forex', or 'volatility'."""

    @abstractmethod
    def prepare_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data into model-ready features.

        Each subclass implements asset-class-specific feature engineering.
        Output must include: symbol/group_id, time_idx, target, and all features.
        """

    @abstractmethod
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Train the model on historical data.

        Args:
            data: Raw data (will be passed through prepare_features).

        Returns:
            Training metrics dict (val_loss, etc.).
        """

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """
        Generate predictions for the latest period.

        Args:
            data: Raw data (will be passed through prepare_features).

        Returns:
            List of ModelPrediction per symbol.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""

    @abstractmethod
    def load(self, path: str) -> bool:
        """Load model from disk. Returns True if successful."""

    def get_info(self) -> ModelInfo:
        """Return model metadata."""
        return ModelInfo(
            name=self.name,
            asset_class=self.asset_class,
            version="1.0",
        )
