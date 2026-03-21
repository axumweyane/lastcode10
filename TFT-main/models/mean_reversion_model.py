"""
Model 8: Mean Reversion Model — Hurst exponent + Ornstein-Uhlenbeck parameter estimation.

Identifies mean-reverting regimes and estimates equilibrium parameters.
Input: OHLCV data with 252+ days history.
Output: per-symbol mean reversion probability and half-life.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from models.base import BaseTFTModel, ModelInfo, ModelPrediction

logger = logging.getLogger(__name__)


def _compute_hurst(prices: np.ndarray, max_lag: int = 100) -> float:
    """Compute Hurst exponent using rescaled range (R/S) analysis."""
    n = len(prices)
    if n < 20:
        return 0.5

    log_returns = np.diff(np.log(prices))
    lags = range(2, min(max_lag, n // 4))
    rs_values = []
    lag_values = []

    for lag in lags:
        chunks = [log_returns[i:i + lag] for i in range(0, len(log_returns) - lag + 1, lag)]
        rs_list = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            cumdev = np.cumsum(chunk - mean_c)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(chunk, ddof=1)
            if s > 1e-10:
                rs_list.append(r / s)
        if rs_list:
            rs_values.append(np.log(np.mean(rs_list)))
            lag_values.append(np.log(lag))

    if len(lag_values) < 3:
        return 0.5

    coeffs = np.polyfit(lag_values, rs_values, 1)
    return float(np.clip(coeffs[0], 0.0, 1.0))


def _fit_ou_params(prices: np.ndarray) -> Dict[str, float]:
    """Fit Ornstein-Uhlenbeck parameters via OLS regression on log prices."""
    n = len(prices)
    if n < 30:
        return {"mu": 0.0, "theta": 0.0, "sigma": 0.0, "half_life": float("inf")}

    log_prices = np.log(prices)
    y = np.diff(log_prices)
    x = log_prices[:-1]

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx < 1e-10:
        return {"mu": float(x_mean), "theta": 0.0, "sigma": 0.0, "half_life": float("inf")}

    beta = np.sum((x - x_mean) * (y - y_mean)) / ss_xx
    alpha = y_mean - beta * x_mean

    theta = -beta  # mean reversion speed
    if theta > 1e-10:
        mu = alpha / theta  # equilibrium level
        half_life = np.log(2) / theta
    else:
        mu = float(x_mean)
        half_life = float("inf")

    residuals = y - (alpha + beta * x)
    sigma = float(np.std(residuals, ddof=2)) if n > 2 else 0.0

    return {
        "mu": float(mu),
        "theta": float(max(theta, 0.0)),
        "sigma": sigma,
        "half_life": float(np.clip(half_life, 0.5, 500.0)),
    }


class MeanReversionModel(BaseTFTModel):
    """
    Hurst exponent + OU parameter estimation for mean reversion detection.
    """

    def __init__(self):
        self._is_loaded = False

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def asset_class(self) -> str:
        return "stocks"

    def prepare_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        required = {"symbol", "close"}
        if not required.issubset(raw_data.columns):
            return pd.DataFrame()
        return raw_data.copy()

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        return {"status": "no_training_needed", "note": "statistical estimation model"}

    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        if data.empty or "close" not in data.columns or "symbol" not in data.columns:
            return []

        predictions = []
        for symbol in data["symbol"].unique():
            sym_data = data[data["symbol"] == symbol].sort_values("timestamp" if "timestamp" in data.columns else data.columns[0])
            prices = sym_data["close"].dropna().values

            if len(prices) < 100:
                continue

            hurst = _compute_hurst(prices)
            ou = _fit_ou_params(prices)

            # Mean reversion probability: lower Hurst = more mean-reverting
            mr_probability = max(0.0, 1.0 - 2.0 * hurst) if hurst < 0.5 else 0.0

            # Deviation from equilibrium
            current_log_price = np.log(prices[-1])
            deviation = current_log_price - ou["mu"]
            price_std = np.std(np.log(prices[-63:])) if len(prices) >= 63 else np.std(np.log(prices))
            deviation_zscore = deviation / price_std if price_std > 1e-10 else 0.0

            # Predicted value: expected reversion direction and magnitude
            # Negative deviation_zscore means below equilibrium (expect up)
            predicted_value = -deviation_zscore * mr_probability

            # Confidence based on Hurst quality and half-life reasonableness
            half_life_quality = 1.0 if 2.0 <= ou["half_life"] <= 30.0 else 0.5
            confidence = mr_probability * half_life_quality

            predictions.append(ModelPrediction(
                symbol=symbol,
                predicted_value=predicted_value,
                lower_bound=predicted_value - abs(deviation_zscore) * 0.5,
                upper_bound=predicted_value + abs(deviation_zscore) * 0.5,
                confidence=confidence,
                horizon_days=int(min(ou["half_life"], 30)),
                model_name=self.name,
                metadata={
                    "hurst_exponent": round(hurst, 4),
                    "half_life": round(ou["half_life"], 2),
                    "ou_mu": round(ou["mu"], 6),
                    "ou_theta": round(ou["theta"], 6),
                    "ou_sigma": round(ou["sigma"], 6),
                    "deviation_zscore": round(deviation_zscore, 4),
                    "mr_probability": round(mr_probability, 4),
                },
            ))

        return predictions

    def save(self, path: str) -> None:
        pass  # Statistical model, no state to save

    def load(self, path: str) -> bool:
        self._is_loaded = True
        logger.info("MeanReversionModel ready (statistical estimation, no weights to load)")
        return True

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            asset_class=self.asset_class,
            version="1.0",
            is_loaded=self._is_loaded,
        )
