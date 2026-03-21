"""
Model 10: Microstructure Model — Daily-frequency volume profile and order flow analysis.

Analyzes volume patterns, VWAP deviations, and buying/selling pressure
from daily OHLCV data to score microstructure quality per symbol.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from models.base import BaseTFTModel, ModelInfo, ModelPrediction

logger = logging.getLogger(__name__)


class MicrostructureModel(BaseTFTModel):
    """
    Daily-frequency microstructure analysis.

    Signals:
      - Abnormal volume (>2x 20-day average)
      - VWAP deviation (close vs estimated VWAP)
      - Close Location Value (buying/selling pressure)
      - Volume trend (accumulation vs distribution)
    """

    def __init__(self, volume_lookback: int = 20):
        self._volume_lookback = volume_lookback
        self._is_loaded = False

    @property
    def name(self) -> str:
        return "microstructure"

    @property
    def asset_class(self) -> str:
        return "stocks"

    def prepare_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        required = {"symbol", "close", "volume"}
        if not required.issubset(raw_data.columns):
            return pd.DataFrame()
        return raw_data.copy()

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        return {"status": "no_training_needed", "note": "statistical microstructure model"}

    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        if data.empty:
            return []

        required = {"symbol", "close", "volume"}
        if not required.issubset(data.columns):
            return []

        has_ohlc = all(c in data.columns for c in ("open", "high", "low"))
        predictions = []

        for symbol in data["symbol"].unique():
            sym_data = data[data["symbol"] == symbol].sort_values(
                "timestamp" if "timestamp" in data.columns else data.columns[0]
            )

            if len(sym_data) < self._volume_lookback + 5:
                continue

            close = sym_data["close"].values
            volume = sym_data["volume"].values.astype(float)

            # 1. Relative volume (current vs 20-day average)
            avg_volume = np.mean(volume[-self._volume_lookback:])
            current_volume = volume[-1]
            relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0

            # 2. VWAP deviation (estimated from OHLC)
            if has_ohlc:
                high = sym_data["high"].values
                low = sym_data["low"].values
                open_ = sym_data["open"].values
                # Typical price as VWAP proxy
                typical = (high[-1] + low[-1] + close[-1]) / 3.0
                vwap_deviation = (close[-1] - typical) / typical if typical > 0 else 0.0

                # Close Location Value: where close falls in the day's range
                day_range = high[-1] - low[-1]
                if day_range > 0:
                    clv = (close[-1] - low[-1]) / day_range  # 0 = closed at low, 1 = closed at high
                else:
                    clv = 0.5
            else:
                vwap_deviation = 0.0
                clv = 0.5

            # 3. Buying pressure (CLV > 0.5 = buyers in control)
            buying_pressure = (clv - 0.5) * 2.0  # [-1, +1]

            # 4. Volume trend (5-day vs 20-day average volume)
            recent_vol = np.mean(volume[-5:])
            volume_trend = (recent_vol / avg_volume - 1.0) if avg_volume > 0 else 0.0

            # 5. Accumulation/Distribution indicator (simplified)
            if has_ohlc and len(sym_data) >= 10:
                high_arr = sym_data["high"].values[-10:]
                low_arr = sym_data["low"].values[-10:]
                close_arr = close[-10:]
                vol_arr = volume[-10:]
                ranges = high_arr - low_arr
                mfm = np.where(ranges > 0, ((close_arr - low_arr) - (high_arr - close_arr)) / ranges, 0.0)
                ad_line = np.cumsum(mfm * vol_arr)
                ad_trend = (ad_line[-1] - ad_line[0]) / (abs(ad_line[0]) + 1e-10)
            else:
                ad_trend = 0.0

            # Composite microstructure signal
            # Abnormal volume + buying pressure + positive AD = bullish
            signal = (
                0.3 * np.clip(buying_pressure, -1, 1) +
                0.2 * np.clip(vwap_deviation * 100, -1, 1) +
                0.2 * np.clip((relative_volume - 1.0) * buying_pressure, -1, 1) +
                0.3 * np.clip(ad_trend, -1, 1)
            )

            confidence = min(0.3 + 0.2 * min(relative_volume, 3.0) / 3.0 + 0.3, 0.9)

            predictions.append(ModelPrediction(
                symbol=symbol,
                predicted_value=float(signal),
                lower_bound=float(signal - 0.3),
                upper_bound=float(signal + 0.3),
                confidence=confidence,
                horizon_days=1,
                model_name=self.name,
                metadata={
                    "relative_volume": round(relative_volume, 4),
                    "vwap_deviation": round(vwap_deviation, 6),
                    "buying_pressure": round(buying_pressure, 4),
                    "close_location_value": round(clv, 4),
                    "volume_trend": round(volume_trend, 4),
                    "ad_trend": round(ad_trend, 4),
                    "is_abnormal_volume": relative_volume > 2.0,
                },
            ))

        return predictions

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> bool:
        self._is_loaded = True
        logger.info("MicrostructureModel ready (statistical, no weights)")
        return True

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            asset_class=self.asset_class,
            version="1.0",
            is_loaded=self._is_loaded,
        )
