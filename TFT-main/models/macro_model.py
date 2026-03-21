"""
Model 9: Macro Regime Model — Yield curve analysis + macro indicators.

Analyzes yield curve shape, rate trends, and dollar strength to classify
the macro regime and produce per-sector tilts.
Input: macro time series from yfinance (^TNX, ^FVX, ^IRX, DXY).
Output: macro regime classification + per-sector tilts.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from models.base import BaseTFTModel, ModelInfo, ModelPrediction

logger = logging.getLogger(__name__)

# GICS-like sector classification for common symbols
SECTOR_MAP = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "META": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "CRM": "Technology", "ADBE": "Technology", "INTC": "Technology",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "WFC": "Financials", "C": "Financials",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "WMT": "Consumer Staples", "COST": "Consumer Staples",
    # Industrials
    "CAT": "Industrials", "BA": "Industrials", "HON": "Industrials",
    "UPS": "Industrials", "GE": "Industrials",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
}

# Regime -> sector tilts (positive = overweight, negative = underweight)
REGIME_SECTOR_TILTS = {
    "steepening_rising": {
        "Financials": 0.6, "Energy": 0.4, "Materials": 0.3, "Industrials": 0.2,
        "Technology": -0.1, "Utilities": -0.4, "Real Estate": -0.3,
        "Consumer Staples": -0.2, "Healthcare": 0.0, "Consumer Discretionary": 0.1,
    },
    "steepening_falling": {
        "Technology": 0.4, "Consumer Discretionary": 0.3, "Real Estate": 0.3,
        "Financials": -0.1, "Energy": -0.2, "Materials": 0.0, "Industrials": 0.1,
        "Utilities": 0.2, "Consumer Staples": 0.1, "Healthcare": 0.1,
    },
    "flattening_rising": {
        "Financials": 0.2, "Technology": 0.1, "Healthcare": 0.3,
        "Consumer Staples": 0.2, "Utilities": -0.2, "Real Estate": -0.4,
        "Energy": 0.1, "Materials": 0.0, "Industrials": 0.0, "Consumer Discretionary": -0.1,
    },
    "inverted": {
        "Utilities": 0.5, "Healthcare": 0.4, "Consumer Staples": 0.4,
        "Technology": -0.2, "Financials": -0.3, "Energy": -0.2,
        "Consumer Discretionary": -0.3, "Industrials": -0.2, "Materials": -0.1,
        "Real Estate": 0.0,
    },
    "neutral": {
        sector: 0.0 for sector in set(SECTOR_MAP.values())
    },
}


class MacroRegimeModel(BaseTFTModel):
    """
    Macro regime detection via yield curve analysis and rate trends.
    """

    def __init__(self):
        self._is_loaded = False

    @property
    def name(self) -> str:
        return "macro_regime"

    @property
    def asset_class(self) -> str:
        return "cross_asset"

    def prepare_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        return raw_data.copy()

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        return {"status": "no_training_needed", "note": "rule-based macro model"}

    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        macro_data = self._fetch_macro_data(data)
        if macro_data is None:
            return []

        yield_spread = macro_data["yield_spread_2y10y"]
        rate_trend = macro_data["rate_trend"]
        dxy_momentum = macro_data["dxy_momentum"]
        curve_regime = macro_data["curve_regime"]
        sector_tilts = REGIME_SECTOR_TILTS.get(curve_regime, REGIME_SECTOR_TILTS["neutral"])

        # Generate predictions for each symbol based on its sector tilt
        predictions = []
        symbols = data["symbol"].unique() if "symbol" in data.columns else []

        for symbol in symbols:
            sector = SECTOR_MAP.get(symbol, "Unknown")
            tilt = sector_tilts.get(sector, 0.0)

            # DXY adjustment: strong dollar hurts EM-exposed industrials
            if sector in ("Industrials", "Materials") and dxy_momentum > 0.02:
                tilt -= 0.1

            # Regime stability -> confidence
            confidence = min(abs(yield_spread) / 2.0 + 0.3, 0.9)

            predictions.append(ModelPrediction(
                symbol=symbol,
                predicted_value=tilt,
                lower_bound=tilt - 0.2,
                upper_bound=tilt + 0.2,
                confidence=confidence,
                horizon_days=21,  # macro regime changes slowly
                model_name=self.name,
                metadata={
                    "yield_spread": round(yield_spread, 4),
                    "curve_regime": curve_regime,
                    "rate_trend": round(rate_trend, 4),
                    "dxy_momentum": round(dxy_momentum, 4),
                    "sector": sector,
                    "sector_tilt": round(tilt, 4),
                    "sector_tilts": {k: round(v, 3) for k, v in sector_tilts.items()},
                },
            ))

        return predictions

    def _fetch_macro_data(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Compute macro indicators from available data or fetch via yfinance."""
        try:
            import yfinance as yf

            # Fetch yield curve data
            tickers = yf.download(
                ["^TNX", "^FVX", "^IRX"],
                period="120d", auto_adjust=True, progress=False,
            )

            if tickers.empty:
                return None

            # 10Y yield
            try:
                tnx = tickers["Close"]["^TNX"].dropna()
            except (KeyError, TypeError):
                return None

            # 5Y yield
            try:
                fvx = tickers["Close"]["^FVX"].dropna()
            except (KeyError, TypeError):
                fvx = tnx  # fallback

            # 3M yield (13-week T-bill)
            try:
                irx = tickers["Close"]["^IRX"].dropna()
            except (KeyError, TypeError):
                irx = pd.Series([4.0])  # fallback

            if len(tnx) < 10:
                return None

            current_10y = float(tnx.iloc[-1])
            current_2y = float(fvx.iloc[-1])  # using 5Y as proxy for 2Y
            current_3m = float(irx.iloc[-1]) if len(irx) > 0 else 4.0

            yield_spread_2y10y = current_10y - current_2y
            yield_spread_3m10y = current_10y - current_3m

            # Rate trend: 21-day change in 10Y
            if len(tnx) >= 22:
                rate_trend = float(tnx.iloc[-1] - tnx.iloc[-22]) / 100
            else:
                rate_trend = 0.0

            # DXY momentum (use UUP ETF as proxy if DXY not available)
            try:
                dxy_data = yf.download("UUP", period="63d", auto_adjust=True, progress=False)
                if not dxy_data.empty:
                    dxy_close = dxy_data["Close"].dropna()
                    if len(dxy_close) >= 22:
                        dxy_momentum = float(dxy_close.iloc[-1]) / float(dxy_close.iloc[-22]) - 1.0
                    else:
                        dxy_momentum = 0.0
                else:
                    dxy_momentum = 0.0
            except Exception:
                dxy_momentum = 0.0

            # Classify regime
            if yield_spread_2y10y < -0.2:
                curve_regime = "inverted"
            elif yield_spread_2y10y > 0.5 and rate_trend > 0:
                curve_regime = "steepening_rising"
            elif yield_spread_2y10y > 0.5 and rate_trend <= 0:
                curve_regime = "steepening_falling"
            elif yield_spread_2y10y > -0.2 and rate_trend > 0:
                curve_regime = "flattening_rising"
            else:
                curve_regime = "neutral"

            return {
                "yield_spread_2y10y": yield_spread_2y10y,
                "yield_spread_3m10y": yield_spread_3m10y,
                "rate_trend": rate_trend,
                "dxy_momentum": dxy_momentum,
                "curve_regime": curve_regime,
                "current_10y": current_10y,
            }

        except ImportError:
            logger.warning("yfinance not available for macro data")
            return None
        except Exception as e:
            logger.warning("Macro data fetch failed: %s", e)
            return None

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> bool:
        self._is_loaded = True
        logger.info("MacroRegimeModel ready (rule-based, no weights)")
        return True

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            asset_class=self.asset_class,
            version="1.0",
            is_loaded=self._is_loaded,
        )
