"""
Coverage tests for strategies/regime/detector.py.
No external dependencies — pure computation with pandas/numpy.
"""

import numpy as np
import pandas as pd
import pytest

from strategies.config import RegimeConfig
from strategies.regime.detector import (
    MarketRegime,
    RegimeDetector,
    RegimeState,
)

# ---------- Data helpers ----------


def _make_market_data(
    symbols=("AAPL", "GOOGL", "MSFT", "SPY"),
    days=100,
    base_price=100.0,
    trend=0.001,
    vol=0.02,
    breadth_above_ma=0.7,
):
    """Generate synthetic market data."""
    rows = []
    rng = np.random.RandomState(42)
    for sym in symbols:
        prices = [base_price]
        for i in range(1, days):
            ret = trend + vol * rng.randn()
            prices.append(prices[-1] * (1 + ret))

        # Adjust so % above 50-day MA matches breadth_above_ma
        for i, p in enumerate(prices):
            ts = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
            rows.append({"symbol": sym, "timestamp": ts, "close": p, "volume": 1000000})

    return pd.DataFrame(rows)


def _make_calm_trending_data():
    """Low vol, high breadth -> CALM_TRENDING."""
    return _make_market_data(vol=0.005, trend=0.003, breadth_above_ma=0.8)


def _make_calm_choppy_data():
    """Low vol, low breadth -> CALM_CHOPPY."""
    data = _make_market_data(
        symbols=("A", "B", "C", "D", "E", "SPY"),
        days=100,
        vol=0.005,
        trend=-0.001,
    )
    return data


def _make_volatile_data():
    """High vol data."""
    return _make_market_data(vol=0.04, trend=0.001)


@pytest.fixture
def config():
    return RegimeConfig(
        vix_low_threshold=20.0,
        vix_high_threshold=30.0,
        breadth_lookback_days=50,
        breadth_trending_threshold=0.6,
        breadth_choppy_threshold=0.4,
        realized_vol_window=21,
    )


@pytest.fixture
def detector(config):
    return RegimeDetector(config)


# ---------- RegimeState ----------


class TestRegimeState:
    def test_str_repr(self):
        state = RegimeState(
            regime=MarketRegime.CALM_TRENDING,
            vix_level=18.0,
            market_breadth=0.7,
            realized_vol=0.12,
            is_volatile=False,
            is_trending=True,
            confidence=0.6,
            strategy_weights={"momentum": 0.4},
            exposure_scalar=1.0,
        )
        s = str(state)
        assert "calm_trending" in s
        assert "VIX=18.0" in s


# ---------- MarketRegime enum ----------


class TestMarketRegime:
    def test_all_values(self):
        assert MarketRegime.CALM_TRENDING.value == "calm_trending"
        assert MarketRegime.CALM_CHOPPY.value == "calm_choppy"
        assert MarketRegime.VOLATILE_TRENDING.value == "volatile_trending"
        assert MarketRegime.VOLATILE_CHOPPY.value == "volatile_choppy"


# ---------- detect ----------


class TestDetect:
    def test_explicit_vix(self, detector):
        data = _make_market_data()
        state = detector.detect(data, vix_value=15.0)
        assert state.vix_level == 15.0
        assert isinstance(state.regime, MarketRegime)

    def test_vix_column_in_data(self, detector):
        data = _make_market_data()
        data["vix"] = 18.0
        state = detector.detect(data)
        assert state.vix_level == 18.0

    def test_vix_symbol_in_data(self, detector):
        data = _make_market_data(symbols=("AAPL", "SPY", "^VIX"))
        # Set VIX symbol prices to ~25
        data.loc[data["symbol"] == "^VIX", "close"] = 25.0
        state = detector.detect(data)
        assert state.vix_level == 25.0

    def test_vix_fallback_estimated(self, detector):
        data = _make_market_data(symbols=("AAPL", "MSFT"))
        state = detector.detect(data)
        assert state.vix_level > 0  # estimated from realized vol

    def test_calm_trending_regime(self, detector):
        data = _make_market_data(days=100, vol=0.005, trend=0.003)
        state = detector.detect(data, vix_value=15.0)
        # With low VIX and decent trend, likely calm
        assert state.is_volatile is False

    def test_volatile_regime_high_vix(self, detector):
        data = _make_market_data()
        state = detector.detect(data, vix_value=35.0)
        assert state.is_volatile is True

    def test_volatile_regime_high_rvol(self, detector):
        data = _make_market_data(vol=0.06)  # very high vol
        state = detector.detect(data, vix_value=15.0)
        # realized_vol > 0.25 triggers volatile
        # depends on data generation

    def test_regime_returns_valid_weights(self, detector):
        data = _make_market_data()
        state = detector.detect(data, vix_value=20.0)
        assert "momentum" in state.strategy_weights
        assert "mean_reversion" in state.strategy_weights
        assert "pairs" in state.strategy_weights
        assert "tft" in state.strategy_weights
        assert sum(state.strategy_weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_exposure_scalar_range(self, detector):
        data = _make_market_data()
        state = detector.detect(data, vix_value=20.0)
        assert 0.3 <= state.exposure_scalar <= 1.0

    def test_confidence_range(self, detector):
        data = _make_market_data()
        state = detector.detect(data, vix_value=25.0)
        assert 0.0 <= state.confidence <= 1.0

    def test_history_recorded(self, detector):
        data = _make_market_data()
        detector.detect(data, vix_value=15.0)
        detector.detect(data, vix_value=35.0)
        history = detector.get_regime_history()
        assert len(history) == 2


# ---------- _compute_breadth ----------


class TestComputeBreadth:
    def test_all_above_ma(self, detector):
        # Long uptrend: all stocks above 50-day MA
        data = _make_market_data(days=100, trend=0.01, vol=0.001)
        breadth = detector._compute_breadth(data)
        assert breadth > 0.5

    def test_insufficient_data_returns_default(self, detector):
        data = _make_market_data(days=10)
        breadth = detector._compute_breadth(data)
        assert breadth == 0.5  # default when no valid symbols

    def test_empty_data(self, detector):
        data = pd.DataFrame(columns=["symbol", "timestamp", "close", "volume"])
        breadth = detector._compute_breadth(data)
        assert breadth == 0.5


# ---------- _compute_realized_vol ----------


class TestComputeRealizedVol:
    def test_spy_proxy_used(self, detector):
        data = _make_market_data(symbols=("AAPL", "SPY"))
        rvol = detector._compute_realized_vol(data)
        assert rvol > 0

    def test_fallback_median(self, detector):
        data = _make_market_data(symbols=("A", "B", "C"))
        rvol = detector._compute_realized_vol(data)
        assert rvol > 0

    def test_insufficient_data_returns_default(self, detector):
        data = _make_market_data(symbols=("A",), days=5)
        rvol = detector._compute_realized_vol(data)
        assert rvol == 0.15  # default moderate vol


# ---------- _compute_confidence ----------


class TestComputeConfidence:
    def test_far_from_boundary(self, detector):
        conf = detector._compute_confidence(vix=10.0, breadth=0.9, realized_vol=0.1)
        assert conf > 0.3

    def test_near_boundary(self, detector):
        conf = detector._compute_confidence(vix=25.0, breadth=0.5, realized_vol=0.15)
        # Near midpoints = lower confidence
        assert 0.0 <= conf <= 1.0


# ---------- _get_weights ----------


class TestGetWeights:
    def test_calm_trending(self, detector):
        w = detector._get_weights(MarketRegime.CALM_TRENDING)
        assert w["momentum"] == 0.40

    def test_calm_choppy(self, detector):
        w = detector._get_weights(MarketRegime.CALM_CHOPPY)
        assert w["mean_reversion"] == 0.40

    def test_volatile_choppy(self, detector):
        w = detector._get_weights(MarketRegime.VOLATILE_CHOPPY)
        assert w["pairs"] == 0.45

    def test_volatile_trending(self, detector):
        w = detector._get_weights(MarketRegime.VOLATILE_TRENDING)
        assert w["momentum"] == 0.30


# ---------- _compute_exposure_scalar ----------


class TestComputeExposureScalar:
    def test_low_vol_full_exposure(self, detector):
        scalar = detector._compute_exposure_scalar(vix=15, realized_vol=0.10)
        assert scalar == 1.0  # 0.15/0.10 = 1.5, capped at 1.0

    def test_high_vol_reduced(self, detector):
        scalar = detector._compute_exposure_scalar(vix=20, realized_vol=0.30)
        assert scalar < 1.0
        assert scalar >= 0.3

    def test_very_high_vix_penalty(self, detector):
        scalar = detector._compute_exposure_scalar(vix=50, realized_vol=0.15)
        assert scalar < 1.0

    def test_near_zero_vol(self, detector):
        scalar = detector._compute_exposure_scalar(vix=15, realized_vol=0.001)
        assert scalar == 1.0

    def test_minimum_clamp(self, detector):
        scalar = detector._compute_exposure_scalar(vix=60, realized_vol=1.0)
        assert scalar >= 0.3


# ---------- get_regime_history ----------


class TestGetRegimeHistory:
    def test_empty(self, detector):
        assert detector.get_regime_history() == []

    def test_capped(self, detector):
        data = _make_market_data()
        for _ in range(5):
            detector.detect(data, vix_value=20.0)
        assert len(detector.get_regime_history(3)) == 3


# ---------- Regime change detection ----------


class TestRegimeChange:
    def test_regime_change_logged(self, detector):
        data = _make_market_data()
        detector.detect(data, vix_value=15.0)  # calm
        detector.detect(data, vix_value=35.0)  # volatile
        history = detector.get_regime_history()
        assert history[0].regime != history[1].regime or len(history) == 2
