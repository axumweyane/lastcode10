"""Tests for MacroRegimeModel."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from models.macro_model import MacroRegimeModel, SECTOR_MAP, REGIME_SECTOR_TILTS
from models.base import ModelPrediction


@pytest.fixture
def model():
    m = MacroRegimeModel()
    m.load("")
    return m


@pytest.fixture
def sample_data():
    n = 100
    dates = pd.date_range("2023-01-01", periods=n)
    symbols = ["AAPL", "JPM", "XOM"]
    rows = []
    for sym in symbols:
        for i, dt in enumerate(dates):
            rows.append({
                "symbol": sym, "timestamp": dt,
                "close": 100 + i * 0.1 + np.random.normal(0, 1),
                "volume": 1000000,
            })
    return pd.DataFrame(rows)


def test_model_name(model):
    assert model.name == "macro_regime"


def test_asset_class(model):
    assert model.asset_class == "cross_asset"


def test_handles_empty_data(model):
    preds = model.predict(pd.DataFrame())
    assert preds == []


def test_sector_map_coverage():
    """SECTOR_MAP should cover major symbols."""
    assert "AAPL" in SECTOR_MAP
    assert "JPM" in SECTOR_MAP
    assert "XOM" in SECTOR_MAP


def test_regime_sector_tilts_structure():
    """Each regime should have sector tilts."""
    for regime, tilts in REGIME_SECTOR_TILTS.items():
        assert isinstance(tilts, dict)
        assert len(tilts) > 0


def test_confidence_range(model, sample_data):
    # Mock the yfinance fetch to avoid network calls
    mock_macro = {
        "yield_spread_2y10y": 0.8,
        "yield_spread_3m10y": 1.2,
        "rate_trend": 0.01,
        "dxy_momentum": 0.005,
        "curve_regime": "steepening_rising",
        "current_10y": 4.5,
    }
    with patch.object(model, '_fetch_macro_data', return_value=mock_macro):
        preds = model.predict(sample_data)
        for p in preds:
            assert 0.0 <= p.confidence <= 1.0


def test_predictions_have_metadata(model, sample_data):
    mock_macro = {
        "yield_spread_2y10y": 0.8,
        "yield_spread_3m10y": 1.2,
        "rate_trend": 0.01,
        "dxy_momentum": 0.005,
        "curve_regime": "steepening_rising",
        "current_10y": 4.5,
    }
    with patch.object(model, '_fetch_macro_data', return_value=mock_macro):
        preds = model.predict(sample_data)
        assert len(preds) > 0
        for p in preds:
            assert isinstance(p, ModelPrediction)
            assert "yield_spread" in p.metadata
            assert "curve_regime" in p.metadata
            assert "sector" in p.metadata


def test_returns_empty_when_no_macro_data(model, sample_data):
    with patch.object(model, '_fetch_macro_data', return_value=None):
        preds = model.predict(sample_data)
        assert preds == []
