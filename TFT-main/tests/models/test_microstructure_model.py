"""Tests for MicrostructureModel."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest
from models.microstructure_model import MicrostructureModel
from models.base import ModelPrediction


@pytest.fixture
def model():
    m = MicrostructureModel()
    m.load("")
    return m


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2023-01-01", periods=n)
    close = 100 + np.cumsum(np.random.normal(0.01, 1, n))
    high = close + np.abs(np.random.normal(0, 0.5, n))
    low = close - np.abs(np.random.normal(0, 0.5, n))
    open_ = close + np.random.normal(0, 0.3, n)
    volume = np.random.randint(500000, 5000000, n)

    return pd.DataFrame({
        "symbol": ["AAPL"] * n,
        "timestamp": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_returns_model_prediction_type(model, sample_data):
    preds = model.predict(sample_data)
    assert all(isinstance(p, ModelPrediction) for p in preds)


def test_handles_empty_data(model):
    preds = model.predict(pd.DataFrame())
    assert preds == []


def test_handles_missing_columns(model):
    preds = model.predict(pd.DataFrame({"symbol": ["AAPL"]}))
    assert preds == []


def test_output_has_required_metadata(model, sample_data):
    preds = model.predict(sample_data)
    assert len(preds) > 0
    for p in preds:
        assert "relative_volume" in p.metadata
        assert "vwap_deviation" in p.metadata
        assert "buying_pressure" in p.metadata
        assert "close_location_value" in p.metadata


def test_confidence_range(model, sample_data):
    preds = model.predict(sample_data)
    for p in preds:
        assert 0.0 <= p.confidence <= 1.0


def test_too_short_data(model):
    data = pd.DataFrame({
        "symbol": ["AAPL"] * 5,
        "timestamp": pd.date_range("2023-01-01", periods=5),
        "close": [100, 101, 102, 103, 104],
        "volume": [1000000] * 5,
    })
    preds = model.predict(data)
    assert preds == []


def test_model_name(model):
    assert model.name == "microstructure"


def test_asset_class(model):
    assert model.asset_class == "stocks"


def test_relative_volume_positive(model, sample_data):
    preds = model.predict(sample_data)
    for p in preds:
        assert p.metadata["relative_volume"] > 0


def test_close_location_value_range(model, sample_data):
    preds = model.predict(sample_data)
    for p in preds:
        assert 0.0 <= p.metadata["close_location_value"] <= 1.0
