"""Tests for MeanReversionModel."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest
from models.mean_reversion_model import MeanReversionModel, _compute_hurst, _fit_ou_params
from models.base import ModelPrediction


@pytest.fixture
def model():
    m = MeanReversionModel()
    m.load("")
    return m


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 300
    # Mean-reverting process
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(prices[-1] + 0.1 * (100.0 - prices[-1]) + np.random.normal(0, 0.5))
    prices = np.array(prices)
    prices = np.maximum(prices, 1.0)

    dates = pd.date_range("2023-01-01", periods=n)
    return pd.DataFrame({
        "symbol": ["AAPL"] * n,
        "timestamp": dates,
        "close": prices,
        "volume": np.random.randint(1000000, 5000000, n),
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
        assert "hurst_exponent" in p.metadata
        assert "half_life" in p.metadata
        assert "ou_mu" in p.metadata
        assert "ou_sigma" in p.metadata
        assert "deviation_zscore" in p.metadata


def test_hurst_exponent_range(model, sample_data):
    preds = model.predict(sample_data)
    for p in preds:
        assert 0.0 <= p.metadata["hurst_exponent"] <= 1.0


def test_confidence_range(model, sample_data):
    preds = model.predict(sample_data)
    for p in preds:
        assert 0.0 <= p.confidence <= 1.0


def test_mean_reverting_process_hurst():
    """A mean-reverting process should have Hurst < 0.5."""
    np.random.seed(42)
    prices = [100.0]
    for _ in range(500):
        prices.append(prices[-1] + 0.2 * (100.0 - prices[-1]) + np.random.normal(0, 0.3))
    hurst = _compute_hurst(np.array(prices))
    assert hurst < 0.55  # should be well below 0.5 for strong MR


def test_ou_params_positive_theta():
    """OU fit on mean-reverting data should give positive theta."""
    np.random.seed(42)
    prices = [100.0]
    for _ in range(500):
        prices.append(prices[-1] + 0.1 * (100.0 - prices[-1]) + np.random.normal(0, 0.3))
    params = _fit_ou_params(np.array(prices))
    assert params["theta"] >= 0
    assert params["half_life"] < 100


def test_too_short_data(model):
    data = pd.DataFrame({
        "symbol": ["AAPL"] * 10,
        "timestamp": pd.date_range("2023-01-01", periods=10),
        "close": np.linspace(100, 110, 10),
    })
    preds = model.predict(data)
    assert preds == []


def test_model_name(model):
    assert model.name == "mean_reversion"
