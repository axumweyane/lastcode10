"""Tests for MeanReversionStrategy."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest
from strategies.stocks.mean_reversion import MeanReversionStrategy
from strategies.config import MeanReversionConfig
from strategies.base import StrategyOutput, SignalDirection


@pytest.fixture
def strategy():
    config = MeanReversionConfig(enabled=True, entry_zscore=1.0, hurst_threshold=0.50)
    return MeanReversionStrategy(config=config, manager=None)


@pytest.fixture
def mean_reverting_data():
    """Data with a strongly mean-reverting process."""
    np.random.seed(42)
    n = 300
    prices = [100.0]
    for _ in range(n - 1):
        # Strong mean reversion toward 100
        prices.append(
            prices[-1] + 0.15 * (100.0 - prices[-1]) + np.random.normal(0, 0.3)
        )
    prices = np.maximum(np.array(prices), 1.0)

    dates = pd.date_range("2023-01-01", periods=n)
    return pd.DataFrame(
        {
            "symbol": ["AAPL"] * n,
            "timestamp": dates,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, n),
        }
    )


def test_returns_strategy_output(strategy, mean_reverting_data):
    output = strategy.generate_signals(mean_reverting_data)
    assert isinstance(output, StrategyOutput)


def test_strategy_name(strategy):
    assert strategy.name == "mean_reversion"


def test_handles_empty_data(strategy):
    output = strategy.generate_signals(
        pd.DataFrame({"symbol": [], "timestamp": [], "close": [], "volume": []})
    )
    assert isinstance(output, StrategyOutput)
    assert output.scores == []


def test_signal_direction_correct(strategy, mean_reverting_data):
    output = strategy.generate_signals(mean_reverting_data)
    for score in output.scores:
        if score.score > 0:
            assert score.direction == SignalDirection.LONG
        elif score.score < 0:
            assert score.direction == SignalDirection.SHORT


def test_confidence_range(strategy, mean_reverting_data):
    output = strategy.generate_signals(mean_reverting_data)
    for score in output.scores:
        assert 0.0 <= score.confidence <= 1.0


def test_has_metadata(strategy, mean_reverting_data):
    output = strategy.generate_signals(mean_reverting_data)
    for score in output.scores:
        assert "hurst_exponent" in score.metadata
        assert "half_life" in score.metadata
        assert "deviation_zscore" in score.metadata


def test_too_short_data_returns_empty(strategy):
    data = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 20,
            "timestamp": pd.date_range("2023-01-01", periods=20),
            "close": np.linspace(100, 110, 20),
            "volume": [1000000] * 20,
        }
    )
    output = strategy.generate_signals(data)
    assert output.scores == []
