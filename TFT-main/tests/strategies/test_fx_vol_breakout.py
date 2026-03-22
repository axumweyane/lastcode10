"""Tests for FXVolBreakoutStrategy."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest
from strategies.fx.vol_breakout import FXVolBreakoutStrategy
from strategies.config import FXVolBreakoutConfig
from strategies.base import StrategyOutput, SignalDirection


@pytest.fixture
def strategy():
    config = FXVolBreakoutConfig(
        enabled=True,
        lookback_days=60,
        squeeze_lookback=60,
        bb_window=20,
        squeeze_percentile=0.15,
        momentum_window=5,
    )
    return FXVolBreakoutStrategy(config=config)


@pytest.fixture
def squeeze_data():
    """Create data with a Bollinger squeeze pattern."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2023-01-01", periods=n)

    # High vol period, then squeeze, then expansion
    prices = [1.0]
    for i in range(n - 1):
        if i < 60:
            vol = 0.01  # normal vol
        elif i < 120:
            vol = 0.002  # squeeze (very low vol)
        else:
            vol = 0.015  # expansion
        drift = 0.0005 if i > 100 else 0.0
        prices.append(prices[-1] * (1 + drift + np.random.normal(0, vol)))

    return pd.DataFrame(
        {
            "symbol": ["EURUSD"] * n,
            "timestamp": dates,
            "close": prices,
            "volume": [0] * n,
        }
    )


def test_returns_strategy_output(strategy, squeeze_data):
    output = strategy.generate_signals(squeeze_data)
    assert isinstance(output, StrategyOutput)


def test_strategy_name(strategy):
    assert strategy.name == "fx_vol_breakout"


def test_handles_empty_data(strategy):
    output = strategy.generate_signals(
        pd.DataFrame({"symbol": [], "timestamp": [], "close": [], "volume": []})
    )
    assert isinstance(output, StrategyOutput)
    assert output.scores == []


def test_handles_short_data(strategy):
    data = pd.DataFrame(
        {
            "symbol": ["EURUSD"] * 10,
            "timestamp": pd.date_range("2023-01-01", periods=10),
            "close": [1.0] * 10,
            "volume": [0] * 10,
        }
    )
    output = strategy.generate_signals(data)
    assert output.scores == []


def test_confidence_range(strategy, squeeze_data):
    output = strategy.generate_signals(squeeze_data)
    for score in output.scores:
        assert 0.0 <= score.confidence <= 1.0


def test_has_breakout_metadata(strategy, squeeze_data):
    output = strategy.generate_signals(squeeze_data)
    for score in output.scores:
        assert "strategy_type" in score.metadata
        assert score.metadata["strategy_type"] == "fx_vol_breakout"
        assert "bandwidth" in score.metadata
        assert "squeeze_intensity" in score.metadata


def test_signal_direction_correct(strategy, squeeze_data):
    output = strategy.generate_signals(squeeze_data)
    for score in output.scores:
        if score.score > 0:
            assert score.direction == SignalDirection.LONG
        elif score.score < 0:
            assert score.direction == SignalDirection.SHORT
