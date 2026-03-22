"""Tests for VolatilityArbitrage ensemble integration."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest
from strategies.options.strategies.vol_arb import VolatilityArbitrage
from strategies.options.config import VolArbConfig
from strategies.base import StrategyOutput, SignalDirection


@pytest.fixture
def strategy():
    config = VolArbConfig(enabled=True)
    return VolatilityArbitrage(config=config)


@pytest.fixture
def stock_data():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n)
    symbols = ["AAPL", "MSFT"]
    rows = []
    for sym in symbols:
        prices = [150.0]
        for i in range(n - 1):
            prices.append(prices[-1] * (1 + np.random.normal(0.0002, 0.015)))
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": dt,
                    "close": prices[i],
                    "volume": 2000000,
                }
            )
    return pd.DataFrame(rows)


def test_returns_strategy_output(strategy, stock_data):
    output = strategy.generate_signals(stock_data)
    assert isinstance(output, StrategyOutput)


def test_strategy_name(strategy):
    assert strategy.name == "vol_arb"


def test_handles_empty_data(strategy):
    output = strategy.generate_signals(
        pd.DataFrame({"symbol": [], "timestamp": [], "close": [], "volume": []})
    )
    assert isinstance(output, StrategyOutput)


def test_confidence_range(strategy, stock_data):
    output = strategy.generate_signals(stock_data)
    for score in output.scores:
        assert 0.0 <= score.confidence <= 1.0


def test_has_vol_metadata(strategy, stock_data):
    output = strategy.generate_signals(stock_data)
    for score in output.scores:
        assert "strategy_type" in score.metadata
        assert score.metadata["strategy_type"] == "vol_arb"


def test_implements_base_strategy(strategy):
    """Vol arb must implement BaseStrategy for ensemble integration."""
    from strategies.base import BaseStrategy

    assert isinstance(strategy, BaseStrategy)


def test_has_get_performance(strategy):
    perf = strategy.get_performance()
    assert perf.strategy_name == "vol_arb"
