"""Tests for SectorRotationStrategy."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest
from strategies.stocks.sector_rotation import SectorRotationStrategy
from strategies.config import SectorRotationConfig
from strategies.base import StrategyOutput, SignalDirection


@pytest.fixture
def strategy():
    config = SectorRotationConfig(enabled=True, min_tilt_threshold=0.05)
    return SectorRotationStrategy(config=config, manager=None)


@pytest.fixture
def multi_sector_data():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n)
    symbols = ["AAPL", "JPM", "XOM", "JNJ", "NEE"]
    rows = []
    trends = {"AAPL": 0.002, "JPM": 0.003, "XOM": -0.001, "JNJ": 0.0, "NEE": -0.002}
    for sym in symbols:
        prices = [100.0]
        for i in range(n - 1):
            prices.append(prices[-1] * (1 + trends[sym] + np.random.normal(0, 0.01)))
        for i, dt in enumerate(dates):
            rows.append({
                "symbol": sym, "timestamp": dt,
                "close": prices[i], "volume": 1000000,
            })
    return pd.DataFrame(rows)


def test_returns_strategy_output(strategy, multi_sector_data):
    output = strategy.generate_signals(multi_sector_data)
    assert isinstance(output, StrategyOutput)


def test_strategy_name(strategy):
    assert strategy.name == "sector_rotation"


def test_handles_empty_data(strategy):
    output = strategy.generate_signals(pd.DataFrame({"symbol": [], "timestamp": [], "close": [], "volume": []}))
    assert isinstance(output, StrategyOutput)


def test_signal_direction_correct(strategy, multi_sector_data):
    output = strategy.generate_signals(multi_sector_data)
    for score in output.scores:
        if score.score > 0:
            assert score.direction == SignalDirection.LONG
        elif score.score < 0:
            assert score.direction == SignalDirection.SHORT


def test_confidence_range(strategy, multi_sector_data):
    output = strategy.generate_signals(multi_sector_data)
    for score in output.scores:
        assert 0.0 <= score.confidence <= 1.0


def test_has_sector_metadata(strategy, multi_sector_data):
    output = strategy.generate_signals(multi_sector_data)
    for score in output.scores:
        assert "sector" in score.metadata
        assert "strategy_type" in score.metadata
