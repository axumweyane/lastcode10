"""Tests for FXMomentumStrategy."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest
from strategies.fx.momentum import FXMomentumStrategy
from strategies.config import FXMomentumConfig
from strategies.base import StrategyOutput, SignalDirection


@pytest.fixture
def strategy():
    config = FXMomentumConfig(enabled=True, min_lookback_days=21, signal_threshold=0.3)
    return FXMomentumStrategy(config=config)


@pytest.fixture
def fx_data():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n)
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    # Different trends per pair
    trends = {"EURUSD": 0.001, "GBPUSD": 0.002, "USDJPY": -0.001, "AUDUSD": -0.0015}
    rows = []
    for pair in pairs:
        prices = [1.0]
        for i in range(n - 1):
            prices.append(prices[-1] * (1 + trends[pair] + np.random.normal(0, 0.003)))
        for i, dt in enumerate(dates):
            rows.append({"symbol": pair, "timestamp": dt, "close": prices[i], "volume": 0})
    return pd.DataFrame(rows)


def test_returns_strategy_output(strategy, fx_data):
    output = strategy.generate_signals(fx_data)
    assert isinstance(output, StrategyOutput)


def test_strategy_name(strategy):
    assert strategy.name == "fx_momentum"


def test_handles_empty_data(strategy):
    output = strategy.generate_signals(pd.DataFrame({"symbol": [], "timestamp": [], "close": [], "volume": []}))
    assert isinstance(output, StrategyOutput)
    assert output.scores == []


def test_signal_direction_matches_score(strategy, fx_data):
    output = strategy.generate_signals(fx_data)
    for score in output.scores:
        if score.score > 0:
            assert score.direction == SignalDirection.LONG
        elif score.score < 0:
            assert score.direction == SignalDirection.SHORT


def test_confidence_range(strategy, fx_data):
    output = strategy.generate_signals(fx_data)
    for score in output.scores:
        assert 0.0 <= score.confidence <= 1.0


def test_z_scoring_applied(strategy, fx_data):
    output = strategy.generate_signals(fx_data)
    if len(output.scores) >= 2:
        z_scores = [s.score for s in output.scores]
        mean = np.mean(z_scores)
        # After z-scoring, mean should be near 0 (but not exact due to filtering)
        assert abs(mean) < 2.0


def test_has_trend_metadata(strategy, fx_data):
    output = strategy.generate_signals(fx_data)
    for score in output.scores:
        assert "trend_consistency" in score.metadata
        assert "strategy_type" in score.metadata
        assert score.metadata["strategy_type"] == "fx_momentum"


def test_max_pairs_limit(fx_data):
    config = FXMomentumConfig(enabled=True, min_lookback_days=21, signal_threshold=0.0,
                              max_pairs_long=1, max_pairs_short=1)
    strategy = FXMomentumStrategy(config=config)
    output = strategy.generate_signals(fx_data)
    longs = [s for s in output.scores if s.direction == SignalDirection.LONG]
    shorts = [s for s in output.scores if s.direction == SignalDirection.SHORT]
    assert len(longs) <= 1
    assert len(shorts) <= 1
