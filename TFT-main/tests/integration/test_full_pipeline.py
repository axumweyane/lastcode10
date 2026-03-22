"""Integration test: run all 11 strategies on sample data and verify combined output."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import pytest

from strategies.momentum.cross_sectional import CrossSectionalMomentum
from strategies.statarb.pairs import PairsTrading
from strategies.fx.carry_trend import FXCarryTrend
from strategies.fx.momentum import FXMomentumStrategy
from strategies.fx.vol_breakout import FXVolBreakoutStrategy
from strategies.stocks.mean_reversion import MeanReversionStrategy
from strategies.stocks.sector_rotation import SectorRotationStrategy
from strategies.options.strategies.vol_arb import VolatilityArbitrage
from strategies.ensemble.combiner import EnsembleCombiner, CombinedSignal
from strategies.config import (
    MomentumConfig,
    StatArbConfig,
    EnsembleConfig,
    FXConfig,
    MeanReversionConfig,
    SectorRotationConfig,
    FXMomentumConfig,
    FXVolBreakoutConfig,
)
from strategies.options.config import VolArbConfig
from strategies.base import StrategyOutput, SignalDirection


@pytest.fixture
def stock_data():
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n)
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "BAC", "XOM"]
    rows = []
    for sym in symbols:
        prices = [100 + np.random.uniform(-20, 20)]
        for i in range(n - 1):
            prices.append(prices[-1] * (1 + np.random.normal(0.0003, 0.015)))
        for i, dt in enumerate(dates):
            p = max(prices[i], 1.0)
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": dt,
                    "open": p * 0.999,
                    "high": p * 1.01,
                    "low": p * 0.99,
                    "close": p,
                    "volume": np.random.randint(500000, 5000000),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def fx_data():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n)
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    rows = []
    for pair in pairs:
        prices = [1.0 + np.random.uniform(-0.3, 0.3)]
        for i in range(n - 1):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.005)))
        for i, dt in enumerate(dates):
            rows.append(
                {
                    "symbol": pair,
                    "timestamp": dt,
                    "close": max(prices[i], 0.1),
                    "volume": 0,
                }
            )
    return pd.DataFrame(rows)


def test_all_stock_strategies_produce_output(stock_data):
    """All stock strategies should produce StrategyOutput without crashing."""
    strategies = [
        (
            "momentum",
            CrossSectionalMomentum(
                MomentumConfig(
                    enabled=True, min_history_days=250, min_avg_dollar_volume=0
                )
            ),
        ),
        (
            "pairs",
            PairsTrading(
                StatArbConfig(
                    enabled=True, cointegration_pvalue=0.10, same_sector_only=False
                )
            ),
        ),
        (
            "mean_reversion",
            MeanReversionStrategy(
                MeanReversionConfig(
                    enabled=True, entry_zscore=1.0, hurst_threshold=0.50
                )
            ),
        ),
        (
            "sector_rotation",
            SectorRotationStrategy(
                SectorRotationConfig(enabled=True, min_tilt_threshold=0.05)
            ),
        ),
        ("vol_arb", VolatilityArbitrage(VolArbConfig(enabled=True))),
    ]

    outputs = []
    for name, strategy in strategies:
        output = strategy.generate_signals(stock_data)
        assert isinstance(
            output, StrategyOutput
        ), f"{name} failed to produce StrategyOutput"
        outputs.append(output)

    # At least some strategies should produce signals
    total_signals = sum(len(o.scores) for o in outputs)
    assert total_signals > 0, "No strategies produced any signals"


def test_fx_strategies_produce_output(fx_data):
    """All FX strategies should produce StrategyOutput."""
    strategies = [
        ("fx_carry", FXCarryTrend(FXConfig(enabled=True))),
        (
            "fx_momentum",
            FXMomentumStrategy(
                FXMomentumConfig(
                    enabled=True, min_lookback_days=21, signal_threshold=0.3
                )
            ),
        ),
        (
            "fx_vol_breakout",
            FXVolBreakoutStrategy(
                FXVolBreakoutConfig(enabled=True, lookback_days=60, squeeze_lookback=60)
            ),
        ),
    ]

    for name, strategy in strategies:
        output = strategy.generate_signals(fx_data)
        assert isinstance(output, StrategyOutput), f"{name} failed"


def test_ensemble_combines_all_strategies(stock_data, fx_data):
    """Ensemble combiner should accept outputs from all strategy types."""
    outputs = []

    # Stock strategies
    mom = CrossSectionalMomentum(
        MomentumConfig(enabled=True, min_history_days=250, min_avg_dollar_volume=0)
    )
    outputs.append(mom.generate_signals(stock_data))

    pairs = PairsTrading(
        StatArbConfig(enabled=True, cointegration_pvalue=0.10, same_sector_only=False)
    )
    outputs.append(pairs.generate_signals(stock_data))

    mr = MeanReversionStrategy(
        MeanReversionConfig(enabled=True, entry_zscore=1.0, hurst_threshold=0.50)
    )
    outputs.append(mr.generate_signals(stock_data))

    sr = SectorRotationStrategy(
        SectorRotationConfig(enabled=True, min_tilt_threshold=0.05)
    )
    outputs.append(sr.generate_signals(stock_data))

    # FX strategies
    fx_carry = FXCarryTrend(FXConfig(enabled=True))
    outputs.append(fx_carry.generate_signals(fx_data))

    fx_mom = FXMomentumStrategy(
        FXMomentumConfig(enabled=True, min_lookback_days=21, signal_threshold=0.3)
    )
    outputs.append(fx_mom.generate_signals(fx_data))

    # Options
    vol_arb = VolatilityArbitrage(VolArbConfig(enabled=True))
    outputs.append(vol_arb.generate_signals(stock_data))

    # Filter out empty outputs
    non_empty = [o for o in outputs if o.scores]

    # Combine
    combiner = EnsembleCombiner(
        EnsembleConfig(
            enabled=True,
            weighting_method="bayesian",
            max_total_positions=20,
        )
    )
    combined = combiner.combine(non_empty)

    assert isinstance(combined, list)
    if non_empty:
        assert len(combined) > 0
        for signal in combined:
            assert isinstance(signal, CombinedSignal)
            assert isinstance(signal.direction, SignalDirection)
            assert 0.0 <= signal.confidence <= 1.0


def test_strategies_handle_overlapping_symbols(stock_data):
    """Multiple strategies producing signals for the same symbols should combine cleanly."""
    mom = CrossSectionalMomentum(
        MomentumConfig(enabled=True, min_history_days=250, min_avg_dollar_volume=0)
    )
    mr = MeanReversionStrategy(
        MeanReversionConfig(enabled=True, entry_zscore=0.5, hurst_threshold=0.55)
    )

    mom_out = mom.generate_signals(stock_data)
    mr_out = mr.generate_signals(stock_data)

    combiner = EnsembleCombiner(EnsembleConfig(enabled=True, weighting_method="equal"))
    combined = combiner.combine([mom_out, mr_out])

    # Symbols that appear in both strategies should have contributing_strategies
    for signal in combined:
        assert len(signal.contributing_strategies) >= 1
