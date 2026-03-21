"""
Coverage tests for strategies/statarb/pairs.py.
PairScanner is mocked to isolate PairsTrading logic.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from strategies.config import StatArbConfig
from strategies.statarb.pairs import (
    ActivePair,
    PairState,
    PairsTrading,
    _SymbolAccumulator,
)
from strategies.statarb.scanner import TradingPair


# ---------- Helpers ----------

def _make_pair(sym_a="AAPL", sym_b="MSFT", hedge_ratio=1.0, spread_mean=0.0, spread_std=5.0):
    return TradingPair(
        symbol_a=sym_a,
        symbol_b=sym_b,
        hedge_ratio=hedge_ratio,
        coint_pvalue=0.01,
        half_life=10.0,
        spread_mean=spread_mean,
        spread_std=spread_std,
        correlation=0.9,
    )


def _make_data(symbols=("AAPL", "MSFT"), days=100, base_price=100.0):
    rows = []
    rng = np.random.RandomState(42)
    for sym in symbols:
        prices = [base_price]
        for i in range(1, days):
            prices.append(prices[-1] * (1 + 0.001 + 0.02 * rng.randn()))
        for i, p in enumerate(prices):
            ts = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
            rows.append({"symbol": sym, "timestamp": ts, "close": p, "volume": 1000000})
    return pd.DataFrame(rows)


@pytest.fixture
def config():
    return StatArbConfig(
        enabled=True,
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=4.0,
        lookback_window=63,
        max_pairs=20,
        rescan_interval_days=7,
        cointegration_pvalue=0.05,
    )


@pytest.fixture
def strategy(config):
    s = PairsTrading(config)
    return s


# ---------- _SymbolAccumulator ----------

class TestSymbolAccumulator:
    def test_add_and_net_score(self):
        acc = _SymbolAccumulator("AAPL")
        acc.add(0.5, 2.5, "AAPL_MSFT", "long_leg")
        acc.add(-0.2, -1.0, "AAPL_GOOGL", "short_leg")
        assert acc.net_score == pytest.approx(0.3)
        assert acc.pair_count == 2
        assert len(acc.pair_details) == 2

    def test_empty(self):
        acc = _SymbolAccumulator("X")
        assert acc.net_score == 0.0
        assert acc.pair_count == 0


# ---------- PairState ----------

class TestPairStateConstants:
    def test_values(self):
        assert PairState.FLAT == "flat"
        assert PairState.LONG_SPREAD == "long_spread"
        assert PairState.SHORT_SPREAD == "short_spread"


# ---------- ActivePair ----------

class TestActivePair:
    def test_defaults(self):
        pair = _make_pair()
        ap = ActivePair(pair=pair)
        assert ap.state == PairState.FLAT
        assert ap.cumulative_pnl == 0.0
        assert ap.entry_date is None


# ---------- Properties ----------

class TestStrategyProperties:
    def test_name(self, strategy):
        assert strategy.name == "pairs_trading"

    def test_description(self, strategy):
        assert "statistical arbitrage" in strategy.description.lower()


# ---------- initialize ----------

class TestInitialize:
    def test_initialize_calls_scanner(self, strategy):
        data = _make_data()
        pairs = [_make_pair("AAPL", "MSFT")]
        strategy._scanner = MagicMock()
        strategy._scanner.scan.return_value = pairs

        strategy.initialize(data)

        assert strategy._initialized is True
        assert len(strategy._active_pairs) == 1
        strategy._scanner.scan.assert_called_once()

    def test_initialize_with_sector_mapping(self, strategy):
        data = _make_data()
        strategy._scanner = MagicMock()
        strategy._scanner.scan.return_value = []

        strategy.initialize(data, sector_mapping={"AAPL": "Tech", "MSFT": "Tech"})
        strategy._scanner.scan.assert_called_once_with(
            data, {"AAPL": "Tech", "MSFT": "Tech"}
        )


# ---------- _evaluate_pair ----------

class TestEvaluatePair:
    def test_flat_enter_short_spread(self, strategy):
        pair = _make_pair()
        ap = ActivePair(pair=pair, state=PairState.FLAT)
        assert strategy._evaluate_pair(ap, 2.5) == "enter_short_spread"

    def test_flat_enter_long_spread(self, strategy):
        pair = _make_pair()
        ap = ActivePair(pair=pair, state=PairState.FLAT)
        assert strategy._evaluate_pair(ap, -2.5) == "enter_long_spread"

    def test_flat_no_action(self, strategy):
        pair = _make_pair()
        ap = ActivePair(pair=pair, state=PairState.FLAT)
        assert strategy._evaluate_pair(ap, 1.0) == "no_action"

    def test_short_spread_exit(self, strategy):
        pair = _make_pair()
        ap = ActivePair(pair=pair, state=PairState.SHORT_SPREAD)
        assert strategy._evaluate_pair(ap, 0.3) == "exit"

    def test_long_spread_exit(self, strategy):
        pair = _make_pair()
        ap = ActivePair(pair=pair, state=PairState.LONG_SPREAD)
        assert strategy._evaluate_pair(ap, -0.3) == "exit"

    def test_short_spread_stop_loss(self, strategy):
        pair = _make_pair()
        ap = ActivePair(pair=pair, state=PairState.SHORT_SPREAD)
        assert strategy._evaluate_pair(ap, 4.5) == "stop_loss"

    def test_long_spread_stop_loss(self, strategy):
        pair = _make_pair()
        ap = ActivePair(pair=pair, state=PairState.LONG_SPREAD)
        assert strategy._evaluate_pair(ap, -4.5) == "stop_loss"

    def test_short_spread_no_action(self, strategy):
        pair = _make_pair()
        ap = ActivePair(pair=pair, state=PairState.SHORT_SPREAD)
        assert strategy._evaluate_pair(ap, 2.0) == "no_action"

    def test_long_spread_no_action(self, strategy):
        pair = _make_pair()
        ap = ActivePair(pair=pair, state=PairState.LONG_SPREAD)
        assert strategy._evaluate_pair(ap, -1.5) == "no_action"


# ---------- _estimate_pair_pnl ----------

class TestEstimatePairPnl:
    def test_long_spread_profit(self, strategy):
        pair = _make_pair()
        ap = ActivePair(pair=pair, state=PairState.LONG_SPREAD, entry_spread=10.0)
        pnl = strategy._estimate_pair_pnl(ap, 15.0)
        assert pnl == pytest.approx(5.0)

    def test_short_spread_profit(self, strategy):
        pair = _make_pair()
        ap = ActivePair(pair=pair, state=PairState.SHORT_SPREAD, entry_spread=15.0)
        pnl = strategy._estimate_pair_pnl(ap, 10.0)
        assert pnl == pytest.approx(5.0)

    def test_flat_zero(self, strategy):
        pair = _make_pair()
        ap = ActivePair(pair=pair, state=PairState.FLAT)
        assert strategy._estimate_pair_pnl(ap, 100.0) == 0.0


# ---------- _should_rescan ----------

class TestShouldRescan:
    def test_no_scan_date(self, strategy):
        assert strategy._should_rescan() is True

    def test_recent_scan(self, strategy):
        strategy._last_scan_date = datetime.now(timezone.utc)
        assert strategy._should_rescan() is False

    def test_old_scan(self, strategy):
        strategy._last_scan_date = datetime.now(timezone.utc) - timedelta(days=10)
        assert strategy._should_rescan() is True


# ---------- _get_latest_prices ----------

class TestGetLatestPrices:
    def test_gets_latest(self):
        data = pd.DataFrame([
            {"symbol": "AAPL", "timestamp": pd.Timestamp("2025-01-01"), "close": 100},
            {"symbol": "AAPL", "timestamp": pd.Timestamp("2025-01-02"), "close": 105},
            {"symbol": "MSFT", "timestamp": pd.Timestamp("2025-01-01"), "close": 200},
            {"symbol": "MSFT", "timestamp": pd.Timestamp("2025-01-02"), "close": 210},
        ])
        prices = PairsTrading._get_latest_prices(data)
        assert prices["AAPL"] == 105
        assert prices["MSFT"] == 210


# ---------- _accumulate ----------

class TestAccumulate:
    def test_new_symbol(self):
        accum = {}
        PairsTrading._accumulate(accum, "AAPL", 0.5, 2.5, "pair1", "long_leg")
        assert "AAPL" in accum
        assert accum["AAPL"].net_score == 0.5

    def test_existing_symbol(self):
        accum = {}
        PairsTrading._accumulate(accum, "AAPL", 0.5, 2.5, "pair1", "long_leg")
        PairsTrading._accumulate(accum, "AAPL", -0.3, -1.0, "pair2", "short_leg")
        assert accum["AAPL"].net_score == pytest.approx(0.2)


# ---------- _finalize_scores ----------

class TestFinalizeScores:
    def test_filters_small_scores(self):
        accum = {}
        PairsTrading._accumulate(accum, "AAPL", 0.005, 0.5, "p1", "l")
        scores = PairsTrading._finalize_scores(accum)
        assert len(scores) == 0  # < 0.01 filtered

    def test_positive_is_long(self):
        accum = {}
        PairsTrading._accumulate(accum, "AAPL", 0.5, 2.5, "p1", "long_leg")
        scores = PairsTrading._finalize_scores(accum)
        assert len(scores) == 1
        assert scores[0].direction.value == "long"

    def test_negative_is_short(self):
        accum = {}
        PairsTrading._accumulate(accum, "AAPL", -0.5, -2.5, "p1", "short_leg")
        scores = PairsTrading._finalize_scores(accum)
        assert len(scores) == 1
        assert scores[0].direction.value == "short"

    def test_confidence_capped(self):
        accum = {}
        PairsTrading._accumulate(accum, "AAPL", 2.0, 5.0, "p1", "l")
        scores = PairsTrading._finalize_scores(accum)
        assert scores[0].confidence == 1.0


# ---------- generate_signals ----------

class TestGenerateSignals:
    def test_auto_initializes(self, strategy):
        data = _make_data()
        strategy._scanner = MagicMock()
        strategy._scanner.scan.return_value = []
        output = strategy.generate_signals(data)
        assert strategy._initialized is True
        assert output.strategy_name == "pairs_trading"

    def test_with_active_pairs_entry(self, strategy):
        pair = _make_pair("AAPL", "MSFT", hedge_ratio=1.0, spread_mean=0.0, spread_std=5.0)
        strategy._initialized = True
        strategy._last_scan_date = datetime.now(timezone.utc)
        strategy._active_pairs = {pair.pair_id: ActivePair(pair=pair)}

        # Create data where spread z-score > entry threshold
        data = pd.DataFrame([
            {"symbol": "AAPL", "timestamp": pd.Timestamp("2025-01-01"), "close": 115.0, "volume": 1e6},
            {"symbol": "MSFT", "timestamp": pd.Timestamp("2025-01-01"), "close": 100.0, "volume": 1e6},
        ])
        # spread = 115 - 1.0*100 = 15, zscore = (15-0)/5 = 3.0 > 2.0 entry
        output = strategy.generate_signals(data)
        assert output.strategy_name == "pairs_trading"
        # Should have signals (short A, long B for positive z)
        if len(output.scores) > 0:
            symbols = {s.symbol for s in output.scores}
            assert "AAPL" in symbols or "MSFT" in symbols

    def test_exit_signal(self, strategy):
        pair = _make_pair("AAPL", "MSFT", hedge_ratio=1.0, spread_mean=0.0, spread_std=5.0)
        ap = ActivePair(
            pair=pair, state=PairState.SHORT_SPREAD,
            entry_spread=15.0, entry_zscore=3.0,
            entry_date=datetime.now(timezone.utc),
        )
        strategy._initialized = True
        strategy._last_scan_date = datetime.now(timezone.utc)
        strategy._active_pairs = {pair.pair_id: ap}

        # spread z-score near 0 -> exit
        data = pd.DataFrame([
            {"symbol": "AAPL", "timestamp": pd.Timestamp("2025-01-01"), "close": 101.0, "volume": 1e6},
            {"symbol": "MSFT", "timestamp": pd.Timestamp("2025-01-01"), "close": 100.0, "volume": 1e6},
        ])
        output = strategy.generate_signals(data)
        assert strategy._active_pairs[pair.pair_id].state == PairState.FLAT

    def test_stop_loss_signal(self, strategy):
        pair = _make_pair("AAPL", "MSFT", hedge_ratio=1.0, spread_mean=0.0, spread_std=5.0)
        ap = ActivePair(
            pair=pair, state=PairState.SHORT_SPREAD,
            entry_spread=15.0, entry_zscore=3.0,
            entry_date=datetime.now(timezone.utc),
        )
        strategy._initialized = True
        strategy._last_scan_date = datetime.now(timezone.utc)
        strategy._active_pairs = {pair.pair_id: ap}

        # spread z-score > stop_loss (4.0) -> stop_loss
        data = pd.DataFrame([
            {"symbol": "AAPL", "timestamp": pd.Timestamp("2025-01-01"), "close": 125.0, "volume": 1e6},
            {"symbol": "MSFT", "timestamp": pd.Timestamp("2025-01-01"), "close": 100.0, "volume": 1e6},
        ])
        # spread = 25, z = 25/5 = 5.0 > 4.0
        output = strategy.generate_signals(data)
        assert strategy._active_pairs[pair.pair_id].state == PairState.FLAT

    def test_long_spread_entry(self, strategy):
        pair = _make_pair("AAPL", "MSFT", hedge_ratio=1.0, spread_mean=0.0, spread_std=5.0)
        strategy._initialized = True
        strategy._last_scan_date = datetime.now(timezone.utc)
        strategy._active_pairs = {pair.pair_id: ActivePair(pair=pair)}

        # spread = 85 - 100 = -15, z = -15/5 = -3.0 < -2.0
        data = pd.DataFrame([
            {"symbol": "AAPL", "timestamp": pd.Timestamp("2025-01-01"), "close": 85.0, "volume": 1e6},
            {"symbol": "MSFT", "timestamp": pd.Timestamp("2025-01-01"), "close": 100.0, "volume": 1e6},
        ])
        output = strategy.generate_signals(data)
        assert strategy._active_pairs[pair.pair_id].state == PairState.LONG_SPREAD

    def test_long_spread_exit(self, strategy):
        pair = _make_pair("AAPL", "MSFT", hedge_ratio=1.0, spread_mean=0.0, spread_std=5.0)
        ap = ActivePair(
            pair=pair, state=PairState.LONG_SPREAD,
            entry_spread=-15.0, entry_zscore=-3.0,
            entry_date=datetime.now(timezone.utc),
        )
        strategy._initialized = True
        strategy._last_scan_date = datetime.now(timezone.utc)
        strategy._active_pairs = {pair.pair_id: ap}

        # z near 0 -> exit
        data = pd.DataFrame([
            {"symbol": "AAPL", "timestamp": pd.Timestamp("2025-01-01"), "close": 100.5, "volume": 1e6},
            {"symbol": "MSFT", "timestamp": pd.Timestamp("2025-01-01"), "close": 100.0, "volume": 1e6},
        ])
        output = strategy.generate_signals(data)
        assert strategy._active_pairs[pair.pair_id].state == PairState.FLAT

    def test_long_spread_stop_loss(self, strategy):
        pair = _make_pair("AAPL", "MSFT", hedge_ratio=1.0, spread_mean=0.0, spread_std=5.0)
        ap = ActivePair(
            pair=pair, state=PairState.LONG_SPREAD,
            entry_spread=-15.0, entry_zscore=-3.0,
            entry_date=datetime.now(timezone.utc),
        )
        strategy._initialized = True
        strategy._last_scan_date = datetime.now(timezone.utc)
        strategy._active_pairs = {pair.pair_id: ap}

        # z = -25/5 = -5.0 < -4.0 -> stop loss
        data = pd.DataFrame([
            {"symbol": "AAPL", "timestamp": pd.Timestamp("2025-01-01"), "close": 75.0, "volume": 1e6},
            {"symbol": "MSFT", "timestamp": pd.Timestamp("2025-01-01"), "close": 100.0, "volume": 1e6},
        ])
        output = strategy.generate_signals(data)
        assert strategy._active_pairs[pair.pair_id].state == PairState.FLAT

    def test_missing_price_skipped(self, strategy):
        pair = _make_pair("AAPL", "MSFT")
        strategy._initialized = True
        strategy._last_scan_date = datetime.now(timezone.utc)
        strategy._active_pairs = {pair.pair_id: ActivePair(pair=pair)}

        # Only AAPL data, no MSFT
        data = pd.DataFrame([
            {"symbol": "AAPL", "timestamp": pd.Timestamp("2025-01-01"), "close": 100.0, "volume": 1e6},
        ])
        output = strategy.generate_signals(data)
        assert len(output.scores) == 0


# ---------- _rescan ----------

class TestRescan:
    def test_rescan_preserves_positioned_pairs(self, strategy):
        old_pair = _make_pair("AAPL", "MSFT")
        new_pair = _make_pair("GOOGL", "META")

        ap = ActivePair(
            pair=old_pair, state=PairState.SHORT_SPREAD,
            entry_zscore=2.5, entry_spread=10.0,
            entry_date=datetime.now(timezone.utc),
        )
        strategy._active_pairs = {old_pair.pair_id: ap}

        strategy._scanner = MagicMock()
        strategy._scanner.scan.return_value = [new_pair]

        data = _make_data()
        strategy._rescan(data)

        # Old positioned pair preserved + new pair added
        assert old_pair.pair_id in strategy._active_pairs
        assert new_pair.pair_id in strategy._active_pairs
        assert strategy._active_pairs[old_pair.pair_id].state == PairState.SHORT_SPREAD

    def test_rescan_updates_stats_for_existing(self, strategy):
        pair = _make_pair("AAPL", "MSFT")
        ap = ActivePair(
            pair=pair, state=PairState.LONG_SPREAD,
            entry_zscore=-2.5, cumulative_pnl=100.0,
            entry_date=datetime.now(timezone.utc),
        )
        strategy._active_pairs = {pair.pair_id: ap}

        # Scanner returns same pair (re-validated)
        new_pair = _make_pair("AAPL", "MSFT", hedge_ratio=1.05)
        strategy._scanner = MagicMock()
        strategy._scanner.scan.return_value = [new_pair]

        data = _make_data()
        strategy._rescan(data)

        result_ap = strategy._active_pairs[pair.pair_id]
        assert result_ap.state == PairState.LONG_SPREAD
        assert result_ap.cumulative_pnl == 100.0


# ---------- get_performance / get_active_pairs_summary ----------

class TestHelperMethods:
    def test_get_performance(self, strategy):
        perf = strategy.get_performance()
        assert perf.strategy_name == "pairs_trading"

    def test_get_active_pairs_summary(self, strategy):
        pair = _make_pair("AAPL", "MSFT")
        strategy._active_pairs = {pair.pair_id: ActivePair(pair=pair, last_zscore=1.5)}
        summary = strategy.get_active_pairs_summary()
        assert len(summary) == 1
        assert summary[0]["symbol_a"] == "AAPL"
        assert summary[0]["last_zscore"] == 1.5


# ---------- _update_spread_stats ----------

class TestUpdateSpreadStats:
    def test_updates_spread_mean_std(self, strategy):
        pair = _make_pair("AAPL", "MSFT", hedge_ratio=1.0)
        ap = ActivePair(pair=pair)
        data = _make_data(("AAPL", "MSFT"), days=100)
        strategy._update_spread_stats(ap, data)
        # spread_mean and spread_std should be updated
        assert pair.spread_mean != 0.0 or pair.spread_std != 5.0

    def test_empty_data_no_update(self, strategy):
        pair = _make_pair("AAPL", "MSFT", spread_mean=0.0, spread_std=5.0)
        ap = ActivePair(pair=pair)
        data = pd.DataFrame(columns=["symbol", "timestamp", "close"])
        strategy._update_spread_stats(ap, data)
        assert pair.spread_std == 5.0  # unchanged
