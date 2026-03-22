"""Tests for the VWAP execution model."""

import asyncio
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.broker.base import (
    OrderInfo,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from trading.execution.vwap import (
    VWAPExecutionModel,
    VWAPExecutionResult,
    VolumeProfileCache,
    SliceResult,
    DEFAULT_NUM_SLICES,
)

# ── Fake broker for testing ──────────────────────────────────────────────────


class FakeBroker:
    """Simulates AlpacaBroker for unit tests."""

    def __init__(self, fill_rate: float = 1.0, fill_price: float = 100.0):
        self.fill_rate = fill_rate  # fraction of each slice filled
        self.fill_price = fill_price
        self.submitted_orders: list = []
        self._order_counter = 0
        self._orders: dict = {}

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def submit_order(self, request: OrderRequest) -> OrderResult:
        self._order_counter += 1
        oid = f"fake-{self._order_counter}"
        filled = int(request.quantity * self.fill_rate)
        self._orders[oid] = OrderInfo(
            order_id=oid,
            ticker=request.ticker,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            filled_quantity=filled,
            status=OrderStatus.FILLED if filled > 0 else OrderStatus.CANCELLED,
            time_in_force=request.time_in_force,
            limit_price=request.limit_price,
            filled_avg_price=self.fill_price if filled > 0 else None,
        )
        self.submitted_orders.append(request)
        return OrderResult(success=True, order_id=oid, status=OrderStatus.FILLED)

    async def get_order(self, order_id: str):
        return self._orders.get(order_id)


class FailingBroker(FakeBroker):
    """Broker that always fails submit_order."""

    async def submit_order(self, request: OrderRequest) -> OrderResult:
        self.submitted_orders.append(request)
        return OrderResult(success=False, message="simulated failure")


# ── Helper ────────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── 1. Slice count ──────────────────────────────────────────────────────────


class TestSliceCount:
    """Verify correct number of slices are generated."""

    def test_default_five_slices(self):
        broker = FakeBroker()
        vwap = VWAPExecutionModel(broker, num_slices=5, slice_interval_s=0)
        result = _run(vwap.execute("AAPL", OrderSide.BUY, 100, 150.0))
        # Should have exactly 5 slices (no sweep needed since 100% fill)
        assert len(result.slices) == 5

    def test_single_slice(self):
        broker = FakeBroker()
        vwap = VWAPExecutionModel(broker, num_slices=1, slice_interval_s=0)
        result = _run(vwap.execute("AAPL", OrderSide.BUY, 50, 150.0))
        assert len(result.slices) == 1

    def test_more_slices_than_quantity(self):
        broker = FakeBroker()
        vwap = VWAPExecutionModel(broker, num_slices=10, slice_interval_s=0)
        result = _run(vwap.execute("AAPL", OrderSide.BUY, 3, 150.0))
        # Can't have more slices than shares
        assert result.total_filled == 3

    def test_zero_quantity_no_slices(self):
        broker = FakeBroker()
        vwap = VWAPExecutionModel(broker, num_slices=5, slice_interval_s=0)
        result = _run(vwap.execute("AAPL", OrderSide.BUY, 0, 150.0))
        assert len(result.slices) == 0
        assert result.total_filled == 0


# ── 2. Volume cap (10% ADV) ────────────────────────────────────────────────


class TestADVCap:
    """Verify ADV cap limits order size."""

    def test_adv_cap_reduces_quantity(self):
        broker = FakeBroker()
        vwap = VWAPExecutionModel(
            broker, num_slices=5, slice_interval_s=0, adv_cap_pct=0.10
        )
        # ADV = 100 shares, 10% cap = 10 shares max
        result = _run(vwap.execute("AAPL", OrderSide.BUY, 50, 150.0, adv=100))
        assert result.adv_capped is True
        assert result.total_requested == 10
        assert result.total_filled <= 10

    def test_below_adv_cap_no_reduction(self):
        broker = FakeBroker()
        vwap = VWAPExecutionModel(
            broker, num_slices=5, slice_interval_s=0, adv_cap_pct=0.10
        )
        # ADV = 10000, cap = 1000, requesting 50 → no cap
        result = _run(vwap.execute("AAPL", OrderSide.BUY, 50, 150.0, adv=10000))
        assert result.adv_capped is False
        assert result.total_requested == 50

    def test_zero_adv_skips_cap(self):
        broker = FakeBroker()
        vwap = VWAPExecutionModel(
            broker, num_slices=5, slice_interval_s=0, adv_cap_pct=0.10
        )
        result = _run(vwap.execute("AAPL", OrderSide.BUY, 200, 150.0, adv=0))
        assert result.adv_capped is False
        assert result.total_requested == 200


# ── 3. Carry-forward unfilled ─────────────────────────────────────────────


class TestCarryForward:
    """Verify unfilled shares carry forward to next slices and final sweep."""

    def test_partial_fill_carries_forward(self):
        broker = FakeBroker(fill_rate=0.5, fill_price=100.0)
        vwap = VWAPExecutionModel(broker, num_slices=3, slice_interval_s=0)
        result = _run(vwap.execute("AAPL", OrderSide.BUY, 100, 100.0))
        # With 50% fill rate, after 3 slices + sweep, we should see more than 3 slices
        assert len(result.slices) >= 3
        # Sweep slice should be present (since not everything fills)
        has_sweep = any(s.status == "sweep" for s in result.slices)
        assert has_sweep

    def test_full_fill_no_sweep(self):
        broker = FakeBroker(fill_rate=1.0, fill_price=100.0)
        vwap = VWAPExecutionModel(broker, num_slices=5, slice_interval_s=0)
        result = _run(vwap.execute("AAPL", OrderSide.BUY, 100, 100.0))
        assert result.total_filled == 100
        # No sweep needed
        has_sweep = any(s.status == "sweep" for s in result.slices)
        assert not has_sweep

    def test_zero_fill_triggers_sweep(self):
        broker = FakeBroker(fill_rate=0.0, fill_price=100.0)

        # Override get_order to return 0 filled
        async def get_order_zero(oid):
            return OrderInfo(
                order_id=oid,
                ticker="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=100,
                filled_quantity=0,
                status=OrderStatus.CANCELLED,
                time_in_force=TimeInForce.IOC,
            )

        broker.get_order = get_order_zero
        vwap = VWAPExecutionModel(broker, num_slices=3, slice_interval_s=0)
        result = _run(vwap.execute("AAPL", OrderSide.BUY, 100, 100.0))
        # All IOC slices fail → sweep fires
        has_sweep = any(s.status == "sweep" for s in result.slices)
        assert has_sweep


# ── 4. Fallback to market order ───────────────────────────────────────────


class TestFallback:
    """Verify fallback to market order on execution errors."""

    def test_exception_triggers_fallback(self):
        broker = FakeBroker()

        # Make the broker raise on first call, then work for fallback
        call_count = [0]
        original_submit = broker.submit_order

        async def broken_submit(req):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated network error")
            return await original_submit(req)

        broker.submit_order = broken_submit

        vwap = VWAPExecutionModel(broker, num_slices=3, slice_interval_s=0)
        result = _run(vwap.execute("AAPL", OrderSide.BUY, 50, 100.0))
        assert result.used_fallback is True
        assert len(result.slices) >= 1
        assert result.slices[0].status == "fallback_market"

    def test_no_fallback_on_success(self):
        broker = FakeBroker()
        vwap = VWAPExecutionModel(broker, num_slices=5, slice_interval_s=0)
        result = _run(vwap.execute("AAPL", OrderSide.BUY, 50, 100.0))
        assert result.used_fallback is False


# ── 5. Slippage calculation ──────────────────────────────────────────────


class TestSlippage:
    """Verify slippage tracking."""

    def test_buy_positive_slippage_when_fill_above_expected(self):
        broker = FakeBroker(fill_rate=1.0, fill_price=101.0)
        vwap = VWAPExecutionModel(broker, num_slices=1, slice_interval_s=0)
        result = _run(vwap.execute("AAPL", OrderSide.BUY, 100, 100.0))
        # Bought at 101 vs expected 100 → positive slippage (bad for buyer)
        assert result.slippage_bps > 0
        expected_bps = (101.0 - 100.0) / 100.0 * 10_000  # 100 bps
        assert abs(result.slippage_bps - expected_bps) < 1

    def test_sell_positive_slippage_when_fill_below_expected(self):
        broker = FakeBroker(fill_rate=1.0, fill_price=99.0)
        vwap = VWAPExecutionModel(broker, num_slices=1, slice_interval_s=0)
        result = _run(vwap.execute("AAPL", OrderSide.SELL, 100, 100.0))
        # Sold at 99 vs expected 100 → positive slippage (bad for seller)
        assert result.slippage_bps > 0
        expected_bps = (100.0 - 99.0) / 100.0 * 10_000  # 100 bps
        assert abs(result.slippage_bps - expected_bps) < 1

    def test_zero_slippage_when_fill_at_expected(self):
        broker = FakeBroker(fill_rate=1.0, fill_price=100.0)
        vwap = VWAPExecutionModel(broker, num_slices=1, slice_interval_s=0)
        result = _run(vwap.execute("AAPL", OrderSide.BUY, 100, 100.0))
        assert abs(result.slippage_bps) < 1


# ── 6. IOC order type ────────────────────────────────────────────────────


class TestIOCOrderType:
    """Verify slices use IOC time-in-force and limit order type."""

    def test_slices_are_ioc_limit(self):
        broker = FakeBroker()
        vwap = VWAPExecutionModel(broker, num_slices=3, slice_interval_s=0)
        _run(vwap.execute("AAPL", OrderSide.BUY, 100, 100.0))
        for req in broker.submitted_orders:
            assert req.order_type == OrderType.LIMIT
            assert req.time_in_force == TimeInForce.IOC
            assert req.limit_price is not None

    def test_limit_price_offset_buy(self):
        broker = FakeBroker()
        vwap = VWAPExecutionModel(
            broker,
            num_slices=1,
            slice_interval_s=0,
            limit_offset_bps=10,  # 10 bps above for buy
        )
        _run(vwap.execute("AAPL", OrderSide.BUY, 100, 100.0))
        req = broker.submitted_orders[0]
        # 10 bps of $100 = $0.10 above
        assert req.limit_price == 100.10

    def test_limit_price_offset_sell(self):
        broker = FakeBroker()
        vwap = VWAPExecutionModel(
            broker,
            num_slices=1,
            slice_interval_s=0,
            limit_offset_bps=10,
        )
        _run(vwap.execute("AAPL", OrderSide.SELL, 100, 100.0))
        req = broker.submitted_orders[0]
        # 10 bps below for sell
        assert req.limit_price == 99.90


# ── 7. Volume profile cache ──────────────────────────────────────────────


class TestVolumeProfileCache:
    """Verify volume profile caching and weight generation."""

    def test_default_profile_used_when_empty(self):
        cache = VolumeProfileCache()
        profile = cache.get_or_default("AAPL")
        assert len(profile) == 13  # 13 half-hour buckets
        assert abs(sum(profile) - 1.0) < 0.01

    def test_put_and_get(self):
        cache = VolumeProfileCache()
        custom = [
            0.2,
            0.1,
            0.1,
            0.1,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.1,
        ]
        cache.put("TSLA", custom)
        retrieved = cache.get("TSLA")
        assert retrieved == custom

    def test_cache_expiry(self):
        cache = VolumeProfileCache(ttl_s=0)  # instant expiry
        cache.put("AAPL", [0.1] * 13)
        import time

        time.sleep(0.01)
        assert cache.get("AAPL") is None

    def test_slice_weights_sum_to_one(self):
        cache = VolumeProfileCache()
        for n in [1, 3, 5, 10, 20]:
            weights = cache.get_slice_weights("AAPL", n)
            assert len(weights) == n
            assert abs(sum(weights) - 1.0) < 1e-10

    def test_slice_weights_proportional_to_profile(self):
        cache = VolumeProfileCache()
        # With default U-shaped profile, first and last weights should be larger
        weights = cache.get_slice_weights("AAPL", 5)
        assert weights[0] > weights[2]  # open > midday
        assert weights[4] > weights[2]  # close > midday


# ── 8. Execution result ──────────────────────────────────────────────────


class TestExecutionResult:
    """Verify VWAPExecutionResult dataclass and serialization."""

    def test_fill_rate(self):
        result = VWAPExecutionResult(
            ticker="AAPL",
            side="buy",
            total_requested=100,
            total_filled=75,
            filled_avg_price=100.0,
        )
        assert result.fill_rate == 0.75

    def test_fill_rate_zero_requested(self):
        result = VWAPExecutionResult(
            ticker="AAPL",
            side="buy",
            total_requested=0,
            total_filled=0,
            filled_avg_price=0.0,
        )
        assert result.fill_rate == 0.0

    def test_to_dict_serializable(self):
        result = VWAPExecutionResult(
            ticker="AAPL",
            side="buy",
            total_requested=100,
            total_filled=80,
            filled_avg_price=150.25,
            expected_price=150.0,
            slippage_bps=16.67,
            slices=[
                SliceResult(0, 50, 40, 150.1, "o1", "filled", "2026-01-01T00:00:00Z")
            ],
        )
        d = result.to_dict()
        json.dumps(d)  # must be JSON-serializable
        assert d["ticker"] == "AAPL"
        assert d["fill_rate"] == 0.8
        assert d["num_slices"] == 1

    def test_execution_stats_aggregation(self):
        broker = FakeBroker()
        vwap = VWAPExecutionModel(broker, num_slices=2, slice_interval_s=0)
        _run(vwap.execute("AAPL", OrderSide.BUY, 50, 100.0))
        _run(vwap.execute("MSFT", OrderSide.SELL, 30, 200.0))
        stats = vwap.get_execution_stats()
        assert stats["total_executions"] == 2


# ── 9. Paper-trader structural tests ─────────────────────────────────────


class TestPaperTraderVWAPWiring:
    """Verify VWAP is wired into paper-trader."""

    def _read_source(self):
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper-trader",
            "main.py",
        )
        with open(main_path) as f:
            return f.read()

    def test_vwap_import(self):
        source = self._read_source()
        assert "from trading.execution.vwap import" in source
        assert "VWAPExecutionModel" in source

    def test_vwap_env_var(self):
        source = self._read_source()
        assert "EXECUTION_USE_VWAP" in source

    def test_vwap_global_var(self):
        source = self._read_source()
        assert "vwap_model" in source

    def test_vwap_initialized(self):
        source = self._read_source()
        assert "VWAPExecutionModel(" in source

    def test_log_execution_stats_called(self):
        source = self._read_source()
        assert "log_execution_stats" in source

    def test_execution_stats_endpoint(self):
        source = self._read_source()
        assert '"/execution/stats"' in source

    def test_execution_stats_table_in_schema(self):
        source = self._read_source()
        assert "paper_execution_stats" in source

    def test_execution_stats_table_in_postgres_schema(self):
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "postgres_schema.py",
        )
        with open(schema_path) as f:
            source = f.read()
        assert "paper_execution_stats" in source

    def test_vwap_in_discord_report(self):
        source = self._read_source()
        assert "vwap_line" in source or "VWAP" in source

    def test_env_template_has_vwap_vars(self):
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".env.template",
        )
        with open(template_path) as f:
            source = f.read()
        assert "EXECUTION_USE_VWAP" in source
        assert "VWAP_NUM_SLICES" in source
