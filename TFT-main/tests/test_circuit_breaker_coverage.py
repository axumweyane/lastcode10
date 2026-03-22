"""
Coverage tests for trading/risk/circuit_breaker.py.
All external dependencies (Redis, broker, notifier, audit) are mocked.
"""

import asyncio
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.broker.base import AccountInfo, OrderResult, OrderStatus, PositionInfo
from trading.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    DrawdownConfig,
    DrawdownMethod,
    _KEY_HWM,
    _KEY_IS_TRIPPED,
    _KEY_SOD,
    _KEY_STATE,
)

# ---------- Fixtures ----------


def _make_account(portfolio_value: float = 100000.0) -> AccountInfo:
    return AccountInfo(
        account_id="test",
        status="ACTIVE",
        currency="USD",
        cash=50000.0,
        portfolio_value=portfolio_value,
        buying_power=100000.0,
        equity=portfolio_value,
        last_equity=100000.0,
        long_market_value=50000.0,
        short_market_value=0.0,
    )


def _make_position(ticker="AAPL", qty=10, side="long", mv=1500.0, pnl=50.0):
    return PositionInfo(
        ticker=ticker,
        quantity=qty,
        side=side,
        market_value=mv,
        cost_basis=mv - pnl,
        unrealized_pnl=pnl,
        unrealized_pnl_percent=pnl / mv,
        current_price=mv / qty,
        avg_entry_price=(mv - pnl) / qty,
    )


@pytest.fixture
def config():
    return CircuitBreakerConfig(
        enabled=True,
        drawdown_configs=[
            DrawdownConfig(DrawdownMethod.HIGH_WATER_MARK, 5.0),
            DrawdownConfig(DrawdownMethod.START_OF_DAY, 3.0),
        ],
        check_interval_seconds=1,
        initial_capital=100000.0,
    )


@pytest.fixture
def disabled_config():
    return CircuitBreakerConfig(enabled=False)


@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.get = AsyncMock(return_value=None)
    r.set = AsyncMock()
    return r


@pytest.fixture
def mock_broker():
    b = AsyncMock()
    b.get_account = AsyncMock(return_value=_make_account())
    b.get_positions = AsyncMock(return_value=[])
    b.close_all_positions = AsyncMock(return_value=[])
    return b


@pytest.fixture
def mock_notifier():
    n = AsyncMock()
    n.notify_circuit_breaker_trip = AsyncMock()
    n.notify_circuit_breaker_reset = AsyncMock()
    return n


@pytest.fixture
def mock_audit():
    a = MagicMock()
    a.get_latest_trip_event.return_value = None
    a.get_latest_snapshot.return_value = None
    a.get_recent_events.return_value = []
    a.log_trip_event.return_value = 1
    a.log_closure.return_value = None
    a.log_portfolio_snapshot.return_value = None
    a.log_reset_event.return_value = 1
    return a


@pytest.fixture
def cb(config, mock_redis, mock_broker, mock_notifier, mock_audit):
    return CircuitBreaker(config, mock_broker, mock_redis, mock_notifier, mock_audit)


# ---------- CircuitBreakerState ----------


class TestCircuitBreakerState:
    def test_to_json_and_from_json(self):
        state = CircuitBreakerState(
            is_tripped=True,
            tripped_at="2026-03-21T10:00:00",
            trip_reason="test",
            hwm=105000.0,
            sod_value=100000.0,
            last_portfolio_value=95000.0,
        )
        raw = state.to_json()
        restored = CircuitBreakerState.from_json(raw)
        assert restored.is_tripped is True
        assert restored.hwm == 105000.0
        assert restored.trip_reason == "test"
        assert restored.sod_value == 100000.0

    def test_default_state(self):
        state = CircuitBreakerState()
        assert state.is_tripped is False
        assert state.hwm == 0.0


# ---------- CircuitBreakerConfig ----------


class TestCircuitBreakerConfig:
    def test_from_env_defaults(self):
        with patch.dict("os.environ", {}, clear=True):
            cfg = CircuitBreakerConfig.from_env()
        assert cfg.enabled is True
        assert len(cfg.drawdown_configs) == 2
        assert cfg.initial_capital == 100000.0

    def test_from_env_custom(self):
        env = {
            "CIRCUIT_BREAKER_ENABLED": "false",
            "CB_CHECK_INTERVAL_SECONDS": "60",
            "CB_INITIAL_CAPITAL": "200000",
            "CB_DRAWDOWN_METHODS": "initial_capital:10.0",
        }
        with patch.dict("os.environ", env, clear=True):
            cfg = CircuitBreakerConfig.from_env()
        assert cfg.enabled is False
        assert cfg.check_interval_seconds == 60
        assert cfg.initial_capital == 200000.0
        assert len(cfg.drawdown_configs) == 1
        assert cfg.drawdown_configs[0].method == DrawdownMethod.INITIAL_CAPITAL

    def test_from_env_invalid_method_skipped(self):
        env = {"CB_DRAWDOWN_METHODS": "bad_method:5.0,high_water_mark:5.0"}
        with patch.dict("os.environ", env, clear=True):
            cfg = CircuitBreakerConfig.from_env()
        assert len(cfg.drawdown_configs) == 1

    def test_from_env_empty_methods_uses_defaults(self):
        env = {"CB_DRAWDOWN_METHODS": "bad:bad"}
        with patch.dict("os.environ", env, clear=True):
            cfg = CircuitBreakerConfig.from_env()
        assert len(cfg.drawdown_configs) == 2  # defaults


# ---------- _load_state ----------


class TestLoadState:
    @pytest.mark.asyncio
    async def test_load_from_redis(self, cb, mock_redis):
        state = CircuitBreakerState(hwm=110000.0, is_tripped=False)
        mock_redis.get.return_value = state.to_json()
        await cb._load_state()
        assert cb.state.hwm == 110000.0

    @pytest.mark.asyncio
    async def test_load_from_postgres_trip(self, cb, mock_redis, mock_audit):
        mock_redis.get.return_value = None
        mock_audit.get_latest_trip_event.return_value = {"event_type": "trip"}
        mock_audit.get_recent_events.return_value = [
            {
                "event_type": "trip",
                "created_at": "2026-03-21T10:00:00",
                "reason": "drawdown",
            }
        ]
        mock_audit.get_latest_snapshot.return_value = {
            "high_water_mark": 120000,
            "portfolio_value": 95000,
        }
        await cb._load_state()
        assert cb.state.is_tripped is True
        assert cb.state.hwm == 120000

    @pytest.mark.asyncio
    async def test_load_no_data_uses_initial_capital(self, cb, mock_redis):
        mock_redis.get.return_value = None
        await cb._load_state()
        assert cb.state.hwm == 100000.0  # initial_capital


# ---------- _calculate_drawdown ----------


class TestCalculateDrawdown:
    def test_hwm_drawdown(self, cb):
        cb.state.hwm = 100000
        dd = cb._calculate_drawdown(DrawdownMethod.HIGH_WATER_MARK, 95000)
        assert dd == pytest.approx(5.0)

    def test_hwm_zero_returns_none(self, cb):
        cb.state.hwm = 0
        assert cb._calculate_drawdown(DrawdownMethod.HIGH_WATER_MARK, 95000) is None

    def test_sod_drawdown(self, cb):
        cb.state.sod_value = 100000
        dd = cb._calculate_drawdown(DrawdownMethod.START_OF_DAY, 97000)
        assert dd == pytest.approx(3.0)

    def test_sod_zero_returns_none(self, cb):
        cb.state.sod_value = 0
        assert cb._calculate_drawdown(DrawdownMethod.START_OF_DAY, 97000) is None

    def test_initial_capital_drawdown(self, cb):
        dd = cb._calculate_drawdown(DrawdownMethod.INITIAL_CAPITAL, 90000)
        assert dd == pytest.approx(10.0)

    def test_initial_capital_zero_returns_none(self, cb):
        cb.config.initial_capital = 0
        assert cb._calculate_drawdown(DrawdownMethod.INITIAL_CAPITAL, 90000) is None


# ---------- check ----------


class TestCheck:
    @pytest.mark.asyncio
    async def test_already_tripped(self, cb):
        cb.state.is_tripped = True
        assert await cb.check() is True

    @pytest.mark.asyncio
    async def test_no_drawdown_not_tripped(self, cb, mock_broker):
        cb.state.hwm = 100000
        cb.state.sod_value = 100000
        mock_broker.get_account.return_value = _make_account(100000)
        assert await cb.check() is False

    @pytest.mark.asyncio
    async def test_hwm_drawdown_trips(self, cb, mock_broker, mock_audit, mock_notifier):
        cb.state.hwm = 100000
        mock_broker.get_account.return_value = _make_account(94000)  # 6% DD > 5%
        result = await cb.check()
        assert result is True
        assert cb.state.is_tripped is True
        mock_audit.log_trip_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_sod_drawdown_trips(self, cb, mock_broker, mock_audit):
        cb.state.hwm = 200000  # well above, won't trigger
        cb.state.sod_value = 100000
        mock_broker.get_account.return_value = _make_account(96000)  # 4% DD > 3%
        result = await cb.check()
        assert result is True

    @pytest.mark.asyncio
    async def test_updates_hwm_on_new_high(self, cb, mock_broker, mock_redis):
        cb.state.hwm = 100000
        cb.state.sod_value = 0
        mock_broker.get_account.return_value = _make_account(110000)
        await cb.check()
        assert cb.state.hwm == 110000
        mock_redis.set.assert_any_call(_KEY_HWM, "110000")

    @pytest.mark.asyncio
    async def test_api_failure_increments_counter(self, cb, mock_broker):
        mock_broker.get_account.side_effect = ConnectionError("timeout")
        await cb.check()
        assert cb._consecutive_api_failures == 1

    @pytest.mark.asyncio
    async def test_consecutive_api_failures_trip(self, cb, mock_broker, mock_audit):
        mock_broker.get_account.side_effect = ConnectionError("timeout")
        for _ in range(3):
            await cb.check()
        assert cb.state.is_tripped is True
        mock_audit.log_trip_event.assert_called_once()


# ---------- _trip ----------


class TestTrip:
    @pytest.mark.asyncio
    async def test_trip_closes_positions(
        self, cb, mock_broker, mock_audit, mock_notifier
    ):
        positions = [_make_position("AAPL"), _make_position("GOOGL")]
        mock_broker.get_positions.return_value = positions
        close_results = [
            OrderResult(success=True, order_id="o1", status=OrderStatus.FILLED),
            OrderResult(success=True, order_id="o2", status=OrderStatus.FILLED),
        ]
        mock_broker.close_all_positions.return_value = close_results

        await cb._trip("test reason", 95000, 5.0, "high_water_mark")

        assert cb.state.is_tripped is True
        mock_audit.log_trip_event.assert_called_once()
        assert mock_audit.log_closure.call_count == 2
        mock_audit.log_portfolio_snapshot.assert_called_once()
        mock_notifier.notify_circuit_breaker_trip.assert_called_once()

    @pytest.mark.asyncio
    async def test_trip_notification_failure_doesnt_crash(
        self, cb, mock_broker, mock_audit, mock_notifier
    ):
        mock_notifier.notify_circuit_breaker_trip.side_effect = Exception(
            "Discord down"
        )
        await cb._trip("test", 95000, 5.0, "hwm")
        assert cb.state.is_tripped is True

    @pytest.mark.asyncio
    async def test_trip_close_all_failure(self, cb, mock_broker, mock_audit):
        mock_broker.close_all_positions.side_effect = Exception("API error")
        await cb._trip("test", 95000, 5.0, "hwm")
        assert cb.state.is_tripped is True
        mock_audit.log_trip_event.assert_called_once()


# ---------- is_tripped ----------


class TestIsTripped:
    @pytest.mark.asyncio
    async def test_tripped_string(self, cb, mock_redis):
        mock_redis.get.return_value = "true"
        assert await cb.is_tripped() is True

    @pytest.mark.asyncio
    async def test_tripped_bytes(self, cb, mock_redis):
        mock_redis.get.return_value = b"true"
        assert await cb.is_tripped() is True

    @pytest.mark.asyncio
    async def test_not_tripped(self, cb, mock_redis):
        mock_redis.get.return_value = "false"
        assert await cb.is_tripped() is False


# ---------- set_start_of_day_value ----------


class TestSetSODValue:
    @pytest.mark.asyncio
    async def test_sets_sod(self, cb, mock_redis, mock_audit):
        await cb.set_start_of_day_value(105000.0)
        assert cb.state.sod_value == 105000.0
        mock_redis.set.assert_any_call(_KEY_SOD, "105000.0", ex=86400)
        mock_audit.log_portfolio_snapshot.assert_called_once_with(
            105000.0, cb.state.hwm, "sod"
        )


# ---------- update_high_water_mark ----------


class TestUpdateHWM:
    @pytest.mark.asyncio
    async def test_updates_when_higher(self, cb, mock_redis):
        cb.state.hwm = 100000
        await cb.update_high_water_mark(110000)
        assert cb.state.hwm == 110000

    @pytest.mark.asyncio
    async def test_no_update_when_lower(self, cb, mock_redis):
        cb.state.hwm = 100000
        await cb.update_high_water_mark(90000)
        assert cb.state.hwm == 100000


# ---------- reset_breaker ----------


class TestResetBreaker:
    @pytest.mark.asyncio
    async def test_reset(self, cb, mock_broker, mock_redis, mock_audit, mock_notifier):
        cb.state.is_tripped = True
        cb.state.trip_reason = "some reason"
        await cb.reset_breaker("admin", "manual reset")
        assert cb.state.is_tripped is False
        assert cb.state.trip_reason is None
        mock_audit.log_reset_event.assert_called_once()
        mock_notifier.notify_circuit_breaker_reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_broker_failure_uses_last_value(
        self, cb, mock_broker, mock_audit, mock_notifier
    ):
        cb.state.is_tripped = True
        cb.state.last_portfolio_value = 99000
        mock_broker.get_account.side_effect = Exception("offline")
        await cb.reset_breaker("admin", "test")
        assert cb.state.is_tripped is False
        mock_audit.log_reset_event.assert_called_once_with("admin", "test", 99000)

    @pytest.mark.asyncio
    async def test_reset_notification_failure(
        self, cb, mock_broker, mock_audit, mock_notifier
    ):
        cb.state.is_tripped = True
        mock_notifier.notify_circuit_breaker_reset.side_effect = Exception("fail")
        await cb.reset_breaker("admin", "test")
        assert cb.state.is_tripped is False


# ---------- start / stop ----------


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_disabled(
        self, disabled_config, mock_redis, mock_broker, mock_notifier, mock_audit
    ):
        cb = CircuitBreaker(
            disabled_config, mock_broker, mock_redis, mock_notifier, mock_audit
        )
        await cb.start()
        assert cb._monitor_task is None

    @pytest.mark.asyncio
    async def test_start_enabled(self, cb, mock_redis):
        mock_redis.get.return_value = None
        await cb.start()
        assert cb._monitor_task is not None
        await cb.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, cb, mock_redis):
        mock_redis.get.return_value = None
        await cb.start()
        task = cb._monitor_task
        await cb.stop()
        assert task.cancelled() or task.done()
