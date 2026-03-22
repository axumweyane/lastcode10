"""
Coverage tests for trading/broker/alpaca.py.
All HTTP calls are mocked via aiohttp test utilities.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from trading.broker.alpaca import AlpacaBroker, _parse_timestamp
from trading.broker.base import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)

# ---------- Mock helpers ----------


def _mock_response(status=200, json_data=None, text_data=""):
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    resp.text = AsyncMock(return_value=text_data)
    return resp


class MockContextManager:
    def __init__(self, resp):
        self.resp = resp

    async def __aenter__(self):
        return self.resp

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def broker():
    return AlpacaBroker(
        api_key="test_key", secret_key="test_secret", base_url="https://mock.api"
    )


# ---------- _parse_timestamp ----------


class TestParseTimestamp:
    def test_valid_iso(self):
        ts = _parse_timestamp("2026-03-21T10:00:00Z")
        assert isinstance(ts, datetime)

    def test_none_returns_none(self):
        assert _parse_timestamp(None) is None

    def test_empty_returns_none(self):
        assert _parse_timestamp("") is None

    def test_invalid_returns_none(self):
        assert _parse_timestamp("not-a-date") is None


# ---------- connect / disconnect ----------


class TestConnectDisconnect:
    @pytest.mark.asyncio
    async def test_connect_creates_session(self, broker):
        await broker.connect()
        assert broker._session is not None
        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_clears_session(self, broker):
        await broker.connect()
        await broker.disconnect()
        assert broker._session is None

    @pytest.mark.asyncio
    async def test_disconnect_when_no_session(self, broker):
        await broker.disconnect()  # should not raise


# ---------- _headers ----------


class TestHeaders:
    def test_headers_contain_keys(self, broker):
        h = broker._headers
        assert h["APCA-API-KEY-ID"] == "test_key"
        assert h["APCA-API-SECRET-KEY"] == "test_secret"


# ---------- _api_call ----------


class TestApiCall:
    @pytest.mark.asyncio
    async def test_api_call_200(self, broker):
        resp = _mock_response(200, {"result": "ok"})
        await broker.connect()
        broker._session.request = MagicMock(return_value=MockContextManager(resp))
        data = await broker._api_call("GET", "/v2/test")
        assert data == {"result": "ok"}
        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_api_call_204(self, broker):
        resp = _mock_response(204)
        await broker.connect()
        broker._session.request = MagicMock(return_value=MockContextManager(resp))
        data = await broker._api_call("DELETE", "/v2/test")
        assert data is None
        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_api_call_207(self, broker):
        resp = _mock_response(207, [{"id": "1"}])
        await broker.connect()
        broker._session.request = MagicMock(return_value=MockContextManager(resp))
        data = await broker._api_call("DELETE", "/v2/positions")
        assert data == [{"id": "1"}]
        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_api_call_error_status(self, broker):
        resp = _mock_response(422, text_data="validation error")
        await broker.connect()
        broker._session.request = MagicMock(return_value=MockContextManager(resp))
        data = await broker._api_call("POST", "/v2/orders")
        assert data is None
        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_api_call_exception(self, broker):
        await broker.connect()
        broker._session.request = MagicMock(side_effect=Exception("network error"))
        data = await broker._api_call("GET", "/v2/test")
        assert data is None
        await broker.disconnect()

    @pytest.mark.asyncio
    async def test_api_call_auto_connects(self, broker):
        resp = _mock_response(200, {"ok": True})
        with patch.object(broker, "connect", new_callable=AsyncMock) as mock_connect:
            # Simulate that connect creates a session
            async def fake_connect():
                broker._session = MagicMock()
                broker._session.closed = False
                broker._session.request = MagicMock(
                    return_value=MockContextManager(resp)
                )

            mock_connect.side_effect = fake_connect
            data = await broker._api_call("GET", "/test")
            mock_connect.assert_called_once()
            assert data == {"ok": True}


# ---------- _parse_account ----------


class TestParseAccount:
    def test_parse_full(self):
        data = {
            "id": "acc123",
            "status": "ACTIVE",
            "currency": "USD",
            "cash": "50000.00",
            "portfolio_value": "100000.00",
            "buying_power": "200000.00",
            "equity": "100000.00",
            "last_equity": "99000.00",
            "long_market_value": "60000.00",
            "short_market_value": "10000.00",
            "pattern_day_trader": True,
            "trading_blocked": False,
            "account_blocked": False,
        }
        acct = AlpacaBroker._parse_account(data)
        assert acct.account_id == "acc123"
        assert acct.portfolio_value == 100000.0
        assert acct.pattern_day_trader is True

    def test_parse_missing_fields(self):
        acct = AlpacaBroker._parse_account({})
        assert acct.account_id == ""
        assert acct.portfolio_value == 0.0


# ---------- get_account ----------


class TestGetAccount:
    @pytest.mark.asyncio
    async def test_get_account_success(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = {"id": "a1", "portfolio_value": "50000"}
            acct = await broker.get_account()
            assert acct.portfolio_value == 50000.0

    @pytest.mark.asyncio
    async def test_get_account_failure_raises(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = None
            with pytest.raises(ConnectionError):
                await broker.get_account()


# ---------- _parse_position ----------


class TestParsePosition:
    def test_parse_full(self):
        data = {
            "symbol": "AAPL",
            "qty": "10",
            "side": "long",
            "market_value": "1500.00",
            "cost_basis": "1400.00",
            "unrealized_pl": "100.00",
            "unrealized_plpc": "0.07",
            "current_price": "150.00",
            "avg_entry_price": "140.00",
        }
        pos = AlpacaBroker._parse_position(data)
        assert pos.ticker == "AAPL"
        assert pos.quantity == 10.0
        assert pos.unrealized_pnl == 100.0


# ---------- get_positions ----------


class TestGetPositions:
    @pytest.mark.asyncio
    async def test_returns_list(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = [{"symbol": "AAPL", "qty": "5"}]
            positions = await broker.get_positions()
            assert len(positions) == 1
            assert positions[0].ticker == "AAPL"

    @pytest.mark.asyncio
    async def test_returns_empty_on_none(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = None
            assert await broker.get_positions() == []


# ---------- get_position ----------


class TestGetPosition:
    @pytest.mark.asyncio
    async def test_returns_position(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = {"symbol": "GOOGL", "qty": "3"}
            pos = await broker.get_position("GOOGL")
            assert pos.ticker == "GOOGL"

    @pytest.mark.asyncio
    async def test_returns_none(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = None
            assert await broker.get_position("MISSING") is None


# ---------- submit_order ----------


class TestSubmitOrder:
    @pytest.mark.asyncio
    async def test_market_order_success(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = {"id": "order123", "status": "accepted"}
            req = OrderRequest(ticker="AAPL", side=OrderSide.BUY, quantity=10)
            result = await broker.submit_order(req)
            assert result.success is True
            assert result.order_id == "order123"

    @pytest.mark.asyncio
    async def test_limit_order(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = {"id": "o2", "status": "new"}
            req = OrderRequest(
                ticker="MSFT",
                side=OrderSide.SELL,
                quantity=5,
                order_type=OrderType.LIMIT,
                limit_price=300.0,
            )
            result = await broker.submit_order(req)
            assert result.success is True
            call_data = mock.call_args[0][2]
            assert call_data["limit_price"] == "300.0"

    @pytest.mark.asyncio
    async def test_stop_order(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = {"id": "o3", "status": "new"}
            req = OrderRequest(
                ticker="TSLA",
                side=OrderSide.SELL,
                quantity=2,
                order_type=OrderType.STOP,
                stop_price=200.0,
            )
            result = await broker.submit_order(req)
            call_data = mock.call_args[0][2]
            assert call_data["stop_price"] == "200.0"

    @pytest.mark.asyncio
    async def test_order_failure(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = None
            req = OrderRequest(ticker="AAPL", side=OrderSide.BUY, quantity=10)
            result = await broker.submit_order(req)
            assert result.success is False


# ---------- cancel_order ----------


class TestCancelOrder:
    @pytest.mark.asyncio
    async def test_cancel(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = None  # DELETE returns 204
            result = await broker.cancel_order("order123")
            assert result.success is True
            assert result.status == OrderStatus.CANCELLED


# ---------- get_order ----------


class TestGetOrder:
    @pytest.mark.asyncio
    async def test_get_order(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = {
                "id": "o1",
                "symbol": "AAPL",
                "side": "buy",
                "type": "market",
                "qty": "10",
                "filled_qty": "10",
                "status": "filled",
                "time_in_force": "day",
            }
            order = await broker.get_order("o1")
            assert order.order_id == "o1"
            assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_get_order_none(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = None
            assert await broker.get_order("missing") is None


# ---------- get_open_orders ----------


class TestGetOpenOrders:
    @pytest.mark.asyncio
    async def test_returns_list(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = [
                {
                    "id": "o1",
                    "symbol": "AAPL",
                    "side": "buy",
                    "type": "market",
                    "qty": "10",
                    "filled_qty": "0",
                    "status": "new",
                    "time_in_force": "day",
                }
            ]
            orders = await broker.get_open_orders()
            assert len(orders) == 1

    @pytest.mark.asyncio
    async def test_returns_empty(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = None
            assert await broker.get_open_orders() == []


# ---------- _parse_order ----------


class TestParseOrder:
    def test_parse_with_prices(self):
        data = {
            "id": "o1",
            "symbol": "AAPL",
            "side": "buy",
            "type": "limit",
            "qty": "10",
            "filled_qty": "5",
            "status": "partially_filled",
            "time_in_force": "gtc",
            "limit_price": "150.00",
            "stop_price": None,
            "filled_avg_price": "149.50",
            "created_at": "2026-03-21T10:00:00Z",
            "updated_at": "2026-03-21T10:01:00Z",
        }
        order = AlpacaBroker._parse_order(data)
        assert order.limit_price == 150.0
        assert order.filled_avg_price == 149.5
        assert order.created_at is not None

    def test_unknown_status_defaults_to_new(self):
        data = {
            "id": "o1",
            "symbol": "X",
            "side": "buy",
            "type": "market",
            "qty": "1",
            "filled_qty": "0",
            "status": "weird_status",
            "time_in_force": "day",
        }
        order = AlpacaBroker._parse_order(data)
        assert order.status == OrderStatus.NEW


# ---------- close_position ----------


class TestClosePosition:
    @pytest.mark.asyncio
    async def test_close_success(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = {"id": "close1", "status": "accepted"}
            result = await broker.close_position("AAPL")
            assert result.success is True

    @pytest.mark.asyncio
    async def test_close_failure(self, broker):
        with patch.object(broker, "_api_call", new_callable=AsyncMock) as mock:
            mock.return_value = None
            result = await broker.close_position("AAPL")
            assert result.success is False


# ---------- close_all_positions ----------


class TestCloseAllPositions:
    @pytest.mark.asyncio
    async def test_no_positions(self, broker):
        with patch.object(
            broker, "_api_call", new_callable=AsyncMock
        ) as mock_api, patch.object(
            broker, "get_positions", new_callable=AsyncMock
        ) as mock_pos:
            mock_pos.return_value = []
            results = await broker.close_all_positions()
            assert results == []

    @pytest.mark.asyncio
    async def test_closes_all(self, broker):
        pos1 = MagicMock(ticker="AAPL")
        pos2 = MagicMock(ticker="GOOGL")

        call_count = 0

        async def fake_get_positions():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [pos1, pos2]
            return []  # verify call

        with patch.object(
            broker, "_api_call", new_callable=AsyncMock
        ) as mock_api, patch.object(
            broker, "get_positions", side_effect=fake_get_positions
        ), patch.object(
            broker, "close_position", new_callable=AsyncMock
        ) as mock_close:
            mock_close.return_value = OrderResult(success=True, order_id="x")
            results = await broker.close_all_positions()
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_retries_failed_close(self, broker):
        pos = MagicMock(ticker="AAPL")

        call_count = 0

        async def fake_close(ticker):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return OrderResult(success=False, message="fail")
            return OrderResult(success=True, order_id="retry_ok")

        with patch.object(broker, "_api_call", new_callable=AsyncMock), patch.object(
            broker, "get_positions", new_callable=AsyncMock
        ) as mock_pos, patch.object(broker, "close_position", side_effect=fake_close):
            mock_pos.side_effect = [
                [pos],
                [],
            ]  # first call returns pos, verify returns empty
            results = await broker.close_all_positions()
            assert len(results) == 1
            assert results[0].success is True


# ---------- _safe_order_status ----------


class TestSafeOrderStatus:
    def test_valid_status(self):
        assert AlpacaBroker._safe_order_status("filled") == OrderStatus.FILLED

    def test_unknown_defaults_new(self):
        assert AlpacaBroker._safe_order_status("unknown_xyz") == OrderStatus.NEW


# ---------- Constructor defaults ----------


class TestConstructorDefaults:
    def test_default_env_vars(self):
        with patch.dict(
            "os.environ", {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}
        ):
            b = AlpacaBroker()
            assert b.api_key == "k"
            assert b.secret_key == "s"
            assert "paper-api" in b.base_url
