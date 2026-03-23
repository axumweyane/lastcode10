"""
OANDA v20 REST API broker for direct forex trading.

Implements BaseBroker for FX pair execution on OANDA practice/live accounts.
Pair format: EUR_USD (OANDA native), converted from EURUSD internally.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from trading.broker.base import (
    AccountInfo,
    BaseBroker,
    OrderInfo,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionInfo,
    TimeInForce,
)

logger = logging.getLogger(__name__)

# FX pair conversion: EURUSD -> EUR_USD
_KNOWN_FX = {
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
    "NZDUSD", "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURAUD",
}


def to_oanda_pair(symbol: str) -> str:
    """Convert EURUSD -> EUR_USD format."""
    s = symbol.upper().replace("_", "").replace("/", "")
    if s in _KNOWN_FX:
        return f"{s[:3]}_{s[3:]}"
    if "_" in symbol:
        return symbol.upper()
    return f"{symbol[:3]}_{symbol[3:]}" if len(symbol) == 6 else symbol


def from_oanda_pair(instrument: str) -> str:
    """Convert EUR_USD -> EURUSD format."""
    return instrument.replace("_", "")


class OandaBroker(BaseBroker):
    """
    OANDA v20 REST API broker for forex execution.

    Usage:
        broker = OandaBroker()
        await broker.connect()
        result = await broker.submit_order(OrderRequest(
            ticker="EURUSD", side=OrderSide.BUY, quantity=1000
        ))
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        account_id: Optional[str] = None,
        account_type: Optional[str] = None,
    ):
        self._token = api_token or os.environ.get("OANDA_API_TOKEN", "")
        self._account_id = account_id or os.environ.get("OANDA_ACCOUNT_ID", "")
        self._account_type = (
            account_type or os.environ.get("OANDA_ACCOUNT_TYPE", "practice")
        )

        if self._account_type == "live":
            self._base_url = "https://api-fxtrade.oanda.com"
        else:
            self._base_url = "https://api-fxpractice.oanda.com"

        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    async def connect(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                headers=self._headers, timeout=timeout
            )
        # Verify connection
        try:
            info = await self.get_account()
            logger.info(
                "OANDA connected: account=%s balance=$%.2f",
                self._account_id,
                info.equity,
            )
        except Exception as e:
            logger.error("OANDA connection failed: %s", e)
            raise

    async def disconnect(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _request(
        self, method: str, path: str, json_data: Optional[dict] = None
    ) -> dict:
        if self._session is None or self._session.closed:
            await self.connect()

        url = f"{self._base_url}{path}"
        async with self._session.request(method, url, json=json_data) as resp:
            data = await resp.json()
            if resp.status >= 400:
                error_msg = data.get("errorMessage", str(data))
                logger.error("OANDA %s %s -> %d: %s", method, path, resp.status, error_msg)
                raise RuntimeError(f"OANDA API error {resp.status}: {error_msg}")
            return data

    async def get_account(self) -> AccountInfo:
        data = await self._request("GET", f"/v3/accounts/{self._account_id}/summary")
        acct = data["account"]
        balance = float(acct["balance"])
        nav = float(acct["NAV"])
        unrealized = float(acct["unrealizedPL"])
        margin_used = float(acct["marginUsed"])
        margin_avail = float(acct["marginAvailable"])

        return AccountInfo(
            account_id=acct["id"],
            status="ACTIVE",
            currency=acct["currency"],
            cash=balance,
            portfolio_value=nav,
            buying_power=margin_avail,
            equity=nav,
            last_equity=balance,
            long_market_value=0.0,
            short_market_value=0.0,
            raw=acct,
        )

    async def get_positions(self) -> List[PositionInfo]:
        data = await self._request(
            "GET", f"/v3/accounts/{self._account_id}/openPositions"
        )
        positions = []
        for pos in data.get("positions", []):
            pinfo = self._parse_position(pos)
            if pinfo is not None:
                positions.append(pinfo)
        return positions

    async def get_position(self, ticker: str) -> Optional[PositionInfo]:
        instrument = to_oanda_pair(ticker)
        try:
            data = await self._request(
                "GET",
                f"/v3/accounts/{self._account_id}/positions/{instrument}",
            )
            return self._parse_position(data.get("position", {}))
        except RuntimeError:
            return None

    def _parse_position(self, pos: dict) -> Optional[PositionInfo]:
        instrument = pos.get("instrument", "")
        long_units = int(pos.get("long", {}).get("units", "0"))
        short_units = int(pos.get("short", {}).get("units", "0"))

        if long_units == 0 and short_units == 0:
            return None

        if long_units > 0:
            side_data = pos["long"]
            qty = long_units
            side = "long"
        else:
            side_data = pos["short"]
            qty = abs(short_units)
            side = "short"

        avg_price = float(side_data.get("averagePrice", "0"))
        unrealized = float(side_data.get("unrealizedPL", "0"))
        # OANDA doesn't give current_price directly in position; estimate it
        if avg_price > 0 and qty > 0:
            if side == "long":
                current_price = avg_price + (unrealized / qty)
            else:
                current_price = avg_price - (unrealized / qty)
        else:
            current_price = avg_price

        cost_basis = avg_price * qty
        pnl_pct = (unrealized / cost_basis * 100) if cost_basis > 0 else 0.0

        return PositionInfo(
            ticker=from_oanda_pair(instrument),
            quantity=float(qty if side == "long" else -qty),
            side=side,
            market_value=current_price * qty * (1 if side == "long" else -1),
            cost_basis=cost_basis,
            unrealized_pnl=unrealized,
            unrealized_pnl_percent=pnl_pct,
            current_price=current_price,
            avg_entry_price=avg_price,
            raw=pos,
        )

    async def submit_order(self, request: OrderRequest) -> OrderResult:
        instrument = to_oanda_pair(request.ticker)

        # OANDA uses signed units: positive = buy, negative = sell
        units = int(request.quantity)
        if request.side == OrderSide.SELL:
            units = -units

        order_body: Dict[str, Any] = {
            "order": {
                "instrument": instrument,
                "units": str(units),
                "type": "MARKET",
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }

        if request.order_type == OrderType.LIMIT and request.limit_price:
            order_body["order"]["type"] = "LIMIT"
            order_body["order"]["price"] = str(request.limit_price)
            order_body["order"]["timeInForce"] = "GTC"

        try:
            data = await self._request(
                "POST",
                f"/v3/accounts/{self._account_id}/orders",
                json_data=order_body,
            )

            # Check for fill
            fill = data.get("orderFillTransaction")
            if fill:
                order_id = fill.get("id", "")
                fill_price = float(fill.get("price", "0"))
                logger.info(
                    "OANDA order filled: %s %s %d units @ %.5f (id=%s)",
                    request.side.value,
                    instrument,
                    abs(units),
                    fill_price,
                    order_id,
                )
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    status=OrderStatus.FILLED,
                    message=f"Filled {abs(units)} {instrument} @ {fill_price}",
                    raw=data,
                )

            # Check for order create (limit/pending)
            create = data.get("orderCreateTransaction")
            if create:
                order_id = create.get("id", "")
                logger.info(
                    "OANDA order created: %s %s %d units (id=%s)",
                    request.side.value,
                    instrument,
                    abs(units),
                    order_id,
                )
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    status=OrderStatus.ACCEPTED,
                    message=f"Order created for {abs(units)} {instrument}",
                    raw=data,
                )

            # Check for rejection
            cancel = data.get("orderCancelTransaction")
            if cancel:
                reason = cancel.get("reason", "unknown")
                logger.warning("OANDA order rejected: %s", reason)
                return OrderResult(
                    success=False,
                    status=OrderStatus.REJECTED,
                    message=f"Order rejected: {reason}",
                    raw=data,
                )

            return OrderResult(
                success=False,
                message=f"Unexpected response: {data}",
                raw=data,
            )

        except Exception as e:
            logger.error("OANDA submit_order failed: %s", e)
            return OrderResult(success=False, message=str(e))

    async def cancel_order(self, order_id: str) -> OrderResult:
        try:
            data = await self._request(
                "PUT",
                f"/v3/accounts/{self._account_id}/orders/{order_id}/cancel",
            )
            return OrderResult(
                success=True,
                order_id=order_id,
                status=OrderStatus.CANCELLED,
                message="Order cancelled",
                raw=data,
            )
        except Exception as e:
            return OrderResult(success=False, message=str(e))

    async def get_order(self, order_id: str) -> Optional[OrderInfo]:
        try:
            data = await self._request(
                "GET", f"/v3/accounts/{self._account_id}/orders/{order_id}"
            )
            order = data.get("order", {})
            return OrderInfo(
                order_id=order.get("id", order_id),
                ticker=from_oanda_pair(order.get("instrument", "")),
                side=OrderSide.BUY if int(order.get("units", "0")) > 0 else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=abs(int(order.get("units", "0"))),
                filled_quantity=0,
                status=OrderStatus.ACCEPTED,
                time_in_force=TimeInForce.GTC,
                raw=order,
            )
        except Exception:
            return None

    async def get_open_orders(self) -> List[OrderInfo]:
        try:
            data = await self._request(
                "GET", f"/v3/accounts/{self._account_id}/pendingOrders"
            )
            orders = []
            for o in data.get("orders", []):
                units = int(o.get("units", "0"))
                orders.append(
                    OrderInfo(
                        order_id=o.get("id", ""),
                        ticker=from_oanda_pair(o.get("instrument", "")),
                        side=OrderSide.BUY if units > 0 else OrderSide.SELL,
                        order_type=OrderType.LIMIT,
                        quantity=abs(units),
                        filled_quantity=0,
                        status=OrderStatus.ACCEPTED,
                        time_in_force=TimeInForce.GTC,
                        raw=o,
                    )
                )
            return orders
        except Exception:
            return []

    async def close_position(self, ticker: str) -> OrderResult:
        instrument = to_oanda_pair(ticker)
        try:
            # Determine which side to close
            pos = await self.get_position(ticker)
            if pos is None:
                return OrderResult(success=False, message=f"No open position for {ticker}")

            if pos.quantity > 0:
                close_body = {"longUnits": "ALL"}
            else:
                close_body = {"shortUnits": "ALL"}

            data = await self._request(
                "PUT",
                f"/v3/accounts/{self._account_id}/positions/{instrument}/close",
                json_data=close_body,
            )

            results = []
            for key in ("longOrderFillTransaction", "shortOrderFillTransaction"):
                fill = data.get(key)
                if fill:
                    results.append(fill.get("id", ""))

            if results:
                logger.info("OANDA closed position: %s (txn=%s)", instrument, results)
                return OrderResult(
                    success=True,
                    order_id=results[0],
                    status=OrderStatus.FILLED,
                    message=f"Closed {instrument}",
                    raw=data,
                )

            return OrderResult(
                success=False,
                message=f"No fills when closing {instrument}",
                raw=data,
            )
        except Exception as e:
            return OrderResult(success=False, message=str(e))

    async def close_all_positions(self) -> List[OrderResult]:
        positions = await self.get_positions()
        results = []
        for pos in positions:
            result = await self.close_position(pos.ticker)
            results.append(result)
        return results
