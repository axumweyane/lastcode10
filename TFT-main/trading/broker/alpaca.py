"""
Alpaca broker implementation.
Uses aiohttp for async API calls, mirroring the existing trading-engine pattern.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

from .base import (
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

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"


class AlpacaBroker(BaseBroker):

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = (base_url or os.getenv("ALPACA_BASE_URL", PAPER_BASE_URL)).rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json",
        }

    async def connect(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        logger.info("AlpacaBroker connected (base_url=%s)", self.base_url)

    async def disconnect(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        logger.info("AlpacaBroker disconnected")

    async def _api_call(
        self, method: str, endpoint: str, data: Optional[Dict] = None
    ) -> Optional[Any]:
        if self._session is None or self._session.closed:
            await self.connect()

        url = f"{self.base_url}{endpoint}"
        try:
            async with self._session.request(
                method=method,
                url=url,
                headers=self._headers,
                json=data if data else None,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status == 204:
                    return None
                if resp.status in (207,):
                    # Multi-status for bulk close
                    return await resp.json()
                body = await resp.text()
                logger.error("Alpaca %s %s -> %s: %s", method, endpoint, resp.status, body)
                return None
        except Exception as e:
            logger.error("Alpaca API call failed %s %s: %s", method, endpoint, e)
            return None

    # ---- Account ----

    async def get_account(self) -> AccountInfo:
        data = await self._api_call("GET", "/v2/account")
        if data is None:
            raise ConnectionError("Failed to retrieve Alpaca account")
        return self._parse_account(data)

    @staticmethod
    def _parse_account(data: Dict) -> AccountInfo:
        return AccountInfo(
            account_id=data.get("id", ""),
            status=data.get("status", ""),
            currency=data.get("currency", "USD"),
            cash=float(data.get("cash", 0)),
            portfolio_value=float(data.get("portfolio_value", 0)),
            buying_power=float(data.get("buying_power", 0)),
            equity=float(data.get("equity", 0)),
            last_equity=float(data.get("last_equity", 0)),
            long_market_value=float(data.get("long_market_value", 0)),
            short_market_value=float(data.get("short_market_value", 0)),
            pattern_day_trader=data.get("pattern_day_trader", False),
            trading_blocked=data.get("trading_blocked", False),
            account_blocked=data.get("account_blocked", False),
            raw=data,
        )

    # ---- Positions ----

    async def get_positions(self) -> List[PositionInfo]:
        data = await self._api_call("GET", "/v2/positions")
        if data is None:
            return []
        return [self._parse_position(p) for p in data]

    async def get_position(self, ticker: str) -> Optional[PositionInfo]:
        data = await self._api_call("GET", f"/v2/positions/{ticker}")
        if data is None:
            return None
        return self._parse_position(data)

    @staticmethod
    def _parse_position(data: Dict) -> PositionInfo:
        return PositionInfo(
            ticker=data.get("symbol", ""),
            quantity=float(data.get("qty", 0)),
            side=data.get("side", "long"),
            market_value=float(data.get("market_value", 0)),
            cost_basis=float(data.get("cost_basis", 0)),
            unrealized_pnl=float(data.get("unrealized_pl", 0)),
            unrealized_pnl_percent=float(data.get("unrealized_plpc", 0)),
            current_price=float(data.get("current_price", 0)),
            avg_entry_price=float(data.get("avg_entry_price", 0)),
            raw=data,
        )

    # ---- Orders ----

    async def submit_order(self, request: OrderRequest) -> OrderResult:
        order_data: Dict[str, Any] = {
            "symbol": request.ticker,
            "qty": str(request.quantity),
            "side": request.side.value,
            "type": request.order_type.value,
            "time_in_force": request.time_in_force.value,
        }
        if request.limit_price is not None:
            order_data["limit_price"] = str(request.limit_price)
        if request.stop_price is not None:
            order_data["stop_price"] = str(request.stop_price)

        data = await self._api_call("POST", "/v2/orders", order_data)
        if data is None:
            return OrderResult(success=False, message="Order submission failed")

        return OrderResult(
            success=True,
            order_id=data.get("id"),
            status=self._safe_order_status(data.get("status", "")),
            message="Order submitted",
            raw=data,
        )

    async def cancel_order(self, order_id: str) -> OrderResult:
        result = await self._api_call("DELETE", f"/v2/orders/{order_id}")
        # DELETE returns 204 on success (result is None)
        return OrderResult(
            success=True,
            order_id=order_id,
            status=OrderStatus.CANCELLED,
            message="Order cancel requested",
        )

    async def get_order(self, order_id: str) -> Optional[OrderInfo]:
        data = await self._api_call("GET", f"/v2/orders/{order_id}")
        if data is None:
            return None
        return self._parse_order(data)

    async def get_open_orders(self) -> List[OrderInfo]:
        data = await self._api_call("GET", "/v2/orders?status=open")
        if data is None:
            return []
        return [self._parse_order(o) for o in data]

    @staticmethod
    def _parse_order(data: Dict) -> OrderInfo:
        return OrderInfo(
            order_id=data.get("id", ""),
            ticker=data.get("symbol", ""),
            side=OrderSide(data.get("side", "buy")),
            order_type=OrderType(data.get("type", "market")),
            quantity=float(data.get("qty", 0)),
            filled_quantity=float(data.get("filled_qty", 0)),
            status=AlpacaBroker._safe_order_status(data.get("status", "")),
            time_in_force=TimeInForce(data.get("time_in_force", "day")),
            limit_price=float(data["limit_price"]) if data.get("limit_price") else None,
            stop_price=float(data["stop_price"]) if data.get("stop_price") else None,
            filled_avg_price=float(data["filled_avg_price"]) if data.get("filled_avg_price") else None,
            created_at=_parse_timestamp(data.get("created_at")),
            updated_at=_parse_timestamp(data.get("updated_at")),
            raw=data,
        )

    # ---- Close positions ----

    async def close_position(self, ticker: str) -> OrderResult:
        data = await self._api_call("DELETE", f"/v2/positions/{ticker}")
        if data is None:
            return OrderResult(success=False, message=f"Failed to close position {ticker}")
        return OrderResult(
            success=True,
            order_id=data.get("id"),
            status=self._safe_order_status(data.get("status", "")),
            message=f"Close order submitted for {ticker}",
            raw=data,
        )

    async def close_all_positions(self) -> List[OrderResult]:
        """Cancel all open orders, close all positions, verify."""
        # 1. Cancel all open orders
        await self._api_call("DELETE", "/v2/orders")
        logger.info("Cancelled all open orders")

        # 2. Get current positions before closing
        positions = await self.get_positions()
        if not positions:
            logger.info("No positions to close")
            return []

        # 3. Close each position individually for per-position audit
        results: List[OrderResult] = []
        for pos in positions:
            result = await self.close_position(pos.ticker)
            if not result.success:
                # Retry up to 2 more times with brief pause
                import asyncio
                for attempt in range(2):
                    await asyncio.sleep(1)
                    result = await self.close_position(pos.ticker)
                    if result.success:
                        break
            results.append(result)

        # 4. Verify no positions remain
        remaining = await self.get_positions()
        if remaining:
            logger.warning(
                "Positions still open after close_all: %s",
                [p.ticker for p in remaining],
            )

        return results

    @staticmethod
    def _safe_order_status(status_str: str) -> OrderStatus:
        try:
            return OrderStatus(status_str)
        except ValueError:
            logger.warning("Unknown order status: %s", status_str)
            return OrderStatus.NEW


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
