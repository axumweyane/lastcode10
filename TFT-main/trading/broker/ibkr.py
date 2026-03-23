"""
Interactive Brokers broker via IB Gateway + ib_insync.

Implements BaseBroker for options execution on IBKR paper/live accounts.
Connects to IB Gateway on localhost:4002 (paper) or localhost:4001 (live).

Options ticker format: AAPL_250418C250 (symbol_YYMMDD[C|P]strike)
Stock ticker format: AAPL (plain symbol, for position lookups)
"""

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

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

# Override mapping for symbols whose standard listed tradingClass differs
# from the ticker symbol. Most S&P 500 symbols have tradingClass == symbol,
# so the default behavior (tradingClass = symbol) is correct. Add entries
# here ONLY for symbols where the standard listed tradingClass is different.
# Example: if XYZ standard options traded as "XYZ1", add {"XYZ": "XYZ1"}.
IBKR_TRADING_CLASS_OVERRIDE: Dict[str, str] = {
    # Currently all S&P 500 symbols use tradingClass == symbol for standard
    # listed options. Flex variants (e.g. "2META", "2AMZN") are IB-cleared
    # and cannot be traded on paper accounts — they are avoided by always
    # passing tradingClass = symbol (or override) on every option contract.
}


def parse_option_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Parse AAPL_250418C250 -> {symbol, expiry, right, strike}.
    Returns None if ticker is not an option format.
    """
    if "_" not in ticker:
        return None
    parts = ticker.split("_", 1)
    if len(parts) != 2 or len(parts[1]) < 5:
        return None

    symbol = parts[0]
    rest = parts[1]

    cp_idx = -1
    for i, ch in enumerate(rest):
        if ch in ("C", "P") and i >= 6:
            cp_idx = i
            break

    if cp_idx < 0:
        return None

    expiry = rest[:cp_idx]
    right = rest[cp_idx]
    strike_str = rest[cp_idx + 1:]

    try:
        strike = float(strike_str)
        if len(expiry) == 6:
            expiry = "20" + expiry
        return {
            "symbol": symbol,
            "expiry": expiry,
            "right": right,
            "strike": strike,
        }
    except ValueError:
        return None


def build_option_ticker(symbol: str, expiry: str, right: str, strike: float) -> str:
    """Build AAPL_250418C250 from components."""
    exp = expiry[2:] if len(expiry) == 8 else expiry
    strike_str = f"{strike:g}"
    return f"{symbol}_{exp}{right}{strike_str}"


class IBKRBroker(BaseBroker):
    """
    Interactive Brokers broker for options execution via IB Gateway.

    Uses ib_insync with nest_asyncio for compatibility with existing
    asyncio event loops (FastAPI/uvicorn).

    Usage:
        broker = IBKRBroker()
        await broker.connect()
        result = await broker.submit_order(OrderRequest(
            ticker="AAPL_250418C250", side=OrderSide.BUY, quantity=1
        ))
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
    ):
        self._host = host or os.environ.get("IBKR_HOST", "127.0.0.1")
        self._port = port or int(os.environ.get("IBKR_PORT", "4002"))
        self._client_id = client_id or int(os.environ.get("IBKR_CLIENT_ID", "10"))
        self._ib = None
        self._ib_loop = None
        self._ib_thread = None

    def _start_ib_thread(self):
        """Start dedicated thread with its own event loop for ib_insync."""
        if self._ib_thread is not None and self._ib_thread.is_alive():
            return

        ready = threading.Event()

        def _thread_main():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._ib_loop = loop
            import nest_asyncio
            nest_asyncio.apply(loop)
            from ib_insync import IB
            self._ib = IB()
            ready.set()
            loop.run_forever()

        self._ib_thread = threading.Thread(
            target=_thread_main, daemon=True, name="ib_insync",
        )
        self._ib_thread.start()
        if not ready.wait(timeout=10):
            raise RuntimeError("IB thread failed to start")

    async def _async_call_ib(self, fn, timeout=60):
        """Run fn() in the IB thread, awaitable from async context."""
        fut = concurrent.futures.Future()

        def _wrapper():
            try:
                fut.set_result(fn())
            except Exception as e:
                fut.set_exception(e)

        self._ib_loop.call_soon_threadsafe(_wrapper)
        return await asyncio.wrap_future(fut)

    def _make_contract(self, ticker: str):
        """Create ib_insync Contract from ticker string.

        ALWAYS sets tradingClass on option contracts to ensure we get
        standard listed options, never Flex variants (e.g. "2META").
        """
        from ib_insync import Option, Stock

        parsed = parse_option_ticker(ticker)
        if parsed:
            sym = parsed["symbol"]
            # Always pass tradingClass: use override if present, else symbol name
            tc = IBKR_TRADING_CLASS_OVERRIDE.get(sym, sym)
            kwargs = dict(
                symbol=sym,
                lastTradeDateOrContractMonth=parsed["expiry"],
                strike=parsed["strike"],
                right=parsed["right"],
                exchange="SMART",
                currency="USD",
                tradingClass=tc,
            )
            return Option(**kwargs)
        else:
            return Stock(ticker, "SMART", "USD")

    async def find_atm_option(
        self, symbol: str, right: str, stock_price: float, max_premium: float = 500.0,
    ) -> Optional[str]:
        """
        Find an ATM option contract for the given symbol.

        Picks the nearest monthly expiry 3-6 weeks out, finds the strike
        closest to stock_price, qualifies the contract, and returns the
        option ticker string (e.g., AAPL_260417C250) or None on failure.
        """
        from datetime import date, timedelta

        today = date.today()
        min_expiry = today + timedelta(weeks=3)
        max_expiry = today + timedelta(weeks=6)

        def _find():
            from ib_insync import Stock as IBStock, Option as IBOption

            # Qualify the underlying to get valid contract details
            underlying = IBStock(symbol, "SMART", "USD")
            qualified = self._ib.qualifyContracts(underlying)
            if not qualified:
                return None

            # Request option chains
            chains = self._ib.reqSecDefOptParams(
                underlying.symbol, "", underlying.secType, underlying.conId,
            )
            if not chains:
                return None

            # Find SMART exchange chain, preferring standard tradingClass
            # (tradingClass == symbol) over Flex variants (e.g. "2META").
            chain = None
            tc_override = IBKR_TRADING_CLASS_OVERRIDE.get(symbol)
            preferred_tc = tc_override or symbol
            smart_chains = [c for c in chains if c.exchange == "SMART"]
            for c in smart_chains:
                if c.tradingClass == preferred_tc:
                    chain = c
                    break
            if chain is None and smart_chains:
                # Fall back to any SMART chain that is NOT a Flex variant
                for c in smart_chains:
                    if not c.tradingClass.startswith(("2", "3")):
                        chain = c
                        break
            if chain is None and smart_chains:
                chain = smart_chains[0]
            if chain is None:
                chain = chains[0]

            # Filter expirations to 3-6 weeks out, prefer monthly (3rd Friday)
            valid_expiries = []
            for exp_str in sorted(chain.expirations):
                try:
                    exp_date = date(int(exp_str[:4]), int(exp_str[4:6]), int(exp_str[6:8]))
                except (ValueError, IndexError):
                    continue
                if min_expiry <= exp_date <= max_expiry:
                    valid_expiries.append((exp_str, exp_date))

            if not valid_expiries:
                # Fallback: take first expiry after min_expiry
                for exp_str in sorted(chain.expirations):
                    try:
                        exp_date = date(int(exp_str[:4]), int(exp_str[4:6]), int(exp_str[6:8]))
                    except (ValueError, IndexError):
                        continue
                    if exp_date >= min_expiry:
                        valid_expiries.append((exp_str, exp_date))
                        break

            if not valid_expiries:
                return None

            # Pick the first valid expiry
            chosen_expiry, _ = valid_expiries[0]

            # Find strike closest to stock price.  The chain's strike list
            # is global across all expiries — not every strike exists for
            # every expiry (e.g. monthly = $5 increments, weekly = $2.50).
            # Try the closest strike first, then fan out to nearby strikes.
            available_strikes = sorted(chain.strikes, key=lambda s: abs(s - stock_price))

            tc = chain.tradingClass or None
            for candidate_strike in available_strikes[:10]:
                opt_kwargs = dict(
                    symbol=symbol,
                    lastTradeDateOrContractMonth=chosen_expiry,
                    strike=candidate_strike,
                    right=right,
                    exchange="SMART",
                    currency="USD",
                )
                if tc:
                    opt_kwargs["tradingClass"] = tc
                opt = IBOption(**opt_kwargs)
                qual = self._ib.qualifyContracts(opt)
                if qual:
                    return build_option_ticker(symbol, chosen_expiry, right, candidate_strike)

            return None

        try:
            ticker = await self._async_call_ib(_find)
            if ticker:
                logger.info(
                    "ATM option found: %s (underlying=%s, price=%.2f, right=%s)",
                    ticker, symbol, stock_price, right,
                )
            return ticker
        except Exception as e:
            logger.error("find_atm_option failed for %s: %s", symbol, e)
            return None

    async def connect(self, retries: int = 3, delay: float = 5.0) -> None:
        """Connect to IB Gateway with automatic retries.

        Args:
            retries: Number of connection attempts (default 3).
            delay: Seconds to wait between retries (default 5).
        """
        self._start_ib_thread()
        if self._ib.isConnected():
            return

        last_error = None
        for attempt in range(1, retries + 1):
            try:
                def _do_connect():
                    self._ib.connect(
                        self._host, self._port, clientId=self._client_id,
                        timeout=30, readonly=False,
                    )
                    try:
                        self._ib.reqAccountSummary()
                        self._ib.sleep(2)
                        nav = 0.0
                        for av in self._ib.accountSummary():
                            if av.tag == "NetLiquidation" and av.currency == "USD":
                                nav = float(av.value)
                                break
                        if nav > 0:
                            logger.info("IBKR NAV: $%.2f", nav)
                    except Exception:
                        logger.info("IBKR connected (account summary pending)")

                await self._async_call_ib(_do_connect)
                logger.info(
                    "IBKR connected: host=%s port=%d client_id=%d (attempt %d/%d)",
                    self._host, self._port, self._client_id, attempt, retries,
                )
                return  # success

            except Exception as e:
                last_error = e
                logger.warning(
                    "IBKR connection attempt %d/%d failed: %s",
                    attempt, retries, e,
                )
                if attempt < retries:
                    await asyncio.sleep(delay)
                    # Reset IB instance for clean retry
                    try:
                        self._start_ib_thread()
                    except Exception:
                        pass

        logger.error("IBKR connection failed after %d attempts: %s", retries, last_error)
        raise ConnectionError(
            f"IBKR connection failed after {retries} attempts: {last_error}"
        )

    async def ensure_connected(self) -> bool:
        """Ensure IB Gateway connection is alive; reconnect if needed.

        Returns True if connected, False if reconnection failed.
        """
        if self._ib is not None and self._ib.isConnected():
            return True
        try:
            await self.connect(retries=3, delay=5.0)
            return True
        except Exception as e:
            logger.error("IBKR ensure_connected failed: %s", e)
            return False

    async def disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            await self._async_call_ib(lambda: self._ib.disconnect())
            logger.info("IBKR disconnected")
        if self._ib_loop is not None:
            self._ib_loop.call_soon_threadsafe(self._ib_loop.stop)

    async def get_account(self) -> AccountInfo:
        def _do():
            ib = self._ib
            ib.reqAccountSummary()
            ib.sleep(2)
            vals: Dict[str, str] = {}
            for av in ib.accountSummary():
                if av.currency == "USD":
                    vals[av.tag] = av.value
            acct_id = ""
            for av in ib.accountSummary():
                acct_id = av.account
                break
            return vals, acct_id

        vals, acct_id = await self._async_call_ib(_do)
        nav = float(vals.get("NetLiquidation", "0"))
        cash = float(vals.get("TotalCashValue", "0"))
        buying_power = float(vals.get("BuyingPower", "0"))

        return AccountInfo(
            account_id=acct_id,
            status="ACTIVE",
            currency="USD",
            cash=cash,
            portfolio_value=nav,
            buying_power=buying_power,
            equity=nav,
            last_equity=nav,
            long_market_value=float(vals.get("GrossPositionValue", "0")),
            short_market_value=0.0,
            raw=vals,
        )

    async def get_positions(self) -> List[PositionInfo]:
        raw_positions = await self._async_call_ib(lambda: self._ib.positions())
        result = []
        for pos in raw_positions:
            pinfo = self._parse_position(pos)
            if pinfo is not None:
                result.append(pinfo)
        return result

    async def get_position(self, ticker: str) -> Optional[PositionInfo]:
        positions = await self.get_positions()
        for p in positions:
            if p.ticker == ticker:
                return p
        return None

    def _parse_position(self, pos) -> Optional[PositionInfo]:
        """Parse ib_insync Position to PositionInfo."""
        contract = pos.contract
        qty = float(pos.position)
        if qty == 0:
            return None

        avg_cost = float(pos.avgCost)

        if contract.secType == "OPT":
            right = "C" if contract.right == "C" else "P"
            ticker = build_option_ticker(
                contract.symbol, contract.lastTradeDateOrContractMonth, right,
                float(contract.strike),
            )
            cost_basis = avg_cost * abs(qty)
        elif contract.secType == "STK":
            ticker = contract.symbol
            cost_basis = avg_cost * abs(qty)
        else:
            ticker = contract.localSymbol or contract.symbol
            cost_basis = avg_cost * abs(qty)

        side = "long" if qty > 0 else "short"

        return PositionInfo(
            ticker=ticker,
            quantity=qty,
            side=side,
            market_value=cost_basis,
            cost_basis=cost_basis,
            unrealized_pnl=0.0,
            unrealized_pnl_percent=0.0,
            current_price=avg_cost,
            avg_entry_price=avg_cost,
            raw={"contract": str(contract), "account": pos.account},
        )

    async def submit_order(self, request: OrderRequest) -> OrderResult:
        # Ensure connection before order
        if not await self.ensure_connected():
            return OrderResult(
                success=False,
                message="IBKR not connected and reconnection failed",
            )

        contract = self._make_contract(request.ticker)

        try:
            qualified = await self._async_call_ib(
                lambda: self._ib.qualifyContracts(contract)
            )
            if not qualified:
                logger.warning(
                    "IBKR qualifyContracts FAILED: %s — contract not found, skipping order",
                    request.ticker,
                )
                return OrderResult(
                    success=False,
                    message=f"Contract not found: {request.ticker}",
                )
            contract = qualified[0]
            logger.info(
                "IBKR qualifyContracts OK: %s (conId=%s, tradingClass=%s)",
                request.ticker, contract.conId, getattr(contract, "tradingClass", ""),
            )
        except Exception as e:
            logger.warning(
                "IBKR qualifyContracts ERROR: %s — %s, skipping order",
                request.ticker, e,
            )
            return OrderResult(
                success=False,
                message=f"Contract qualification failed: {e}",
            )

        from ib_insync import LimitOrder, MarketOrder

        action = "BUY" if request.side == OrderSide.BUY else "SELL"
        qty = int(request.quantity)

        if request.order_type == OrderType.LIMIT and request.limit_price:
            order = LimitOrder(action, qty, request.limit_price)
        else:
            order = MarketOrder(action, qty)

        order.tif = "DAY"

        try:
            def _do_order():
                trade = self._ib.placeOrder(contract, order)
                if request.order_type == OrderType.MARKET:
                    for _ in range(30):
                        self._ib.sleep(0.5)
                        if trade.orderStatus.status in (
                            "Filled", "Cancelled", "Inactive",
                        ):
                            break
                return trade

            trade = await self._async_call_ib(_do_order)
            status = trade.orderStatus.status
            order_id = str(trade.order.orderId)

            if status == "Filled":
                fill_price = trade.orderStatus.avgFillPrice
                logger.info(
                    "IBKR order filled: %s %d %s @ $%.2f (id=%s)",
                    action, qty, request.ticker, fill_price, order_id,
                )
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    status=OrderStatus.FILLED,
                    message=f"Filled {qty} {request.ticker} @ ${fill_price:.2f}",
                    raw={"status": status, "avgFillPrice": fill_price},
                )
            elif status in ("Submitted", "PreSubmitted", "PendingSubmit"):
                logger.info(
                    "IBKR order submitted: %s %d %s (id=%s)",
                    action, qty, request.ticker, order_id,
                )
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    status=OrderStatus.ACCEPTED,
                    message=f"Order submitted for {qty} {request.ticker}",
                    raw={"status": status},
                )
            else:
                reason = trade.orderStatus.whyHeld or status
                logger.warning(
                    "IBKR order failed: %s — %s", request.ticker, reason,
                )
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    message=f"Order {status}: {reason}",
                    raw={"status": status},
                )

        except Exception as e:
            logger.error("IBKR submit_order failed: %s", e)
            return OrderResult(success=False, message=str(e))

    async def cancel_order(self, order_id: str) -> OrderResult:
        try:
            def _do():
                for trade in self._ib.openTrades():
                    if str(trade.order.orderId) == order_id:
                        self._ib.cancelOrder(trade.order)
                        return True
                return False

            found = await self._async_call_ib(_do)
            if found:
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    status=OrderStatus.CANCELLED,
                    message="Order cancelled",
                )
            return OrderResult(
                success=False,
                message=f"Order {order_id} not found in open trades",
            )
        except Exception as e:
            return OrderResult(success=False, message=str(e))

    async def get_order(self, order_id: str) -> Optional[OrderInfo]:
        def _do():
            for trade in self._ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    return trade
            return None

        trade = await self._async_call_ib(_do)
        if trade is not None:
            return self._parse_trade(trade)
        return None

    async def get_open_orders(self) -> List[OrderInfo]:
        trades = await self._async_call_ib(lambda: self._ib.openTrades())
        result = []
        for trade in trades:
            info = self._parse_trade(trade)
            if info is not None:
                result.append(info)
        return result

    def _parse_trade(self, trade) -> OrderInfo:
        """Convert ib_insync Trade to OrderInfo."""
        contract = trade.contract
        order = trade.order

        if contract.secType == "OPT":
            right = "C" if contract.right == "C" else "P"
            ticker = build_option_ticker(
                contract.symbol, contract.lastTradeDateOrContractMonth, right,
                float(contract.strike),
            )
        else:
            ticker = contract.symbol

        side = OrderSide.BUY if order.action == "BUY" else OrderSide.SELL

        status_map = {
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "Inactive": OrderStatus.REJECTED,
            "Submitted": OrderStatus.ACCEPTED,
            "PreSubmitted": OrderStatus.PENDING_NEW,
        }
        status = status_map.get(trade.orderStatus.status, OrderStatus.NEW)

        return OrderInfo(
            order_id=str(order.orderId),
            ticker=ticker,
            side=side,
            order_type=(
                OrderType.LIMIT if order.orderType == "LMT" else OrderType.MARKET
            ),
            quantity=float(order.totalQuantity),
            filled_quantity=float(trade.orderStatus.filled),
            status=status,
            time_in_force=TimeInForce.DAY,
            limit_price=order.lmtPrice if order.orderType == "LMT" else None,
            filled_avg_price=trade.orderStatus.avgFillPrice or None,
            raw={"order": str(order), "status": trade.orderStatus.status},
        )

    async def close_position(self, ticker: str) -> OrderResult:
        pos = await self.get_position(ticker)
        if pos is None:
            return OrderResult(
                success=False, message=f"No open position for {ticker}"
            )

        close_side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
        req = OrderRequest(
            ticker=ticker,
            side=close_side,
            quantity=abs(pos.quantity),
        )
        return await self.submit_order(req)

    async def close_all_positions(self) -> List[OrderResult]:
        positions = await self.get_positions()
        results = []
        for pos in positions:
            result = await self.close_position(pos.ticker)
            results.append(result)
        return results
