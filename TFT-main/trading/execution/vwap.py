"""
VWAP execution model — replaces immediate market orders with time-sliced
IOC limit orders weighted by intraday volume profile.

Each parent order is split into N slices (default 5) sent at fixed intervals
(default 60 s). Each slice is an IOC limit order whose size is proportional
to the historical intraday volume at that time-of-day.  Unfilled shares
carry forward to the next slice. A per-order 10 % ADV cap prevents
excessive market impact.

If all slices are exhausted and shares remain, a final market sweep order
is submitted.  On any unrecoverable error the model falls back to a single
market order through the underlying broker.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from trading.broker.base import (
    BaseBroker,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_NUM_SLICES = 5
DEFAULT_SLICE_INTERVAL_S = 60
DEFAULT_ADV_CAP_PCT = 0.10       # 10 % of average daily volume
DEFAULT_LIMIT_OFFSET_BPS = 5     # limit price offset from mid in bps
DEFAULT_VOLUME_CACHE_TTL_S = 86_400  # 1 day


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SliceResult:
    """Result of one time-slice within a VWAP execution."""
    slice_index: int
    requested_qty: float
    filled_qty: float
    filled_avg_price: Optional[float]
    order_id: Optional[str]
    status: str
    timestamp: str


@dataclass
class VWAPExecutionResult:
    """Aggregate result of a full VWAP execution."""
    ticker: str
    side: str
    total_requested: float
    total_filled: float
    filled_avg_price: float
    slices: List[SliceResult] = field(default_factory=list)
    expected_price: float = 0.0
    slippage_bps: float = 0.0
    used_fallback: bool = False
    adv_capped: bool = False
    elapsed_s: float = 0.0
    error: str = ""

    @property
    def fill_rate(self) -> float:
        if self.total_requested <= 0:
            return 0.0
        return self.total_filled / self.total_requested

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "side": self.side,
            "total_requested": self.total_requested,
            "total_filled": self.total_filled,
            "filled_avg_price": round(self.filled_avg_price, 4),
            "expected_price": round(self.expected_price, 4),
            "slippage_bps": round(self.slippage_bps, 2),
            "fill_rate": round(self.fill_rate, 4),
            "num_slices": len(self.slices),
            "used_fallback": self.used_fallback,
            "adv_capped": self.adv_capped,
            "elapsed_s": round(self.elapsed_s, 1),
            "error": self.error,
        }


# ── Volume profile ─────────────────────────────────────────────────────────────

class VolumeProfileCache:
    """
    Caches intraday volume distribution per ticker.

    The profile is a list of 13 floats representing the fraction of daily
    volume in each 30-minute bucket from 09:30 to 16:00 ET.  When no real
    data is available a U-shaped default profile is used (higher volume at
    open and close).
    """

    # U-shaped default: elevated at open/close, dip mid-day
    DEFAULT_PROFILE: List[float] = [
        0.12, 0.09, 0.08, 0.07, 0.06, 0.06, 0.06,
        0.06, 0.06, 0.07, 0.08, 0.09, 0.10,
    ]

    def __init__(self, ttl_s: int = DEFAULT_VOLUME_CACHE_TTL_S):
        self._cache: Dict[str, tuple] = {}  # ticker -> (profile, expire_ts)
        self._ttl_s = ttl_s

    def get(self, ticker: str) -> Optional[List[float]]:
        entry = self._cache.get(ticker)
        if entry is None:
            return None
        profile, expire_ts = entry
        if time.time() > expire_ts:
            del self._cache[ticker]
            return None
        return profile

    def put(self, ticker: str, profile: List[float]) -> None:
        self._cache[ticker] = (profile, time.time() + self._ttl_s)

    def get_or_default(self, ticker: str) -> List[float]:
        profile = self.get(ticker)
        if profile is not None:
            return profile
        return list(self.DEFAULT_PROFILE)

    def get_slice_weights(self, ticker: str, num_slices: int) -> List[float]:
        """
        Map the 13-bucket intraday profile into ``num_slices`` execution
        weights.  Each weight is proportional to the summed volume in
        the profile buckets covered by that slice.
        """
        profile = self.get_or_default(ticker)
        n_buckets = len(profile)

        if num_slices >= n_buckets:
            # One bucket per slice, with leftover assigned to last slice
            weights = profile[:num_slices]
            while len(weights) < num_slices:
                weights.append(weights[-1] if weights else 1.0 / num_slices)
        else:
            # Aggregate buckets into slices
            buckets_per_slice = n_buckets / num_slices
            weights = []
            for i in range(num_slices):
                start = int(i * buckets_per_slice)
                end = int((i + 1) * buckets_per_slice)
                weights.append(sum(profile[start:end]))

        # Normalize
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / num_slices] * num_slices
        return weights


# ── VWAP Execution Model ──────────────────────────────────────────────────────

class VWAPExecutionModel:
    """
    Wraps a ``BaseBroker`` and executes orders using time-sliced VWAP.

    Usage::

        vwap = VWAPExecutionModel(broker)
        result = await vwap.execute(
            ticker="AAPL", side=OrderSide.BUY, quantity=200,
            current_price=185.50, adv=5_000_000,
        )
    """

    def __init__(
        self,
        broker: BaseBroker,
        num_slices: int = DEFAULT_NUM_SLICES,
        slice_interval_s: int = DEFAULT_SLICE_INTERVAL_S,
        adv_cap_pct: float = DEFAULT_ADV_CAP_PCT,
        limit_offset_bps: int = DEFAULT_LIMIT_OFFSET_BPS,
        volume_cache: Optional[VolumeProfileCache] = None,
    ):
        self.broker = broker
        self.num_slices = max(1, num_slices)
        self.slice_interval_s = max(1, slice_interval_s)
        self.adv_cap_pct = adv_cap_pct
        self.limit_offset_bps = limit_offset_bps
        self.volume_cache = volume_cache or VolumeProfileCache()
        self._execution_history: List[VWAPExecutionResult] = []

    async def execute(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float,
        current_price: float,
        adv: float = 0,
    ) -> VWAPExecutionResult:
        """
        Execute an order using VWAP slicing.

        Args:
            ticker: Symbol to trade.
            side: BUY or SELL.
            quantity: Total shares requested.
            current_price: Latest market price (used for limit pricing).
            adv: Average daily volume.  If > 0, the order is capped at
                 ``adv * adv_cap_pct``.

        Returns:
            VWAPExecutionResult with per-slice detail and slippage.
        """
        start = time.monotonic()
        result = VWAPExecutionResult(
            ticker=ticker,
            side=side.value,
            total_requested=quantity,
            total_filled=0.0,
            filled_avg_price=0.0,
            expected_price=current_price,
        )

        # ADV cap
        if adv > 0:
            max_qty = int(adv * self.adv_cap_pct)
            if max_qty < 1:
                max_qty = 1
            if quantity > max_qty:
                logger.warning(
                    "VWAP %s %s: capping %d → %d (10%% of ADV %d)",
                    side.value, ticker, int(quantity), max_qty, int(adv),
                )
                quantity = max_qty
                result.total_requested = quantity
                result.adv_capped = True

        if quantity < 1:
            result.elapsed_s = time.monotonic() - start
            return result

        try:
            await self._execute_slices(ticker, side, quantity, current_price, result)
        except Exception as e:
            logger.error("VWAP execution error for %s: %s — falling back to market", ticker, e)
            result.error = str(e)
            await self._fallback_market(ticker, side, quantity, result)

        result.elapsed_s = time.monotonic() - start
        self._execution_history.append(result)
        return result

    async def _execute_slices(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float,
        current_price: float,
        result: VWAPExecutionResult,
    ) -> None:
        weights = self.volume_cache.get_slice_weights(ticker, self.num_slices)
        remaining = quantity
        total_cost = 0.0

        for i in range(self.num_slices):
            if remaining < 1:
                break

            # Slice size: proportional to volume weight, at least 1 share
            slice_qty = max(1, int(round(quantity * weights[i])))
            # On last slice, send everything remaining
            if i == self.num_slices - 1:
                slice_qty = int(remaining)
            else:
                slice_qty = min(slice_qty, int(remaining))

            if slice_qty < 1:
                continue

            # Limit price: slightly better than current price to increase fill
            offset = current_price * (self.limit_offset_bps / 10_000)
            if side == OrderSide.BUY:
                limit_price = round(current_price + offset, 2)
            else:
                limit_price = round(current_price - offset, 2)

            order_req = OrderRequest(
                ticker=ticker,
                side=side,
                quantity=slice_qty,
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.IOC,
                limit_price=limit_price,
            )

            order_result = await self.broker.submit_order(order_req)

            # Determine fill
            filled_qty = 0.0
            filled_price = None
            status_str = "unknown"

            if order_result and order_result.success and order_result.order_id:
                # Wait briefly for IOC fill confirmation
                await asyncio.sleep(0.5)
                order_info = await self.broker.get_order(order_result.order_id)
                if order_info is not None:
                    filled_qty = order_info.filled_quantity
                    filled_price = order_info.filled_avg_price
                    status_str = order_info.status.value if order_info.status else "unknown"
                else:
                    status_str = "no_info"
            elif order_result:
                status_str = order_result.status.value if order_result.status else "failed"

            slice_result = SliceResult(
                slice_index=i,
                requested_qty=slice_qty,
                filled_qty=filled_qty,
                filled_avg_price=filled_price,
                order_id=order_result.order_id if order_result else None,
                status=status_str,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            result.slices.append(slice_result)

            if filled_qty > 0 and filled_price is not None:
                total_cost += filled_qty * filled_price
                remaining -= filled_qty

            logger.info(
                "VWAP %s %s slice %d/%d: req=%d filled=%d @ %s (remaining=%d)",
                side.value, ticker, i + 1, self.num_slices,
                slice_qty, int(filled_qty),
                f"${filled_price:.2f}" if filled_price else "N/A",
                int(remaining),
            )

            # Wait before next slice (skip wait on last slice)
            if i < self.num_slices - 1 and remaining >= 1:
                await asyncio.sleep(self.slice_interval_s)

        # Sweep: if shares remain after all slices, send market order
        if remaining >= 1:
            logger.info("VWAP %s %s: sweeping %d remaining shares via market order", side.value, ticker, int(remaining))
            sweep = await self.broker.submit_order(OrderRequest(
                ticker=ticker, side=side, quantity=int(remaining),
            ))
            sweep_filled = 0.0
            sweep_price = None
            if sweep and sweep.success and sweep.order_id:
                await asyncio.sleep(1)
                info = await self.broker.get_order(sweep.order_id)
                if info is not None:
                    sweep_filled = info.filled_quantity
                    sweep_price = info.filled_avg_price
            result.slices.append(SliceResult(
                slice_index=self.num_slices,
                requested_qty=int(remaining),
                filled_qty=sweep_filled,
                filled_avg_price=sweep_price,
                order_id=sweep.order_id if sweep else None,
                status="sweep",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            if sweep_filled > 0 and sweep_price is not None:
                total_cost += sweep_filled * sweep_price
                remaining -= sweep_filled

        # Aggregate
        result.total_filled = quantity - remaining
        if result.total_filled > 0:
            result.filled_avg_price = total_cost / result.total_filled
        else:
            result.filled_avg_price = 0.0

        # Slippage in bps
        if result.expected_price > 0 and result.total_filled > 0:
            if side == OrderSide.BUY:
                result.slippage_bps = (
                    (result.filled_avg_price - result.expected_price)
                    / result.expected_price * 10_000
                )
            else:
                result.slippage_bps = (
                    (result.expected_price - result.filled_avg_price)
                    / result.expected_price * 10_000
                )

    async def _fallback_market(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float,
        result: VWAPExecutionResult,
    ) -> None:
        """Fall back to a single market order."""
        result.used_fallback = True
        try:
            order_result = await self.broker.submit_order(OrderRequest(
                ticker=ticker, side=side, quantity=int(quantity),
            ))
            filled_qty = 0.0
            filled_price = None
            if order_result and order_result.success and order_result.order_id:
                await asyncio.sleep(1)
                info = await self.broker.get_order(order_result.order_id)
                if info is not None:
                    filled_qty = info.filled_quantity
                    filled_price = info.filled_avg_price

            result.total_filled = filled_qty
            result.filled_avg_price = filled_price or 0.0
            result.slices.append(SliceResult(
                slice_index=0,
                requested_qty=int(quantity),
                filled_qty=filled_qty,
                filled_avg_price=filled_price,
                order_id=order_result.order_id if order_result else None,
                status="fallback_market",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
        except Exception as fallback_err:
            logger.error("VWAP fallback market order also failed for %s: %s", ticker, fallback_err)
            result.error += f"; fallback also failed: {fallback_err}"

    def get_execution_history(self, n: int = 50) -> List[VWAPExecutionResult]:
        return self._execution_history[-n:]

    def get_execution_stats(self) -> dict:
        """Aggregate statistics over recent executions."""
        history = self._execution_history
        if not history:
            return {"total_executions": 0}

        slippages = [r.slippage_bps for r in history if r.total_filled > 0]
        fill_rates = [r.fill_rate for r in history]
        fallbacks = sum(1 for r in history if r.used_fallback)

        return {
            "total_executions": len(history),
            "avg_slippage_bps": round(sum(slippages) / len(slippages), 2) if slippages else 0.0,
            "max_slippage_bps": round(max(slippages), 2) if slippages else 0.0,
            "avg_fill_rate": round(sum(fill_rates) / len(fill_rates), 4) if fill_rates else 0.0,
            "fallback_count": fallbacks,
            "adv_capped_count": sum(1 for r in history if r.adv_capped),
        }
