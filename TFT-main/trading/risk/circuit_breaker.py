"""
Drawdown circuit breaker with configurable methods:
  - High Water Mark
  - Start of Day
  - Initial Capital

Redis holds hot state (sub-millisecond is_tripped checks).
PostgreSQL is the durable source of truth.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis

from trading.broker.base import BaseBroker, OrderResult
from trading.notifications.alerts import NotificationManager
from trading.persistence.audit import AuditLogger

logger = logging.getLogger(__name__)

# Redis key constants
_KEY_IS_TRIPPED = "circuit_breaker:is_tripped"
_KEY_STATE = "circuit_breaker:state"
_KEY_HWM = "circuit_breaker:high_water_mark"
_KEY_SOD = "circuit_breaker:start_of_day_value"

_SOD_TTL_SECONDS = 86400  # 24 hours


class DrawdownMethod(str, Enum):
    HIGH_WATER_MARK = "high_water_mark"
    START_OF_DAY = "start_of_day"
    INITIAL_CAPITAL = "initial_capital"


@dataclass
class DrawdownConfig:
    method: DrawdownMethod
    threshold_percent: float


@dataclass
class CircuitBreakerConfig:
    enabled: bool = True
    drawdown_configs: List[DrawdownConfig] = field(default_factory=list)
    check_interval_seconds: int = 30
    initial_capital: float = 100000.0

    @classmethod
    def from_env(cls) -> "CircuitBreakerConfig":
        enabled = os.getenv("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
        interval = int(os.getenv("CB_CHECK_INTERVAL_SECONDS", "30"))
        initial_capital = float(os.getenv("CB_INITIAL_CAPITAL", "100000.0"))

        configs: List[DrawdownConfig] = []
        raw = os.getenv("CB_DRAWDOWN_METHODS", "high_water_mark:5.0,start_of_day:3.0")
        for part in raw.split(","):
            part = part.strip()
            if ":" not in part:
                continue
            method_str, thresh_str = part.split(":", 1)
            try:
                method = DrawdownMethod(method_str.strip())
                threshold = float(thresh_str.strip())
                configs.append(
                    DrawdownConfig(method=method, threshold_percent=threshold)
                )
            except (ValueError, KeyError) as e:
                logger.warning("Skipping invalid drawdown config '%s': %s", part, e)

        if not configs:
            configs = [
                DrawdownConfig(DrawdownMethod.HIGH_WATER_MARK, 5.0),
                DrawdownConfig(DrawdownMethod.START_OF_DAY, 3.0),
            ]

        return cls(
            enabled=enabled,
            drawdown_configs=configs,
            check_interval_seconds=interval,
            initial_capital=initial_capital,
        )


@dataclass
class CircuitBreakerState:
    is_tripped: bool = False
    tripped_at: Optional[str] = None
    trip_reason: Optional[str] = None
    hwm: float = 0.0
    sod_value: float = 0.0
    last_check_time: Optional[str] = None
    last_portfolio_value: float = 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "is_tripped": self.is_tripped,
                "tripped_at": self.tripped_at,
                "trip_reason": self.trip_reason,
                "hwm": self.hwm,
                "sod_value": self.sod_value,
                "last_check_time": self.last_check_time,
                "last_portfolio_value": self.last_portfolio_value,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> "CircuitBreakerState":
        d = json.loads(raw)
        return cls(**d)


class CircuitBreaker:
    """
    Core safety mechanism. Monitors portfolio drawdown against configured
    thresholds and trips (closes all positions) when breached.
    """

    def __init__(
        self,
        config: CircuitBreakerConfig,
        broker: BaseBroker,
        redis_client: aioredis.Redis,
        notifier: NotificationManager,
        audit: AuditLogger,
    ):
        self.config = config
        self.broker = broker
        self.redis = redis_client
        self.notifier = notifier
        self.audit = audit
        self.state = CircuitBreakerState()
        self._monitor_task: Optional[asyncio.Task] = None
        self._consecutive_api_failures = 0
        self._MAX_API_FAILURES = 3

    async def start(self) -> None:
        """Load state from Redis (fall back to PostgreSQL), start monitor loop."""
        if not self.config.enabled:
            logger.info("Circuit breaker is disabled")
            return

        await self._load_state()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "Circuit breaker started (methods=%s, interval=%ds)",
            [
                f"{c.method.value}:{c.threshold_percent}%"
                for c in self.config.drawdown_configs
            ],
            self.config.check_interval_seconds,
        )

    async def stop(self) -> None:
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Circuit breaker stopped")

    async def _load_state(self) -> None:
        """Recover state: Redis first, then PostgreSQL."""
        raw = await self.redis.get(_KEY_STATE)
        if raw:
            self.state = CircuitBreakerState.from_json(raw)
            logger.info(
                "Loaded circuit breaker state from Redis (tripped=%s)",
                self.state.is_tripped,
            )
            return

        # Fall back to PostgreSQL
        logger.info("No Redis state found, recovering from PostgreSQL")
        latest_event = self.audit.get_latest_trip_event()
        latest_snapshot = self.audit.get_latest_snapshot()

        if latest_event:
            # Check if last trip was followed by a reset
            events = self.audit.get_recent_events(limit=2)
            if events and events[0]["event_type"] == "trip":
                self.state.is_tripped = True
                self.state.tripped_at = str(events[0]["created_at"])
                self.state.trip_reason = events[0].get("reason")

        if latest_snapshot:
            self.state.hwm = latest_snapshot.get("high_water_mark", 0) or 0
            self.state.last_portfolio_value = (
                latest_snapshot.get("portfolio_value", 0) or 0
            )

        if self.state.hwm == 0:
            self.state.hwm = self.config.initial_capital

        await self._save_state()

    async def _save_state(self) -> None:
        self.state.last_check_time = datetime.now(timezone.utc).isoformat()
        await self.redis.set(_KEY_STATE, self.state.to_json())
        await self.redis.set(
            _KEY_IS_TRIPPED, "true" if self.state.is_tripped else "false"
        )
        if self.state.hwm > 0:
            await self.redis.set(_KEY_HWM, str(self.state.hwm))

    async def check(self) -> bool:
        """
        Core check loop. Returns True if tripped (before or now).
        """
        if self.state.is_tripped:
            return True

        # Get current portfolio value from broker
        try:
            account = await self.broker.get_account()
            portfolio_value = account.portfolio_value
            self._consecutive_api_failures = 0
        except Exception as e:
            self._consecutive_api_failures += 1
            logger.error(
                "Broker API failure (%d/%d): %s",
                self._consecutive_api_failures,
                self._MAX_API_FAILURES,
                e,
            )
            if self._consecutive_api_failures >= self._MAX_API_FAILURES:
                await self._trip(
                    reason=f"Fail-safe: {self._MAX_API_FAILURES} consecutive broker API failures",
                    current_value=self.state.last_portfolio_value,
                    drawdown_pct=0.0,
                    drawdown_method="api_failure",
                )
                return True
            return False

        self.state.last_portfolio_value = portfolio_value

        # Update HWM
        if portfolio_value > self.state.hwm:
            self.state.hwm = portfolio_value
            await self.redis.set(_KEY_HWM, str(self.state.hwm))

        # Check each drawdown method
        for dc in self.config.drawdown_configs:
            drawdown_pct = self._calculate_drawdown(dc.method, portfolio_value)
            if drawdown_pct is not None and drawdown_pct >= dc.threshold_percent:
                await self._trip(
                    reason=(
                        f"{dc.method.value} drawdown {drawdown_pct:.2f}% "
                        f">= threshold {dc.threshold_percent}%"
                    ),
                    current_value=portfolio_value,
                    drawdown_pct=drawdown_pct,
                    drawdown_method=dc.method.value,
                )
                return True

        await self._save_state()
        return False

    def _calculate_drawdown(
        self, method: DrawdownMethod, current_value: float
    ) -> Optional[float]:
        if method == DrawdownMethod.HIGH_WATER_MARK:
            if self.state.hwm <= 0:
                return None
            return ((self.state.hwm - current_value) / self.state.hwm) * 100

        if method == DrawdownMethod.START_OF_DAY:
            if self.state.sod_value <= 0:
                return None
            return ((self.state.sod_value - current_value) / self.state.sod_value) * 100

        if method == DrawdownMethod.INITIAL_CAPITAL:
            if self.config.initial_capital <= 0:
                return None
            return (
                (self.config.initial_capital - current_value)
                / self.config.initial_capital
            ) * 100

        return None

    async def _trip(
        self,
        reason: str,
        current_value: float,
        drawdown_pct: float,
        drawdown_method: str,
    ) -> None:
        logger.critical("CIRCUIT BREAKER TRIPPING: %s", reason)

        # 1. SET tripped flag FIRST (blocks concurrent orders)
        self.state.is_tripped = True
        self.state.tripped_at = datetime.now(timezone.utc).isoformat()
        self.state.trip_reason = reason
        await self.redis.set(_KEY_IS_TRIPPED, "true")
        await self.redis.set(_KEY_STATE, self.state.to_json())

        # 2. Close all positions
        close_results: List[OrderResult] = []
        positions = await self.broker.get_positions()
        try:
            close_results = await self.broker.close_all_positions()
        except Exception as e:
            logger.error("Error during close_all_positions: %s", e)

        # 3. Log trip event to PostgreSQL
        event_id = self.audit.log_trip_event(
            reason=reason,
            drawdown_method=drawdown_method,
            drawdown_percent=drawdown_pct,
            portfolio_value=current_value,
            hwm=self.state.hwm,
            sod_value=self.state.sod_value,
            initial_capital=self.config.initial_capital,
            positions_closed=len(close_results),
        )

        # 4. Log individual closures
        for i, result in enumerate(close_results):
            pos = positions[i] if i < len(positions) else None
            self.audit.log_closure(
                event_id=event_id,
                ticker=pos.ticker if pos else "UNKNOWN",
                quantity=pos.quantity if pos else 0,
                side=pos.side if pos else "unknown",
                market_value=pos.market_value if pos else 0,
                unrealized_pnl=pos.unrealized_pnl if pos else 0,
                close_order_id=result.order_id,
                close_status=result.status.value if result.status else "unknown",
            )

        # 5. Log portfolio snapshot
        self.audit.log_portfolio_snapshot(current_value, self.state.hwm, "trip")

        # 6. Notify (fire-and-forget)
        try:
            await self.notifier.notify_circuit_breaker_trip(
                reason=reason,
                drawdown_percent=drawdown_pct,
                portfolio_value=current_value,
                positions_closed=len(close_results),
            )
        except Exception as e:
            logger.error("Notification failed during trip: %s", e)

        logger.critical(
            "Circuit breaker tripped: %s | closed %d positions | event_id=%d",
            reason,
            len(close_results),
            event_id,
        )

    async def is_tripped(self) -> bool:
        """O(1) Redis GET for pre-trade checks."""
        val = await self.redis.get(_KEY_IS_TRIPPED)
        return val == "true" or val == b"true"

    async def set_start_of_day_value(self, value: float) -> None:
        self.state.sod_value = value
        await self.redis.set(_KEY_SOD, str(value), ex=_SOD_TTL_SECONDS)
        await self._save_state()
        self.audit.log_portfolio_snapshot(value, self.state.hwm, "sod")
        logger.info("Start-of-day value set to $%.2f", value)

    async def update_high_water_mark(self, value: float) -> None:
        if value > self.state.hwm:
            self.state.hwm = value
            await self.redis.set(_KEY_HWM, str(value))
            await self._save_state()

    async def reset_breaker(self, operator: str, reason: str) -> None:
        """Manual re-enable. Requires operator identity and reason for audit."""
        logger.info("Circuit breaker reset by %s: %s", operator, reason)

        # Get current portfolio value for logging
        try:
            account = await self.broker.get_account()
            portfolio_value = account.portfolio_value
        except Exception:
            portfolio_value = self.state.last_portfolio_value

        self.state.is_tripped = False
        self.state.tripped_at = None
        self.state.trip_reason = None
        self._consecutive_api_failures = 0

        await self.redis.set(_KEY_IS_TRIPPED, "false")
        await self._save_state()

        self.audit.log_reset_event(operator, reason, portfolio_value)
        self.audit.log_portfolio_snapshot(portfolio_value, self.state.hwm, "reset")

        try:
            await self.notifier.notify_circuit_breaker_reset(
                operator=operator,
                reason=reason,
                portfolio_value=portfolio_value,
            )
        except Exception as e:
            logger.error("Notification failed during reset: %s", e)

    async def _monitor_loop(self) -> None:
        """Background task: check() every N seconds."""
        while True:
            try:
                await asyncio.sleep(self.config.check_interval_seconds)
                if self.state.is_tripped:
                    continue
                await self.check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitor loop error: %s", e)
                await asyncio.sleep(5)
