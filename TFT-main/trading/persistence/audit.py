"""
PostgreSQL audit tables and logging for circuit breaker events.
Mirrors the psycopg2 context-manager pattern from postgres_data_loader.py.
"""

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import Json

logger = logging.getLogger(__name__)

CIRCUIT_BREAKER_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    id              SERIAL PRIMARY KEY,
    event_type      VARCHAR(32) NOT NULL,   -- 'trip' or 'reset'
    triggered_by    VARCHAR(128),
    reason          TEXT,
    drawdown_method VARCHAR(32),
    drawdown_percent DOUBLE PRECISION,
    portfolio_value  DOUBLE PRECISION,
    hwm             DOUBLE PRECISION,
    sod_value       DOUBLE PRECISION,
    initial_capital DOUBLE PRECISION,
    positions_closed INTEGER DEFAULT 0,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS circuit_breaker_closures (
    id              SERIAL PRIMARY KEY,
    event_id        INTEGER NOT NULL REFERENCES circuit_breaker_events(id),
    ticker          VARCHAR(16) NOT NULL,
    quantity        DOUBLE PRECISION,
    side            VARCHAR(8),
    market_value    DOUBLE PRECISION,
    unrealized_pnl  DOUBLE PRECISION,
    close_order_id  VARCHAR(64),
    close_status    VARCHAR(32),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id              SERIAL PRIMARY KEY,
    portfolio_value DOUBLE PRECISION NOT NULL,
    high_water_mark DOUBLE PRECISION,
    snapshot_type   VARCHAR(32) NOT NULL,   -- 'periodic', 'trip', 'reset', 'sod'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cb_events_type ON circuit_breaker_events(event_type);
CREATE INDEX IF NOT EXISTS idx_cb_events_created ON circuit_breaker_events(created_at);
CREATE INDEX IF NOT EXISTS idx_cb_closures_event ON circuit_breaker_closures(event_id);
CREATE INDEX IF NOT EXISTS idx_cb_closures_ticker ON circuit_breaker_closures(ticker);
CREATE INDEX IF NOT EXISTS idx_portfolio_snap_type ON portfolio_snapshots(snapshot_type);
"""


class AuditLogger:
    """Persistent audit trail for circuit breaker events."""

    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        if db_config:
            self.db_config = db_config
        else:
            self.db_config = {
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "database": os.getenv("DB_NAME", "stock_trading_analysis"),
                "user": os.getenv("DB_USER", "postgres"),
                "password": os.getenv("DB_PASSWORD", ""),
            }

    @contextmanager
    def _get_connection(self):
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.db_config["host"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                port=self.db_config["port"],
            )
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()

    def create_schema(self) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CIRCUIT_BREAKER_SCHEMA_SQL)
            conn.commit()
        logger.info("Circuit breaker audit schema created/verified")

    def log_trip_event(
        self,
        reason: str,
        drawdown_method: str,
        drawdown_percent: float,
        portfolio_value: float,
        hwm: float,
        sod_value: float,
        initial_capital: float,
        positions_closed: int,
        triggered_by: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO circuit_breaker_events
                        (event_type, triggered_by, reason, drawdown_method,
                         drawdown_percent, portfolio_value, hwm, sod_value,
                         initial_capital, positions_closed, metadata)
                    VALUES ('trip', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        triggered_by,
                        reason,
                        drawdown_method,
                        drawdown_percent,
                        portfolio_value,
                        hwm,
                        sod_value,
                        initial_capital,
                        positions_closed,
                        Json(metadata or {}),
                    ),
                )
                event_id = cur.fetchone()[0]
            conn.commit()
        logger.info("Logged trip event id=%d reason=%s", event_id, reason)
        return event_id

    def log_closure(
        self,
        event_id: int,
        ticker: str,
        quantity: float,
        side: str,
        market_value: float,
        unrealized_pnl: float,
        close_order_id: Optional[str] = None,
        close_status: Optional[str] = None,
    ) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO circuit_breaker_closures
                        (event_id, ticker, quantity, side, market_value,
                         unrealized_pnl, close_order_id, close_status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        event_id,
                        ticker,
                        quantity,
                        side,
                        market_value,
                        unrealized_pnl,
                        close_order_id,
                        close_status,
                    ),
                )
            conn.commit()

    def log_reset_event(
        self,
        operator: str,
        reason: str,
        portfolio_value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO circuit_breaker_events
                        (event_type, triggered_by, reason, portfolio_value, metadata)
                    VALUES ('reset', %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (operator, reason, portfolio_value, Json(metadata or {})),
                )
                event_id = cur.fetchone()[0]
            conn.commit()
        logger.info("Logged reset event id=%d by=%s", event_id, operator)
        return event_id

    def log_portfolio_snapshot(
        self,
        portfolio_value: float,
        hwm: float,
        snapshot_type: str,
    ) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO portfolio_snapshots
                        (portfolio_value, high_water_mark, snapshot_type)
                    VALUES (%s, %s, %s)
                    """,
                    (portfolio_value, hwm, snapshot_type),
                )
            conn.commit()

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, event_type, triggered_by, reason, drawdown_method,
                           drawdown_percent, portfolio_value, hwm, sod_value,
                           initial_capital, positions_closed, metadata, created_at
                    FROM circuit_breaker_events
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                cols = [desc[0] for desc in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_closures_for_event(self, event_id: int) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, event_id, ticker, quantity, side, market_value,
                           unrealized_pnl, close_order_id, close_status, created_at
                    FROM circuit_breaker_closures
                    WHERE event_id = %s
                    ORDER BY created_at
                    """,
                    (event_id,),
                )
                cols = [desc[0] for desc in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT portfolio_value, high_water_mark, snapshot_type, created_at
                    FROM portfolio_snapshots
                    ORDER BY created_at DESC
                    LIMIT 1
                    """)
                row = cur.fetchone()
                if row:
                    cols = [desc[0] for desc in cur.description]
                    return dict(zip(cols, row))
                return None

    def get_latest_trip_event(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, event_type, triggered_by, reason, drawdown_method,
                           drawdown_percent, portfolio_value, hwm, sod_value,
                           initial_capital, positions_closed, metadata, created_at
                    FROM circuit_breaker_events
                    WHERE event_type = 'trip'
                    ORDER BY created_at DESC
                    LIMIT 1
                    """)
                row = cur.fetchone()
                if row:
                    cols = [desc[0] for desc in cur.description]
                    return dict(zip(cols, row))
                return None
