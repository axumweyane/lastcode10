"""
Dead Letter Queue with PostgreSQL persistence and exponential backoff retry.

Usage:
    dlq = DeadLetterQueue(db_url="postgresql://...", service_name="sentiment-engine")
    dlq.persist(topic="market-data", key="AAPL", value={"price": 100}, error="timeout")
    due = dlq.retry(processor_fn)
"""

import enum
import json
import logging
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# Backoff configuration
BASE_DELAY_S = 1.0
MULTIPLIER = 2
MAX_RETRIES = 5
MAX_DELAY_S = 60.0
JITTER_FRACTION = 0.25  # 0-25% random jitter

# Background retry interval
RETRY_POLL_INTERVAL_S = 30


class DLQStatus(str, enum.Enum):
    PENDING = "PENDING"
    RETRYING = "RETRYING"
    EXHAUSTED = "EXHAUSTED"
    RESOLVED = "RESOLVED"


CREATE_DLQ_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dead_letter_queue (
    id              SERIAL PRIMARY KEY,
    service_name    VARCHAR(128) NOT NULL,
    topic           VARCHAR(256) NOT NULL,
    message_key     TEXT,
    message_value   JSONB NOT NULL DEFAULT '{}',
    error           TEXT NOT NULL DEFAULT '',
    retry_count     INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 5,
    next_retry_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status          VARCHAR(16) NOT NULL DEFAULT 'PENDING'
);

CREATE INDEX IF NOT EXISTS idx_dlq_status_next_retry
    ON dead_letter_queue(status, next_retry_at);
CREATE INDEX IF NOT EXISTS idx_dlq_service
    ON dead_letter_queue(service_name);
CREATE INDEX IF NOT EXISTS idx_dlq_created
    ON dead_letter_queue(created_at);
"""


def compute_backoff(retry_count: int) -> float:
    """Compute delay with exponential backoff and jitter.

    Formula: min(base * 2^retry_count + jitter, max_delay)
    Jitter: random 0-25% of the computed delay.
    """
    delay = BASE_DELAY_S * (MULTIPLIER**retry_count)
    jitter = random.uniform(0, JITTER_FRACTION * delay)
    return min(delay + jitter, MAX_DELAY_S)


class DeadLetterQueue:
    """PostgreSQL-backed dead letter queue with retry support."""

    def __init__(
        self,
        db_url: str,
        service_name: str,
        max_retries: int = MAX_RETRIES,
        on_exhausted: Optional[Callable[[Dict], None]] = None,
    ):
        self.db_url = db_url
        self.service_name = service_name
        self.max_retries = max_retries
        self._on_exhausted = on_exhausted
        self._ensure_table()

    def _get_conn(self):
        return psycopg2.connect(self.db_url)

    def _ensure_table(self):
        try:
            conn = self._get_conn()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(CREATE_DLQ_TABLE_SQL)
            conn.close()
        except Exception as e:
            logger.warning("DLQ table creation skipped (may already exist): %s", e)

    def persist(
        self,
        topic: str,
        key: Optional[str],
        value: Any,
        error: str,
    ) -> int:
        """Persist a failed message to the dead letter queue.

        Returns the DLQ row id.
        """
        if not isinstance(value, str):
            value = json.dumps(value, default=str)

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO dead_letter_queue
                        (service_name, topic, message_key, message_value, error,
                         retry_count, max_retries, next_retry_at, status)
                    VALUES (%s, %s, %s, %s::jsonb, %s, 0, %s, NOW(), %s)
                    RETURNING id
                    """,
                    (
                        self.service_name,
                        topic,
                        key,
                        value,
                        str(error),
                        self.max_retries,
                        DLQStatus.PENDING.value,
                    ),
                )
                row_id = cur.fetchone()[0]
            conn.commit()
            logger.info(
                "DLQ persisted: id=%d service=%s topic=%s key=%s",
                row_id,
                self.service_name,
                topic,
                key,
            )
            return row_id
        finally:
            conn.close()

    def retry(
        self,
        processor: Callable[[str, Optional[str], Any], None],
    ) -> int:
        """Fetch due messages and retry them.

        Args:
            processor: callable(topic, key, value) that re-processes the message.
                       Raise an exception to indicate failure.

        Returns:
            Number of messages successfully retried.
        """
        conn = self._get_conn()
        success_count = 0
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, topic, message_key, message_value, retry_count, max_retries
                    FROM dead_letter_queue
                    WHERE service_name = %s
                      AND status IN (%s, %s)
                      AND next_retry_at <= NOW()
                    ORDER BY next_retry_at
                    LIMIT 50
                    FOR UPDATE SKIP LOCKED
                    """,
                    (
                        self.service_name,
                        DLQStatus.PENDING.value,
                        DLQStatus.RETRYING.value,
                    ),
                )
                rows = cur.fetchall()

            for row in rows:
                row_id = row["id"]
                retry_count = row["retry_count"] + 1
                try:
                    processor(row["topic"], row["message_key"], row["message_value"])
                    # Success — mark resolved
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE dead_letter_queue
                            SET status = %s, retry_count = %s, updated_at = NOW()
                            WHERE id = %s
                            """,
                            (DLQStatus.RESOLVED.value, retry_count, row_id),
                        )
                    conn.commit()
                    success_count += 1
                    logger.info(
                        "DLQ retry succeeded: id=%d (attempt %d)", row_id, retry_count
                    )

                except Exception as e:
                    if retry_count >= row["max_retries"]:
                        # Exhausted
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                UPDATE dead_letter_queue
                                SET status = %s, retry_count = %s, error = %s, updated_at = NOW()
                                WHERE id = %s
                                """,
                                (
                                    DLQStatus.EXHAUSTED.value,
                                    retry_count,
                                    str(e),
                                    row_id,
                                ),
                            )
                        conn.commit()
                        logger.error(
                            "DLQ exhausted: id=%d after %d retries — %s",
                            row_id,
                            retry_count,
                            e,
                        )
                        if self._on_exhausted:
                            try:
                                self._on_exhausted(dict(row))
                            except Exception:
                                pass
                    else:
                        # Schedule next retry
                        delay = compute_backoff(retry_count)
                        next_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                UPDATE dead_letter_queue
                                SET status = %s, retry_count = %s, error = %s,
                                    next_retry_at = %s, updated_at = NOW()
                                WHERE id = %s
                                """,
                                (
                                    DLQStatus.RETRYING.value,
                                    retry_count,
                                    str(e),
                                    next_at,
                                    row_id,
                                ),
                            )
                        conn.commit()
                        logger.warning(
                            "DLQ retry failed: id=%d attempt=%d next_retry=%.1fs — %s",
                            row_id,
                            retry_count,
                            delay,
                            e,
                        )

            return success_count
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Return DLQ statistics for the dashboard."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Counts by status
                cur.execute(
                    """
                    SELECT status, COUNT(*) as count
                    FROM dead_letter_queue
                    WHERE service_name = %s
                    GROUP BY status
                    """,
                    (self.service_name,),
                )
                by_status = {row["status"]: row["count"] for row in cur.fetchall()}

                # Recent failures (last 20)
                cur.execute(
                    """
                    SELECT id, topic, message_key, error, retry_count, status,
                           created_at, updated_at
                    FROM dead_letter_queue
                    WHERE service_name = %s
                      AND status != %s
                    ORDER BY updated_at DESC
                    LIMIT 20
                    """,
                    (self.service_name, DLQStatus.RESOLVED.value),
                )
                recent = [dict(r) for r in cur.fetchall()]
                for r in recent:
                    for k in ("created_at", "updated_at"):
                        if r.get(k):
                            r[k] = r[k].isoformat()

                # Retry success rate
                cur.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE status = %s) AS resolved,
                        COUNT(*) FILTER (WHERE status = %s) AS exhausted,
                        COUNT(*) AS total
                    FROM dead_letter_queue
                    WHERE service_name = %s
                      AND status IN (%s, %s)
                    """,
                    (
                        DLQStatus.RESOLVED.value,
                        DLQStatus.EXHAUSTED.value,
                        self.service_name,
                        DLQStatus.RESOLVED.value,
                        DLQStatus.EXHAUSTED.value,
                    ),
                )
                rate_row = cur.fetchone()
                total = rate_row["total"] or 0
                resolved = rate_row["resolved"] or 0
                success_rate = (resolved / total * 100) if total > 0 else 0.0

            return {
                "service_name": self.service_name,
                "by_status": {
                    DLQStatus.PENDING.value: by_status.get(DLQStatus.PENDING.value, 0),
                    DLQStatus.RETRYING.value: by_status.get(
                        DLQStatus.RETRYING.value, 0
                    ),
                    DLQStatus.EXHAUSTED.value: by_status.get(
                        DLQStatus.EXHAUSTED.value, 0
                    ),
                    DLQStatus.RESOLVED.value: by_status.get(
                        DLQStatus.RESOLVED.value, 0
                    ),
                },
                "recent_failures": recent,
                "retry_success_rate_pct": round(success_rate, 1),
            }
        finally:
            conn.close()

    def get_all_stats(self) -> Dict[str, Any]:
        """Return DLQ statistics across ALL services."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT status, COUNT(*) as count
                    FROM dead_letter_queue
                    GROUP BY status
                    """,
                )
                by_status = {row["status"]: row["count"] for row in cur.fetchall()}

                cur.execute(
                    """
                    SELECT id, service_name, topic, message_key, error,
                           retry_count, status, created_at, updated_at
                    FROM dead_letter_queue
                    WHERE status != %s
                    ORDER BY updated_at DESC
                    LIMIT 20
                    """,
                    (DLQStatus.RESOLVED.value,),
                )
                recent = [dict(r) for r in cur.fetchall()]
                for r in recent:
                    for k in ("created_at", "updated_at"):
                        if r.get(k):
                            r[k] = r[k].isoformat()

                cur.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE status = %s) AS resolved,
                        COUNT(*) FILTER (WHERE status = %s) AS exhausted,
                        COUNT(*) AS total
                    FROM dead_letter_queue
                    WHERE status IN (%s, %s)
                    """,
                    (
                        DLQStatus.RESOLVED.value,
                        DLQStatus.EXHAUSTED.value,
                        DLQStatus.RESOLVED.value,
                        DLQStatus.EXHAUSTED.value,
                    ),
                )
                rate_row = cur.fetchone()
                total = rate_row["total"] or 0
                resolved = rate_row["resolved"] or 0
                success_rate = (resolved / total * 100) if total > 0 else 0.0

            return {
                "by_status": {
                    DLQStatus.PENDING.value: by_status.get(DLQStatus.PENDING.value, 0),
                    DLQStatus.RETRYING.value: by_status.get(
                        DLQStatus.RETRYING.value, 0
                    ),
                    DLQStatus.EXHAUSTED.value: by_status.get(
                        DLQStatus.EXHAUSTED.value, 0
                    ),
                    DLQStatus.RESOLVED.value: by_status.get(
                        DLQStatus.RESOLVED.value, 0
                    ),
                },
                "recent_failures": recent,
                "retry_success_rate_pct": round(success_rate, 1),
            }
        finally:
            conn.close()


def start_retry_worker(
    dlq: DeadLetterQueue,
    processor: Callable[[str, Optional[str], Any], None],
    interval_s: float = RETRY_POLL_INTERVAL_S,
) -> threading.Thread:
    """Start a background thread that retries DLQ messages every `interval_s` seconds."""

    def _loop():
        while True:
            try:
                count = dlq.retry(processor)
                if count > 0:
                    logger.info("DLQ retry worker: %d messages retried", count)
            except Exception as e:
                logger.error("DLQ retry worker error: %s", e)
            time.sleep(interval_s)

    t = threading.Thread(
        target=_loop, name=f"dlq-retry-{dlq.service_name}", daemon=True
    )
    t.start()
    return t
