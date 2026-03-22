"""Tests for dead letter queue: persistence, backoff, retry, status transitions."""

import json
import os
import sys
import time
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.common.dlq import (
    DeadLetterQueue,
    DLQStatus,
    compute_backoff,
    start_retry_worker,
    BASE_DELAY_S,
    MAX_DELAY_S,
    MAX_RETRIES,
    JITTER_FRACTION,
    CREATE_DLQ_TABLE_SQL,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_conn():
    """Create a mock psycopg2 connection with cursor context manager."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    return conn, cursor


# ── 1. Exponential backoff computation ───────────────────────────────────────


class TestComputeBackoff:
    """Test the exponential backoff formula."""

    def test_base_delay_at_retry_0(self):
        """First retry should be ~1s (base delay + jitter)."""
        delays = [compute_backoff(0) for _ in range(100)]
        for d in delays:
            assert BASE_DELAY_S <= d <= BASE_DELAY_S * (1 + JITTER_FRACTION)

    def test_exponential_sequence(self):
        """Verify the 1s, 2s, 4s, 8s, 16s exponential pattern (ignoring jitter)."""
        expected_bases = [1, 2, 4, 8, 16]
        for i, expected in enumerate(expected_bases):
            delays = [compute_backoff(i) for _ in range(50)]
            for d in delays:
                # Base is expected, max jitter is 25% of base
                assert expected <= d <= expected * (1 + JITTER_FRACTION)

    def test_max_delay_cap(self):
        """Delays should never exceed MAX_DELAY_S (60s)."""
        for retry in range(20):
            for _ in range(50):
                assert compute_backoff(retry) <= MAX_DELAY_S

    def test_jitter_within_range(self):
        """Jitter should be between 0 and 25% of the base delay."""
        for retry in range(5):
            base = BASE_DELAY_S * (2**retry)
            if base >= MAX_DELAY_S:
                continue
            delays = [compute_backoff(retry) for _ in range(200)]
            min_expected = base
            max_expected = min(base * (1 + JITTER_FRACTION), MAX_DELAY_S)
            for d in delays:
                assert (
                    min_expected <= d <= max_expected
                ), f"retry={retry}: delay {d} outside [{min_expected}, {max_expected}]"

    def test_jitter_has_variance(self):
        """Jitter should add actual randomness — not all values identical."""
        delays = [compute_backoff(2) for _ in range(100)]
        unique = set(round(d, 6) for d in delays)
        assert len(unique) > 1, "Jitter should produce varying delays"


# ── 2. DLQ Status enum ──────────────────────────────────────────────────────


class TestDLQStatus:
    def test_all_statuses(self):
        assert DLQStatus.PENDING.value == "PENDING"
        assert DLQStatus.RETRYING.value == "RETRYING"
        assert DLQStatus.EXHAUSTED.value == "EXHAUSTED"
        assert DLQStatus.RESOLVED.value == "RESOLVED"

    def test_status_is_string(self):
        for s in DLQStatus:
            assert isinstance(s.value, str)


# ── 3. Message persistence ──────────────────────────────────────────────────


class TestDLQPersist:
    """Test that persist() saves messages correctly."""

    def test_persist_returns_id(self):
        with patch("services.common.dlq.psycopg2") as mock_pg:
            conn, cursor = _mock_conn()
            mock_pg.connect.return_value = conn
            cursor.fetchone.return_value = (42,)

            dlq = DeadLetterQueue.__new__(DeadLetterQueue)
            dlq.db_url = "postgresql://test"
            dlq.service_name = "test-service"
            dlq.max_retries = 5
            dlq._ensure_table = MagicMock()

            row_id = dlq.persist(
                topic="market-data",
                key="AAPL",
                value={"price": 150.0},
                error="connection timeout",
            )
            assert row_id == 42

    def test_persist_serializes_dict_to_json(self):
        with patch("services.common.dlq.psycopg2") as mock_pg:
            conn, cursor = _mock_conn()
            mock_pg.connect.return_value = conn
            cursor.fetchone.return_value = (1,)

            dlq = DeadLetterQueue.__new__(DeadLetterQueue)
            dlq.db_url = "postgresql://test"
            dlq.service_name = "test-service"
            dlq.max_retries = 5
            dlq._ensure_table = MagicMock()

            dlq.persist(topic="t", key="k", value={"x": 1}, error="err")

            # Check that execute was called with JSON-serialized value
            execute_args = cursor.execute.call_args[0]
            params = execute_args[1]
            assert json.loads(params[3]) == {"x": 1}

    def test_persist_stores_error_string(self):
        with patch("services.common.dlq.psycopg2") as mock_pg:
            conn, cursor = _mock_conn()
            mock_pg.connect.return_value = conn
            cursor.fetchone.return_value = (1,)

            dlq = DeadLetterQueue.__new__(DeadLetterQueue)
            dlq.db_url = "postgresql://test"
            dlq.service_name = "test-service"
            dlq.max_retries = 5
            dlq._ensure_table = MagicMock()

            dlq.persist(topic="t", key=None, value={}, error="TimeoutError: 15s")

            params = cursor.execute.call_args[0][1]
            assert "TimeoutError: 15s" in params[4]


# ── 4. Retry logic and status transitions ────────────────────────────────────


class TestDLQRetry:
    """Test the retry() method and status transitions."""

    def _make_dlq(self, mock_pg):
        conn, cursor = _mock_conn()
        mock_pg.connect.return_value = conn
        mock_pg.extras = MagicMock()
        mock_pg.extras.RealDictCursor = "RealDictCursor"

        dlq = DeadLetterQueue.__new__(DeadLetterQueue)
        dlq.db_url = "postgresql://test"
        dlq.service_name = "test-service"
        dlq.max_retries = 5
        dlq._on_exhausted = None
        dlq._ensure_table = MagicMock()
        return dlq, conn, cursor

    def test_retry_calls_processor_and_resolves(self):
        """Successful retry should mark message as RESOLVED."""
        with patch("services.common.dlq.psycopg2") as mock_pg:
            dlq, conn, cursor = self._make_dlq(mock_pg)

            # Return one due message
            cursor.fetchall.return_value = [
                {
                    "id": 1,
                    "topic": "t",
                    "message_key": "k",
                    "message_value": {"x": 1},
                    "retry_count": 0,
                    "max_retries": 5,
                }
            ]

            processor = MagicMock()
            count = dlq.retry(processor)

            assert count == 1
            processor.assert_called_once_with("t", "k", {"x": 1})

            # Check RESOLVED status was written
            update_calls = [
                c
                for c in cursor.execute.call_args_list
                if c[0][0].strip().startswith("UPDATE")
            ]
            assert len(update_calls) == 1
            assert DLQStatus.RESOLVED.value in update_calls[0][0][1]

    def test_retry_failure_schedules_next(self):
        """Failed retry should increment count and schedule next retry."""
        with patch("services.common.dlq.psycopg2") as mock_pg:
            dlq, conn, cursor = self._make_dlq(mock_pg)

            cursor.fetchall.return_value = [
                {
                    "id": 1,
                    "topic": "t",
                    "message_key": "k",
                    "message_value": {},
                    "retry_count": 1,
                    "max_retries": 5,
                }
            ]

            processor = MagicMock(side_effect=ValueError("processing failed"))
            count = dlq.retry(processor)

            assert count == 0  # No success

            update_calls = [
                c
                for c in cursor.execute.call_args_list
                if c[0][0].strip().startswith("UPDATE")
            ]
            assert len(update_calls) == 1
            assert DLQStatus.RETRYING.value in update_calls[0][0][1]

    def test_max_retries_marks_exhausted(self):
        """After max_retries, message should be EXHAUSTED."""
        with patch("services.common.dlq.psycopg2") as mock_pg:
            dlq, conn, cursor = self._make_dlq(mock_pg)

            cursor.fetchall.return_value = [
                {
                    "id": 1,
                    "topic": "t",
                    "message_key": "k",
                    "message_value": {},
                    "retry_count": 4,
                    "max_retries": 5,
                }
            ]

            processor = MagicMock(side_effect=ValueError("still failing"))
            count = dlq.retry(processor)

            assert count == 0

            update_calls = [
                c
                for c in cursor.execute.call_args_list
                if c[0][0].strip().startswith("UPDATE")
            ]
            assert len(update_calls) == 1
            assert DLQStatus.EXHAUSTED.value in update_calls[0][0][1]

    def test_exhausted_calls_on_exhausted_callback(self):
        """on_exhausted callback should fire when max retries exhausted."""
        with patch("services.common.dlq.psycopg2") as mock_pg:
            dlq, conn, cursor = self._make_dlq(mock_pg)
            callback = MagicMock()
            dlq._on_exhausted = callback

            row = {
                "id": 1,
                "topic": "t",
                "message_key": "k",
                "message_value": {},
                "retry_count": 4,
                "max_retries": 5,
            }
            cursor.fetchall.return_value = [row]

            processor = MagicMock(side_effect=ValueError("fail"))
            dlq.retry(processor)

            callback.assert_called_once()

    def test_retry_returns_count_of_successes(self):
        """retry() should return count of successfully processed messages."""
        with patch("services.common.dlq.psycopg2") as mock_pg:
            dlq, conn, cursor = self._make_dlq(mock_pg)

            call_count = {"n": 0}

            def alternating_processor(topic, key, value):
                call_count["n"] += 1
                if call_count["n"] % 2 == 0:
                    raise ValueError("fail")

            cursor.fetchall.return_value = [
                {
                    "id": i,
                    "topic": "t",
                    "message_key": "k",
                    "message_value": {},
                    "retry_count": 0,
                    "max_retries": 5,
                }
                for i in range(4)
            ]

            count = dlq.retry(alternating_processor)
            assert count == 2  # messages 1 and 3 succeed (odd calls)

    def test_no_messages_returns_zero(self):
        """retry() with no due messages should return 0."""
        with patch("services.common.dlq.psycopg2") as mock_pg:
            dlq, conn, cursor = self._make_dlq(mock_pg)
            cursor.fetchall.return_value = []

            count = dlq.retry(MagicMock())
            assert count == 0


# ── 5. Status transitions ───────────────────────────────────────────────────


class TestDLQStatusTransitions:
    """Verify the complete lifecycle: PENDING -> RETRYING -> RESOLVED/EXHAUSTED."""

    def test_pending_to_retrying(self):
        """First failed retry moves from PENDING to RETRYING."""
        with patch("services.common.dlq.psycopg2") as mock_pg:
            conn, cursor = _mock_conn()
            mock_pg.connect.return_value = conn
            mock_pg.extras = MagicMock()
            mock_pg.extras.RealDictCursor = "RealDictCursor"

            dlq = DeadLetterQueue.__new__(DeadLetterQueue)
            dlq.db_url = "postgresql://test"
            dlq.service_name = "test"
            dlq.max_retries = 5
            dlq._on_exhausted = None
            dlq._ensure_table = MagicMock()

            cursor.fetchall.return_value = [
                {
                    "id": 1,
                    "topic": "t",
                    "message_key": None,
                    "message_value": {},
                    "retry_count": 0,
                    "max_retries": 5,
                }
            ]

            dlq.retry(MagicMock(side_effect=Exception("fail")))

            update_calls = [
                c
                for c in cursor.execute.call_args_list
                if c[0][0].strip().startswith("UPDATE")
            ]
            params = update_calls[0][0][1]
            assert params[0] == DLQStatus.RETRYING.value
            assert params[1] == 1  # retry_count incremented

    def test_retrying_to_resolved(self):
        """Successful retry moves from RETRYING to RESOLVED."""
        with patch("services.common.dlq.psycopg2") as mock_pg:
            conn, cursor = _mock_conn()
            mock_pg.connect.return_value = conn
            mock_pg.extras = MagicMock()
            mock_pg.extras.RealDictCursor = "RealDictCursor"

            dlq = DeadLetterQueue.__new__(DeadLetterQueue)
            dlq.db_url = "postgresql://test"
            dlq.service_name = "test"
            dlq.max_retries = 5
            dlq._on_exhausted = None
            dlq._ensure_table = MagicMock()

            cursor.fetchall.return_value = [
                {
                    "id": 1,
                    "topic": "t",
                    "message_key": None,
                    "message_value": {},
                    "retry_count": 2,
                    "max_retries": 5,
                }
            ]

            dlq.retry(MagicMock())  # Succeeds

            update_calls = [
                c
                for c in cursor.execute.call_args_list
                if c[0][0].strip().startswith("UPDATE")
            ]
            params = update_calls[0][0][1]
            assert params[0] == DLQStatus.RESOLVED.value

    def test_retrying_to_exhausted(self):
        """Last failed retry moves from RETRYING to EXHAUSTED."""
        with patch("services.common.dlq.psycopg2") as mock_pg:
            conn, cursor = _mock_conn()
            mock_pg.connect.return_value = conn
            mock_pg.extras = MagicMock()
            mock_pg.extras.RealDictCursor = "RealDictCursor"

            dlq = DeadLetterQueue.__new__(DeadLetterQueue)
            dlq.db_url = "postgresql://test"
            dlq.service_name = "test"
            dlq.max_retries = 3
            dlq._on_exhausted = None
            dlq._ensure_table = MagicMock()

            cursor.fetchall.return_value = [
                {
                    "id": 1,
                    "topic": "t",
                    "message_key": None,
                    "message_value": {},
                    "retry_count": 2,
                    "max_retries": 3,
                }
            ]

            dlq.retry(MagicMock(side_effect=Exception("fail")))

            update_calls = [
                c
                for c in cursor.execute.call_args_list
                if c[0][0].strip().startswith("UPDATE")
            ]
            params = update_calls[0][0][1]
            assert params[0] == DLQStatus.EXHAUSTED.value


# ── 6. Background retry worker ──────────────────────────────────────────────


class TestRetryWorker:
    """Test the background retry worker thread."""

    def test_worker_starts_daemon_thread(self):
        dlq = MagicMock()
        dlq.retry.return_value = 0

        t = start_retry_worker(dlq, MagicMock(), interval_s=0.05)
        assert t.is_alive()
        assert t.daemon is True
        time.sleep(0.15)
        assert dlq.retry.call_count >= 2

    def test_worker_calls_retry_with_processor(self):
        dlq = MagicMock()
        dlq.retry.return_value = 0
        processor = MagicMock()

        start_retry_worker(dlq, processor, interval_s=0.05)
        time.sleep(0.1)

        assert dlq.retry.call_count >= 1
        dlq.retry.assert_called_with(processor)

    def test_worker_survives_exceptions(self):
        dlq = MagicMock()
        dlq.retry.side_effect = Exception("db down")

        t = start_retry_worker(dlq, MagicMock(), interval_s=0.05)
        time.sleep(0.15)
        assert t.is_alive()  # Should not crash


# ── 7. DLQ table in schema ──────────────────────────────────────────────────


class TestDLQSchema:
    """Verify DLQ table exists in postgres_schema.py."""

    @pytest.fixture(autouse=True)
    def load_schema(self):
        from postgres_schema import CREATE_SCHEMA_SQL

        self.sql = CREATE_SCHEMA_SQL

    def test_dlq_table_exists(self):
        assert "dead_letter_queue" in self.sql

    def test_dlq_has_required_columns(self):
        for col in [
            "service_name",
            "topic",
            "message_key",
            "message_value",
            "error",
            "retry_count",
            "max_retries",
            "next_retry_at",
            "created_at",
            "updated_at",
            "status",
        ]:
            assert col in self.sql

    def test_dlq_indexes_exist(self):
        assert "idx_dlq_status_next_retry" in self.sql
        assert "idx_dlq_service" in self.sql
        assert "idx_dlq_created" in self.sql

    def test_dlq_default_status_is_pending(self):
        assert "DEFAULT 'PENDING'" in self.sql


# ── 8. DLQ integration in microservices ──────────────────────────────────────


class TestDLQIntegration:
    """Verify DLQ is wired into each microservice."""

    def _read_file(self, path):
        with open(os.path.join(BASE_DIR, path)) as f:
            return f.read()

    def test_sentiment_engine_imports_dlq(self):
        content = self._read_file("microservices/sentiment-engine/main.py")
        assert "from services.common.dlq import" in content

    def test_sentiment_engine_persists_to_dlq(self):
        content = self._read_file("microservices/sentiment-engine/main.py")
        assert "dlq.persist(" in content

    def test_sentiment_engine_starts_retry_worker(self):
        content = self._read_file("microservices/sentiment-engine/main.py")
        assert "start_retry_worker" in content

    def test_trading_engine_imports_dlq(self):
        content = self._read_file("microservices/trading-engine/main.py")
        assert "from services.common.dlq import" in content

    def test_trading_engine_persists_to_dlq(self):
        content = self._read_file("microservices/trading-engine/main.py")
        assert "dlq.persist(" in content

    def test_tft_predictor_imports_dlq(self):
        content = self._read_file("microservices/tft-predictor/main.py")
        assert "from services.common.dlq import" in content

    def test_tft_predictor_persists_to_dlq(self):
        content = self._read_file("microservices/tft-predictor/main.py")
        assert "dlq.persist(" in content

    def test_orchestrator_imports_dlq(self):
        content = self._read_file("microservices/orchestrator/main.py")
        assert "from services.common.dlq import" in content

    def test_orchestrator_persists_to_dlq(self):
        content = self._read_file("microservices/orchestrator/main.py")
        assert "dlq.persist(" in content


# ── 9. DLQ dashboard endpoint ───────────────────────────────────────────────


class TestDLQDashboard:
    """Verify /dlq endpoint exists in paper-trader."""

    def test_dlq_endpoint_defined(self):
        content = self._read_file("paper-trader/main.py")
        assert '@app.get("/dlq")' in content

    def test_dlq_endpoint_returns_stats(self):
        content = self._read_file("paper-trader/main.py")
        assert "get_all_stats" in content

    def _read_file(self, path):
        with open(os.path.join(BASE_DIR, path)) as f:
            return f.read()


# ── 10. CREATE_DLQ_TABLE_SQL consistency ─────────────────────────────────────


class TestCreateDLQTableSQL:
    """Verify the embedded CREATE TABLE SQL in dlq.py is valid."""

    def test_has_all_columns(self):
        for col in [
            "service_name",
            "topic",
            "message_key",
            "message_value",
            "error",
            "retry_count",
            "max_retries",
            "next_retry_at",
            "created_at",
            "updated_at",
            "status",
        ]:
            assert col in CREATE_DLQ_TABLE_SQL

    def test_has_indexes(self):
        assert "idx_dlq_status_next_retry" in CREATE_DLQ_TABLE_SQL
        assert "idx_dlq_service" in CREATE_DLQ_TABLE_SQL
