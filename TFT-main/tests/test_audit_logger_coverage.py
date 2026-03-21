"""
Coverage tests for trading/persistence/audit.py.
PostgreSQL is mocked — no real database needed.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch, call

import pytest

from trading.persistence.audit import AuditLogger, CIRCUIT_BREAKER_SCHEMA_SQL


# ---------- Mock helpers ----------

def _make_mock_cursor(fetchone_val=None, fetchall_val=None, description=None):
    cur = MagicMock()
    cur.fetchone.return_value = fetchone_val
    cur.fetchall.return_value = fetchall_val or []
    cur.description = description
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    return cur


def _make_mock_conn(cursor):
    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


@pytest.fixture
def audit():
    return AuditLogger(db_config={
        "host": "localhost",
        "port": 5432,
        "database": "test_db",
        "user": "test_user",
        "password": "test_pass",
    })


@pytest.fixture
def mock_conn_factory():
    """Returns a function that patches _get_connection with a mock."""
    def factory(audit_logger, cursor):
        conn = _make_mock_conn(cursor)

        @contextmanager
        def fake_get_connection():
            yield conn

        patcher = patch.object(audit_logger, "_get_connection", fake_get_connection)
        return patcher, conn

    return factory


# ---------- Constructor ----------

class TestConstructor:
    def test_explicit_config(self):
        cfg = {"host": "h", "port": 1234, "database": "d", "user": "u", "password": "p"}
        a = AuditLogger(db_config=cfg)
        assert a.db_config == cfg

    def test_env_defaults(self):
        with patch.dict("os.environ", {
            "DB_HOST": "envhost", "DB_PORT": "9999",
            "DB_NAME": "envdb", "DB_USER": "envuser", "DB_PASSWORD": "envpass",
        }):
            a = AuditLogger()
            assert a.db_config["host"] == "envhost"
            assert a.db_config["port"] == 9999


# ---------- create_schema ----------

class TestCreateSchema:
    def test_create_schema(self, audit, mock_conn_factory):
        cur = _make_mock_cursor()
        patcher, conn = mock_conn_factory(audit, cur)
        with patcher:
            audit.create_schema()
        cur.execute.assert_called_once_with(CIRCUIT_BREAKER_SCHEMA_SQL)
        conn.commit.assert_called_once()


# ---------- log_trip_event ----------

class TestLogTripEvent:
    def test_returns_event_id(self, audit, mock_conn_factory):
        cur = _make_mock_cursor(fetchone_val=(42,))
        patcher, conn = mock_conn_factory(audit, cur)
        with patcher:
            event_id = audit.log_trip_event(
                reason="test drawdown",
                drawdown_method="high_water_mark",
                drawdown_percent=5.5,
                portfolio_value=94500,
                hwm=100000,
                sod_value=98000,
                initial_capital=100000,
                positions_closed=3,
            )
        assert event_id == 42
        conn.commit.assert_called_once()

    def test_with_metadata(self, audit, mock_conn_factory):
        cur = _make_mock_cursor(fetchone_val=(1,))
        patcher, conn = mock_conn_factory(audit, cur)
        with patcher:
            event_id = audit.log_trip_event(
                reason="test",
                drawdown_method="sod",
                drawdown_percent=3.0,
                portfolio_value=97000,
                hwm=100000,
                sod_value=100000,
                initial_capital=100000,
                positions_closed=0,
                triggered_by="manual",
                metadata={"extra": "info"},
            )
        assert event_id == 1

    def test_custom_triggered_by(self, audit, mock_conn_factory):
        cur = _make_mock_cursor(fetchone_val=(5,))
        patcher, conn = mock_conn_factory(audit, cur)
        with patcher:
            audit.log_trip_event(
                reason="r", drawdown_method="m", drawdown_percent=1.0,
                portfolio_value=99000, hwm=100000, sod_value=0,
                initial_capital=100000, positions_closed=0,
                triggered_by="operator_john",
            )
        # Verify the triggered_by param was passed
        args = cur.execute.call_args[0][1]
        assert args[0] == "operator_john"


# ---------- log_closure ----------

class TestLogClosure:
    def test_log_closure(self, audit, mock_conn_factory):
        cur = _make_mock_cursor()
        patcher, conn = mock_conn_factory(audit, cur)
        with patcher:
            audit.log_closure(
                event_id=42,
                ticker="AAPL",
                quantity=100.0,
                side="long",
                market_value=15000.0,
                unrealized_pnl=-500.0,
                close_order_id="order123",
                close_status="filled",
            )
        cur.execute.assert_called_once()
        conn.commit.assert_called_once()

    def test_log_closure_optional_fields(self, audit, mock_conn_factory):
        cur = _make_mock_cursor()
        patcher, conn = mock_conn_factory(audit, cur)
        with patcher:
            audit.log_closure(
                event_id=1, ticker="X", quantity=10,
                side="short", market_value=500, unrealized_pnl=0,
            )
        args = cur.execute.call_args[0][1]
        assert args[6] is None  # close_order_id
        assert args[7] is None  # close_status


# ---------- log_reset_event ----------

class TestLogResetEvent:
    def test_returns_event_id(self, audit, mock_conn_factory):
        cur = _make_mock_cursor(fetchone_val=(99,))
        patcher, conn = mock_conn_factory(audit, cur)
        with patcher:
            event_id = audit.log_reset_event(
                operator="admin",
                reason="manual recovery",
                portfolio_value=100000,
            )
        assert event_id == 99

    def test_with_metadata(self, audit, mock_conn_factory):
        cur = _make_mock_cursor(fetchone_val=(10,))
        patcher, conn = mock_conn_factory(audit, cur)
        with patcher:
            event_id = audit.log_reset_event(
                operator="bot", reason="auto", portfolio_value=50000,
                metadata={"source": "auto_recovery"},
            )
        assert event_id == 10


# ---------- log_portfolio_snapshot ----------

class TestLogPortfolioSnapshot:
    def test_log_snapshot(self, audit, mock_conn_factory):
        cur = _make_mock_cursor()
        patcher, conn = mock_conn_factory(audit, cur)
        with patcher:
            audit.log_portfolio_snapshot(100000.0, 105000.0, "periodic")
        args = cur.execute.call_args[0][1]
        assert args == (100000.0, 105000.0, "periodic")
        conn.commit.assert_called_once()

    def test_trip_snapshot_type(self, audit, mock_conn_factory):
        cur = _make_mock_cursor()
        patcher, conn = mock_conn_factory(audit, cur)
        with patcher:
            audit.log_portfolio_snapshot(95000, 100000, "trip")
        args = cur.execute.call_args[0][1]
        assert args[2] == "trip"


# ---------- get_recent_events ----------

class TestGetRecentEvents:
    def test_returns_events(self, audit, mock_conn_factory):
        desc = [
            ("id",), ("event_type",), ("triggered_by",), ("reason",),
            ("drawdown_method",), ("drawdown_percent",), ("portfolio_value",),
            ("hwm",), ("sod_value",), ("initial_capital",),
            ("positions_closed",), ("metadata",), ("created_at",),
        ]
        rows = [
            (1, "trip", "system", "drawdown", "hwm", 5.0, 95000, 100000, 0, 100000, 2, {}, "2026-03-21"),
        ]
        cur = _make_mock_cursor(fetchall_val=rows, description=desc)
        patcher, _ = mock_conn_factory(audit, cur)
        with patcher:
            events = audit.get_recent_events(limit=5)
        assert len(events) == 1
        assert events[0]["event_type"] == "trip"

    def test_empty_returns_empty_list(self, audit, mock_conn_factory):
        desc = [("id",), ("event_type",)]
        cur = _make_mock_cursor(fetchall_val=[], description=desc)
        patcher, _ = mock_conn_factory(audit, cur)
        with patcher:
            events = audit.get_recent_events()
        assert events == []


# ---------- get_closures_for_event ----------

class TestGetClosuresForEvent:
    def test_returns_closures(self, audit, mock_conn_factory):
        desc = [
            ("id",), ("event_id",), ("ticker",), ("quantity",), ("side",),
            ("market_value",), ("unrealized_pnl",), ("close_order_id",),
            ("close_status",), ("created_at",),
        ]
        rows = [
            (1, 42, "AAPL", 100, "long", 15000, -500, "o1", "filled", "2026-03-21"),
            (2, 42, "GOOGL", 50, "short", 8000, 200, "o2", "filled", "2026-03-21"),
        ]
        cur = _make_mock_cursor(fetchall_val=rows, description=desc)
        patcher, _ = mock_conn_factory(audit, cur)
        with patcher:
            closures = audit.get_closures_for_event(42)
        assert len(closures) == 2
        assert closures[0]["ticker"] == "AAPL"


# ---------- get_latest_snapshot ----------

class TestGetLatestSnapshot:
    def test_returns_snapshot(self, audit, mock_conn_factory):
        desc = [("portfolio_value",), ("high_water_mark",), ("snapshot_type",), ("created_at",)]
        cur = _make_mock_cursor(
            fetchone_val=(100000, 105000, "periodic", "2026-03-21"),
            description=desc,
        )
        patcher, _ = mock_conn_factory(audit, cur)
        with patcher:
            snap = audit.get_latest_snapshot()
        assert snap["portfolio_value"] == 100000
        assert snap["high_water_mark"] == 105000

    def test_returns_none_when_empty(self, audit, mock_conn_factory):
        desc = [("portfolio_value",), ("high_water_mark",), ("snapshot_type",), ("created_at",)]
        cur = _make_mock_cursor(fetchone_val=None, description=desc)
        patcher, _ = mock_conn_factory(audit, cur)
        with patcher:
            snap = audit.get_latest_snapshot()
        assert snap is None


# ---------- get_latest_trip_event ----------

class TestGetLatestTripEvent:
    def test_returns_event(self, audit, mock_conn_factory):
        desc = [
            ("id",), ("event_type",), ("triggered_by",), ("reason",),
            ("drawdown_method",), ("drawdown_percent",), ("portfolio_value",),
            ("hwm",), ("sod_value",), ("initial_capital",),
            ("positions_closed",), ("metadata",), ("created_at",),
        ]
        row = (1, "trip", "system", "drawdown", "hwm", 5.0, 95000, 100000, 0, 100000, 2, {}, "2026-03-21")
        cur = _make_mock_cursor(fetchone_val=row, description=desc)
        patcher, _ = mock_conn_factory(audit, cur)
        with patcher:
            event = audit.get_latest_trip_event()
        assert event["event_type"] == "trip"

    def test_returns_none_when_no_events(self, audit, mock_conn_factory):
        desc = [("id",), ("event_type",)]
        cur = _make_mock_cursor(fetchone_val=None, description=desc)
        patcher, _ = mock_conn_factory(audit, cur)
        with patcher:
            event = audit.get_latest_trip_event()
        assert event is None


# ---------- _get_connection error handling ----------

class TestGetConnectionErrors:
    def test_connection_error_propagates(self, audit):
        with patch("trading.persistence.audit.psycopg2.connect", side_effect=Exception("conn fail")):
            with pytest.raises(Exception, match="conn fail"):
                with audit._get_connection() as conn:
                    pass

    def test_rollback_on_error(self, audit):
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock()
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch("trading.persistence.audit.psycopg2.connect", return_value=mock_conn):
            with pytest.raises(ValueError):
                with audit._get_connection() as conn:
                    raise ValueError("query failed")
        mock_conn.rollback.assert_called_once()
        mock_conn.close.assert_called_once()
