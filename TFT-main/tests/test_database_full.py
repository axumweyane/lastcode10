"""
Full PostgreSQL database tests for APEX Trading System.
Tests schema integrity, table existence, column definitions, and TimescaleDB setup.
Connects to: localhost:15432 / apex / apex_user / apex_pass
"""

import pytest
import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 15432,
    "dbname": "apex",
    "user": "apex_user",
    "password": "apex_pass",
}


@pytest.fixture(scope="module")
def db_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def db_cursor(db_conn):
    cur = db_conn.cursor()
    yield cur
    cur.close()


# ---------- 1. Connection works ----------

class TestConnection:
    def test_connection_succeeds(self):
        conn = psycopg2.connect(**DB_CONFIG)
        assert conn.status == psycopg2.extensions.STATUS_READY
        conn.close()

    def test_database_name(self, db_conn):
        assert db_conn.info.dbname == "apex"

    def test_server_responds_to_query(self, db_cursor):
        db_cursor.execute("SELECT 1")
        assert db_cursor.fetchone()[0] == 1


# ---------- 2. All expected tables exist ----------

EXPECTED_TABLES = [
    "bayesian_weight_state",
    "calibration_snapshots",
    "circuit_breaker_closures",
    "circuit_breaker_events",
    "dead_letter_queue",
    "decision_records",
    "features",
    "forex_candles",
    "forex_cot",
    "forex_economic_calendar",
    "forex_positions",
    "iv_surface",
    "market_raw_minute",
    "model_performance",
    "ohlcv_bars",
    "options_chains",
    "options_flow_features",
    "options_positions",
    "orders",
    "paper_daily_snapshots",
    "paper_execution_stats",
    "paper_risk_reports",
    "paper_signal_analyses",
    "paper_strategy_signals",
    "paper_trades",
    "portfolio_snapshots",
    "positions",
    "risk_events",
    "signal_attribution",
    "signals",
    "signals_scored",
]


class TestTablesExist:
    def test_all_expected_tables_exist(self, db_cursor):
        db_cursor.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname='public'"
        )
        actual = {row[0] for row in db_cursor.fetchall()}
        missing = [t for t in EXPECTED_TABLES if t not in actual]
        assert missing == [], f"Missing tables: {missing}"

    @pytest.mark.parametrize("table", EXPECTED_TABLES)
    def test_table_exists(self, db_cursor, table):
        db_cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema='public' AND table_name=%s)",
            (table,),
        )
        assert db_cursor.fetchone()[0], f"Table {table} does not exist"


# ---------- 3. paper_trades has required columns ----------

class TestPaperTradesSchema:
    REQUIRED_COLUMNS = ["symbol", "side", "quantity", "price"]

    @pytest.mark.parametrize("column", REQUIRED_COLUMNS)
    def test_paper_trades_has_column(self, db_cursor, column):
        db_cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='paper_trades' AND column_name=%s",
            (column,),
        )
        result = db_cursor.fetchone()
        assert result is not None, f"paper_trades missing column: {column}"

    def test_paper_trades_has_date_column(self, db_cursor):
        """trade_date or time column for timestamps."""
        db_cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='paper_trades' AND column_name IN ('trade_date', 'time')"
        )
        assert db_cursor.fetchone() is not None, "paper_trades has no date/time column"


# ---------- 4. portfolio_snapshots has high_water_mark (SCRUM-23) ----------

class TestPortfolioSnapshotsSchema:
    def test_has_high_water_mark(self, db_cursor):
        db_cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='portfolio_snapshots' AND column_name='high_water_mark'"
        )
        assert db_cursor.fetchone() is not None, (
            "portfolio_snapshots missing high_water_mark column (SCRUM-23 fix)"
        )

    def test_has_portfolio_value(self, db_cursor):
        db_cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='portfolio_snapshots' AND column_name='portfolio_value'"
        )
        assert db_cursor.fetchone() is not None

    def test_has_snapshot_type(self, db_cursor):
        db_cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='portfolio_snapshots' AND column_name='snapshot_type'"
        )
        assert db_cursor.fetchone() is not None


# ---------- 5. dead_letter_queue has required columns ----------

class TestDeadLetterQueueSchema:
    REQUIRED_COLUMNS = ["id", "source_service", "status", "payload", "retry_count"]

    @pytest.mark.parametrize("column", REQUIRED_COLUMNS)
    def test_dlq_has_column(self, db_cursor, column):
        db_cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='dead_letter_queue' AND column_name=%s",
            (column,),
        )
        result = db_cursor.fetchone()
        assert result is not None, f"dead_letter_queue missing column: {column}"


# ---------- 6. SELECT COUNT(*) works on every table ----------

class TestTableReadable:
    @pytest.mark.parametrize("table", EXPECTED_TABLES)
    def test_select_count(self, db_cursor, table):
        db_cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
        count = db_cursor.fetchone()[0]
        assert count >= 0, f"SELECT COUNT(*) failed on {table}"


# ---------- 7. TimescaleDB extension is installed ----------

class TestTimescaleDB:
    def test_extension_installed(self, db_cursor):
        db_cursor.execute(
            "SELECT extname FROM pg_extension WHERE extname='timescaledb'"
        )
        result = db_cursor.fetchone()
        assert result is not None, "TimescaleDB extension not installed"
        assert result[0] == "timescaledb"

    def test_timescaledb_version(self, db_cursor):
        db_cursor.execute(
            "SELECT extversion FROM pg_extension WHERE extname='timescaledb'"
        )
        version = db_cursor.fetchone()[0]
        assert version is not None
        # Should be 2.x
        major = int(version.split(".")[0])
        assert major >= 2, f"TimescaleDB version too old: {version}"
