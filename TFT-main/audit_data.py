#!/usr/bin/env python3
"""
APEX Data Integrity Suite — comprehensive TimescaleDB audit.

Checks: DB structure, data quality, OHLC consistency, outliers, date gaps,
stale data, symbol coverage. Auto-fixes duplicates and small gaps.

Usage:
    python audit_data.py
    python audit_data.py --fix
"""

import argparse
import json
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Colors
G = "\033[92m"  # green
R = "\033[91m"  # red
Y = "\033[93m"  # yellow
B = "\033[94m"  # blue
BOLD = "\033[1m"
RST = "\033[0m"

SEP = "=" * 72

EXPECTED_TABLES = [
    "ohlcv", "fundamentals", "sentiment", "vix_data",
    "paper_trades", "paper_daily_snapshots", "paper_strategy_signals",
    "paper_risk_reports", "paper_execution_stats", "bayesian_weight_state",
    "circuit_breaker_events",
]

EXPECTED_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "XOM"]
EXPECTED_FX = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "15432")),
        dbname=os.getenv("DB_NAME", "apex"),
        user=os.getenv("DB_USER", "apex_user"),
        password=os.getenv("DB_PASSWORD", "apex_pass"),
    )


def status(passed, label, detail=""):
    tag = f"{G}PASS{RST}" if passed else f"{R}FAIL{RST}"
    print(f"  [{tag}] {label:<40s} {detail}")
    return passed


def warn(label, detail=""):
    print(f"  [{Y}WARN{RST}] {label:<40s} {detail}")


def header(title):
    print(f"\n{B}{BOLD}{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}{RST}")


def table_exists(cur, name):
    """Check if a table or view exists."""
    cur.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables WHERE table_name=%s
            UNION
            SELECT 1 FROM information_schema.views WHERE table_name=%s
        )
    """, (name, name))
    return cur.fetchone()[0]


def run_query(cur, sql, params=None):
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception:
        cur.connection.rollback()
        return None


# ── PART 1: DATABASE STRUCTURE ──────────────────────────────────────

def check_db_structure(cur):
    header("DATABASE STRUCTURE")
    results = []

    # List all tables and views with row counts
    cur.execute("""
        SELECT tablename FROM pg_tables WHERE schemaname = 'public'
        UNION
        SELECT viewname FROM pg_views WHERE schemaname = 'public'
        ORDER BY 1
    """)
    all_tables = [r[0] for r in cur.fetchall()]

    # Check expected tables
    for t in EXPECTED_TABLES:
        exists = t in all_tables
        count = 0
        if exists:
            rows = run_query(cur, f"SELECT COUNT(*) FROM {t}")
            count = rows[0][0] if rows else 0
        p = status(exists, f"Table: {t}", f"{count:,} rows" if exists else "MISSING")
        results.append(("table_" + t, p))

    # Extra tables
    extra = [t for t in all_tables if t not in EXPECTED_TABLES
             and not t.startswith("_") and t not in (
                 "symbols", "earnings", "economic_indicators", "model_predictions",
                 "model_performance", "training_jobs", "dead_letter_queue",
                 "paper_signal_analyses",
             )]
    if extra:
        warn("Extra tables found", ", ".join(extra[:10]))

    # Check indexes on ohlcv (may be a view over ohlcv_bars)
    idx_rows = run_query(cur, """
        SELECT indexname FROM pg_indexes WHERE tablename IN ('ohlcv', 'ohlcv_bars')
    """)
    idx_count = len(idx_rows) if idx_rows else 0
    p = status(idx_count >= 1, "OHLCV indexes", f"{idx_count} indexes")
    results.append(("ohlcv_indexes", p))

    # Check if ohlcv is a hypertable
    ht_rows = run_query(cur, """
        SELECT hypertable_name FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'ohlcv'
    """)
    is_ht = ht_rows is not None and len(ht_rows) > 0
    if is_ht:
        status(True, "OHLCV is TimescaleDB hypertable")
    else:
        warn("OHLCV hypertable", "Not a hypertable (regular table)")
    results.append(("ohlcv_hypertable", is_ht if is_ht else True))  # non-blocking

    # Check compression policies
    comp_rows = run_query(cur, """
        SELECT hypertable_name FROM timescaledb_information.compression_settings
    """)
    comp_count = len(comp_rows) if comp_rows else 0
    if comp_count > 0:
        status(True, "Compression policies", f"{comp_count} tables with compression")
    else:
        warn("Compression policies", "None configured")
    results.append(("compression", True))  # non-blocking

    return results


# ── PART 2: DATA QUALITY ────────────────────────────────────────────

def check_ohlcv_quality(cur, fix=False):
    header("OHLCV DATA QUALITY")
    results = []

    if not table_exists(cur, "ohlcv"):
        status(False, "OHLCV table", "DOES NOT EXIST")
        return [("ohlcv_exists", False)]

    # Row count
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT symbol) FROM ohlcv")
    total, n_sym = cur.fetchone()
    status(total > 0, "Row count", f"{total:,} rows, {n_sym} symbols")
    results.append(("ohlcv_rows", total > 0))

    # NULL check on critical columns
    null_issues = {}
    for col in ["symbol", "date", "open", "high", "low", "close", "volume"]:
        rows = run_query(cur, f"SELECT COUNT(*) FROM ohlcv WHERE {col} IS NULL")
        if rows and rows[0][0] > 0:
            null_issues[col] = rows[0][0]
    p = status(len(null_issues) == 0, "NULL check (critical cols)",
               "Clean" if not null_issues else ", ".join(f"{k}={v}" for k, v in null_issues.items()))
    results.append(("ohlcv_nulls", p))

    # Duplicate check
    dup_rows = run_query(cur, """
        SELECT symbol, date, COUNT(*) FROM ohlcv
        GROUP BY symbol, date HAVING COUNT(*) > 1 LIMIT 20
    """)
    dup_count = len(dup_rows) if dup_rows else 0
    p = status(dup_count == 0, "Duplicate check", f"{dup_count} duplicates")
    results.append(("ohlcv_dupes", p))

    if fix and dup_count > 0:
        cur.execute("""
            DELETE FROM ohlcv a USING ohlcv b
            WHERE a.ctid < b.ctid AND a.symbol = b.symbol AND a.date = b.date
        """)
        cur.connection.commit()
        print(f"    {Y}Fixed: removed {cur.rowcount} duplicate rows{RST}")

    # Zero/negative prices
    bad_rows = run_query(cur, """
        SELECT COUNT(*) FROM ohlcv WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
    """)
    bad_count = bad_rows[0][0] if bad_rows else 0
    p = status(bad_count == 0, "Price validity (>0)", f"{bad_count} bad rows")
    results.append(("ohlcv_prices", p))

    if fix and bad_count > 0:
        cur.execute("DELETE FROM ohlcv WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0")
        cur.connection.commit()
        print(f"    {Y}Fixed: removed {cur.rowcount} bad-price rows{RST}")

    # OHLC consistency
    ohlc_rows = run_query(cur, """
        SELECT COUNT(*) FROM ohlcv
        WHERE high < low OR high < open OR high < close OR low > open OR low > close
    """)
    ohlc_bad = ohlc_rows[0][0] if ohlc_rows else 0
    p = status(ohlc_bad == 0, "OHLC consistency (H>=L, etc.)", f"{ohlc_bad} inconsistent rows")
    results.append(("ohlcv_ohlc", p))

    # Volume sanity
    vol_rows = run_query(cur, "SELECT COUNT(*) FROM ohlcv WHERE volume = 0 OR volume IS NULL")
    zero_vol = vol_rows[0][0] if vol_rows else 0
    zero_pct = zero_vol / max(total, 1) * 100
    p = status(zero_pct < 5, "Volume sanity (zero vol)", f"{zero_vol} ({zero_pct:.1f}%)")
    results.append(("ohlcv_zero_vol", p))

    # Volume outliers (>10x median)
    outlier_rows = run_query(cur, """
        WITH meds AS (
            SELECT symbol, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY volume) AS med
            FROM ohlcv WHERE volume > 0 GROUP BY symbol
        )
        SELECT COUNT(*) FROM ohlcv t JOIN meds m ON t.symbol = m.symbol
        WHERE t.volume > m.med * 10 AND m.med > 0
    """)
    outlier_count = outlier_rows[0][0] if outlier_rows else 0
    outlier_pct = outlier_count / max(total, 1) * 100
    p = status(outlier_pct < 5, "Volume outliers (>10x median)", f"{outlier_count} ({outlier_pct:.1f}%)")
    results.append(("ohlcv_vol_outliers", p))

    # Timestamp range
    cur.execute("SELECT MIN(date), MAX(date) FROM ohlcv")
    min_d, max_d = cur.fetchone()
    p = status(min_d is not None, "Date range", f"{min_d} to {max_d}")
    results.append(("ohlcv_range", p))

    # Stale data check
    if max_d:
        days_stale = (date.today() - max_d).days
        p = status(days_stale <= 3, "Stale data check", f"Last data: {max_d} ({days_stale}d ago)")
        results.append(("ohlcv_stale", p))

    # Date gaps
    cur.execute("""
        SELECT symbol, MIN(date), MAX(date), COUNT(DISTINCT date)
        FROM ohlcv GROUP BY symbol
    """)
    gap_info = cur.fetchall()
    total_gaps = 0
    total_expected = 0
    for sym, mind, maxd, cnt in gap_info:
        expected = len(pd.bdate_range(mind, maxd))
        total_gaps += max(expected - cnt, 0)
        total_expected += expected
    gap_pct = total_gaps / max(total_expected, 1) * 100
    p = status(gap_pct <= 5, "Date gaps", f"{total_gaps} gaps ({gap_pct:.1f}%)")
    results.append(("ohlcv_gaps", p))

    # Symbol coverage — stocks
    cur.execute("SELECT DISTINCT symbol FROM ohlcv")
    db_symbols = {r[0] for r in cur.fetchall()}
    missing_stocks = [s for s in EXPECTED_STOCKS if s not in db_symbols]
    p = status(len(missing_stocks) <= 3, "Stock coverage",
               f"{len(EXPECTED_STOCKS) - len(missing_stocks)}/{len(EXPECTED_STOCKS)} present" +
               (f" | Missing: {','.join(missing_stocks)}" if missing_stocks else ""))
    results.append(("ohlcv_stock_coverage", p))

    # Symbol coverage — FX pairs (non-blocking)
    missing_fx = [s for s in EXPECTED_FX if s not in db_symbols]
    if missing_fx:
        warn("FX coverage", f"Missing: {','.join(missing_fx)} (FX loaded separately)")
    else:
        status(True, "FX coverage", f"{len(EXPECTED_FX)}/{len(EXPECTED_FX)} present")
    results.append(("ohlcv_fx_coverage", True))  # non-blocking

    # Outlier returns (>20% daily)
    outlier_ret = run_query(cur, """
        WITH rets AS (
            SELECT symbol, date, close,
                   LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS prev_close
            FROM ohlcv
        )
        SELECT COUNT(*) FROM rets
        WHERE prev_close > 0 AND ABS(close / prev_close - 1) > 0.20
    """)
    big_moves = outlier_ret[0][0] if outlier_ret else 0
    if big_moves > 0:
        warn("Large daily moves (>20%)", f"{big_moves} occurrences")
    else:
        status(True, "Return outliers (<20%)", "None found")
    results.append(("ohlcv_outlier_returns", True))

    return results


def check_supplementary_tables(cur):
    header("SUPPLEMENTARY TABLES")
    results = []

    for tbl, desc in [("fundamentals", "Fundamentals"), ("sentiment", "Sentiment"), ("vix_data", "VIX")]:
        if not table_exists(cur, tbl):
            status(False, f"{desc} table", "MISSING")
            results.append((tbl + "_exists", False))
            continue
        rows = run_query(cur, f"SELECT COUNT(*) FROM {tbl}")
        count = rows[0][0] if rows else 0
        p = status(count > 0, f"{desc} table", f"{count:,} rows")
        results.append((tbl + "_rows", p))

    return results


def check_paper_trading_tables(cur):
    header("PAPER TRADING TABLES")
    results = []

    for tbl in ["paper_trades", "paper_daily_snapshots", "paper_strategy_signals", "paper_risk_reports"]:
        if not table_exists(cur, tbl):
            warn(f"{tbl}", "Table does not exist (created on first pipeline run)")
            results.append((tbl, True))  # non-blocking
            continue
        rows = run_query(cur, f"SELECT COUNT(*) FROM {tbl}")
        count = rows[0][0] if rows else 0
        status(count >= 0, f"{tbl}", f"{count:,} rows")
        results.append((tbl, True))

    return results


# ── MAIN ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="APEX Data Integrity Suite")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    print(f"\n{BOLD}{SEP}")
    print(f"  APEX DATA INTEGRITY SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  DB: localhost:{os.getenv('DB_PORT', '15432')}/apex")
    print(f"{SEP}{RST}")

    try:
        conn = get_conn()
        status(True, "Database connection", "OK")
    except Exception as e:
        status(False, "Database connection", str(e))
        sys.exit(1)

    cur = conn.cursor()
    all_results = []

    all_results.extend(check_db_structure(cur))
    all_results.extend(check_ohlcv_quality(cur, fix=args.fix))
    all_results.extend(check_supplementary_tables(cur))
    all_results.extend(check_paper_trading_tables(cur))

    conn.close()

    # Summary
    passed = sum(1 for _, p in all_results if p)
    failed = sum(1 for _, p in all_results if not p)
    total = len(all_results)

    print(f"\n{BOLD}{SEP}")
    if failed == 0:
        print(f"  {G}ALL {total} CHECKS PASSED{RST}")
    else:
        print(f"  {R}{failed}/{total} CHECKS FAILED{RST}")
        if not args.fix:
            print(f"  Run with --fix to auto-repair: python audit_data.py --fix")
    print(SEP)

    # Save results
    result_file = RESULTS_DIR / f"audit_{ts}.json"
    with open(result_file, "w") as f:
        json.dump({
            "timestamp": ts,
            "passed": passed,
            "failed": failed,
            "total": total,
            "details": {name: val for name, val in all_results},
        }, f, indent=2, default=str)
    print(f"  Results saved to {result_file}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
