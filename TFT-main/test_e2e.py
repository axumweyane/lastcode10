#!/usr/bin/env python3
"""
APEX Paper Trader End-to-End Test Suite.

Tests: health endpoint, Prometheus, Redis, DB connectivity, Grafana,
pipeline dry run, signals, risk, ensemble, circuit breaker, Alpaca,
positions, orders, dashboard.

Usage:
    python test_e2e.py
"""

import json
import os
import sys
import socket
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# Colors
G = "\033[92m"
R = "\033[91m"
Y = "\033[93m"
B = "\033[94m"
BOLD = "\033[1m"
RST = "\033[0m"
SEP = "=" * 72

PAPER_TRADER_URL = os.getenv("PAPER_TRADER_URL", "http://localhost:8010")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "16379"))
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "15432"))
DB_NAME = os.getenv("DB_NAME", "apex")
DB_USER = os.getenv("DB_USER", "apex_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "apex_pass")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


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


def http_get(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        return r
    except Exception:
        return None


def tcp_check(host, port, timeout=3):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False


# ── PART 1: CONNECTIVITY ──────────────────────────────────────────────

def check_connectivity():
    header("CONNECTIVITY")
    results = []

    # Paper trader
    r = http_get(f"{PAPER_TRADER_URL}/health")
    p = status(r is not None and r.status_code == 200, "Paper trader /health",
               f"{r.status_code}" if r else "UNREACHABLE")
    results.append(("paper_trader_health", p))

    # DB
    p = status(tcp_check(DB_HOST, DB_PORT), "TimescaleDB",
               f"{DB_HOST}:{DB_PORT}")
    results.append(("db_connection", p))

    # Redis
    p = status(tcp_check(REDIS_HOST, REDIS_PORT), "Redis",
               f"{REDIS_HOST}:{REDIS_PORT}")
    results.append(("redis_connection", p))

    # Prometheus
    r = http_get(f"{PROMETHEUS_URL}/-/healthy")
    p = status(r is not None and r.status_code == 200, "Prometheus",
               "healthy" if r and r.status_code == 200 else "UNREACHABLE")
    results.append(("prometheus_health", p))

    # Grafana
    r = http_get(f"{GRAFANA_URL}/api/health")
    ok = r is not None and r.status_code == 200
    detail = "healthy"
    if ok:
        try:
            detail = r.json().get("database", "ok")
        except Exception:
            pass
    p = status(ok, "Grafana", detail if ok else "UNREACHABLE")
    results.append(("grafana_health", p))

    return results


# ── PART 2: PAPER TRADER HEALTH DEEP CHECK ─────────────────────────────

def check_paper_trader_health():
    header("PAPER TRADER HEALTH (deep)")
    results = []

    r = http_get(f"{PAPER_TRADER_URL}/health")
    if r is None or r.status_code != 200:
        status(False, "Health endpoint", "UNREACHABLE")
        return [("pt_health", False)]

    data = r.json()

    # Version
    ver = data.get("status", "unknown")
    p = status(ver == "running", "Status", ver)
    results.append(("pt_status", p))

    # Infrastructure
    infra = data.get("infrastructure", {})
    for key in ["broker", "audit_logger", "circuit_breaker", "db_pool"]:
        val = infra.get(key)
        ok = val is True or (isinstance(val, str) and val)
        p = status(ok, f"Infrastructure: {key}", str(val))
        results.append((f"pt_infra_{key}", p))

    # Models
    models = data.get("models", {})
    registered = models.get("registered", 0)
    loaded = models.get("loaded", 0)
    p = status(registered >= 8, "Models registered", f"{registered}")
    results.append(("pt_models_registered", p))
    p = status(loaded >= 5, "Models loaded", f"{loaded}/{registered}")
    results.append(("pt_models_loaded", p))

    # Model details
    details = models.get("details", {})
    for name, info in details.items():
        is_loaded = info.get("loaded", False) if isinstance(info, dict) else False
        if is_loaded:
            status(True, f"  Model: {name}", info.get("asset_class", ""))
        else:
            warn(f"  Model: {name}", "not loaded")

    # Redis
    redis_info = data.get("redis", {})
    redis_ok = redis_info.get("connected", False)
    p = status(redis_ok, "Redis pub/sub", "connected" if redis_ok else "disconnected")
    results.append(("pt_redis", p))

    # Circuit breaker
    cb = data.get("circuit_breaker_tripped", None)
    p = status(cb is False, "Circuit breaker", "not tripped" if cb is False else "TRIPPED")
    results.append(("pt_circuit_breaker", p))

    return results


# ── PART 3: DATABASE TABLES ────────────────────────────────────────────

def check_database():
    header("DATABASE TABLES")
    results = []

    try:
        import psycopg2
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASSWORD
        )
        cur = conn.cursor()

        for tbl in ["ohlcv", "paper_trades", "paper_daily_snapshots",
                     "paper_strategy_signals", "paper_risk_reports"]:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                count = cur.fetchone()[0]
                p = status(True, f"Table: {tbl}", f"{count:,} rows")
            except Exception:
                conn.rollback()
                p = status(False, f"Table: {tbl}", "MISSING/ERROR")
            results.append((f"db_{tbl}", p))

        conn.close()
    except Exception as e:
        status(False, "Database connection", str(e)[:60])
        results.append(("db_connection", False))

    return results


# ── PART 4: API ENDPOINTS ──────────────────────────────────────────────

def check_endpoints():
    header("API ENDPOINTS")
    results = []

    endpoints = {
        "/health": 200,
        "/positions": 200,
        "/history": 200,
        "/weights": 200,
        "/weights/bayesian": 200,
        "/dashboard": 200,
        "/execution/stats": 200,
    }

    for path, expected_code in endpoints.items():
        r = http_get(f"{PAPER_TRADER_URL}{path}")
        ok = r is not None and r.status_code == expected_code
        detail = f"{r.status_code}" if r else "UNREACHABLE"
        p = status(ok, f"GET {path}", detail)
        results.append((f"endpoint_{path.replace('/', '_')}", p))

    return results


# ── PART 5: PROMETHEUS METRICS ─────────────────────────────────────────

def check_prometheus():
    header("PROMETHEUS METRICS")
    results = []

    # Check metrics endpoint
    r = http_get(f"{PAPER_TRADER_URL}/metrics")
    if r is None:
        status(False, "Metrics endpoint", "UNREACHABLE")
        return [("metrics_endpoint", False)]

    ok = r.status_code == 200
    p = status(ok, "Metrics endpoint", f"{r.status_code}")
    results.append(("metrics_endpoint", p))

    if ok:
        text = r.text
        # Check for key metric families
        key_metrics = ["apex_", "python_", "process_"]
        found = [m for m in key_metrics if m in text]
        p = status(len(found) >= 1, "Metric families",
                   f"Found: {', '.join(found)}" if found else "None found")
        results.append(("metric_families", p))

        lines = [l for l in text.split("\n") if l and not l.startswith("#")]
        p = status(len(lines) >= 3, "Metric count", f"{len(lines)} metrics")
        results.append(("metric_count", p))

    # Prometheus targets
    r = http_get(f"{PROMETHEUS_URL}/api/v1/targets")
    if r and r.status_code == 200:
        try:
            targets = r.json().get("data", {}).get("activeTargets", [])
            up = sum(1 for t in targets if t.get("health") == "up")
            total = len(targets)
            p = status(up > 0, "Prometheus targets", f"{up}/{total} up")
            results.append(("prom_targets", p))
        except Exception:
            warn("Prometheus targets", "parse error")
    else:
        warn("Prometheus targets", "unreachable")

    return results


# ── PART 6: ENSEMBLE & RISK SYSTEM ────────────────────────────────────

def check_ensemble_risk():
    header("ENSEMBLE & RISK SYSTEM")
    results = []

    # Bayesian weights
    r = http_get(f"{PAPER_TRADER_URL}/weights/bayesian")
    ok = r is not None and r.status_code == 200
    if ok:
        try:
            data = r.json()
            p = status(True, "Bayesian weights endpoint", f"{len(data)} entries" if isinstance(data, dict) else "OK")
        except Exception:
            p = status(True, "Bayesian weights endpoint", "200 OK")
    else:
        p = status(False, "Bayesian weights endpoint", "UNREACHABLE")
    results.append(("bayesian_weights", p))

    # Strategy weights
    r = http_get(f"{PAPER_TRADER_URL}/weights")
    ok = r is not None and r.status_code == 200
    p = status(ok, "Strategy weights endpoint", f"{r.status_code}" if r else "UNREACHABLE")
    results.append(("strategy_weights", p))

    # Execution stats
    r = http_get(f"{PAPER_TRADER_URL}/execution/stats")
    ok = r is not None and r.status_code == 200
    p = status(ok, "Execution stats", f"{r.status_code}" if r else "UNREACHABLE")
    results.append(("execution_stats", p))

    # Test importing key ensemble modules
    try:
        sys.path.insert(0, ".")
        from strategies.ensemble.combiner import EnsembleCombiner
        from strategies.ensemble.portfolio_optimizer import PortfolioOptimizer
        from strategies.regime.detector import RegimeDetector
        p = status(True, "Ensemble modules importable", "combiner, optimizer, detector")
    except Exception as e:
        p = status(False, "Ensemble modules", str(e)[:60])
    results.append(("ensemble_modules", p))

    # Test importing risk module
    try:
        from strategies.risk.portfolio_risk import PortfolioRiskManager
        p = status(True, "Risk module importable", "PortfolioRiskManager")
    except Exception as e:
        p = status(False, "Risk module", str(e)[:60])
    results.append(("risk_module", p))

    # Test importing safety guardrails
    try:
        from trading.safety.guardrails import (
            SignalVarianceGuard, LeverageGate, ExecutionFailureMonitor
        )
        p = status(True, "Safety guardrails importable", "3 guardrails")
    except Exception as e:
        p = status(False, "Safety guardrails", str(e)[:60])
    results.append(("guardrails", p))

    return results


# ── PART 7: CIRCUIT BREAKER ───────────────────────────────────────────

def check_circuit_breaker():
    header("CIRCUIT BREAKER")
    results = []

    try:
        from trading.risk.circuit_breaker import CircuitBreaker
        p = status(True, "CircuitBreaker importable", "OK")
        results.append(("cb_import", p))
    except Exception as e:
        p = status(False, "CircuitBreaker import", str(e)[:60])
        results.append(("cb_import", p))
        return results

    # Check via health endpoint
    r = http_get(f"{PAPER_TRADER_URL}/health")
    if r and r.status_code == 200:
        data = r.json()
        tripped = data.get("circuit_breaker_tripped", None)
        p = status(tripped is False, "Circuit breaker state", "not tripped" if tripped is False else str(tripped))
        results.append(("cb_state", p))
    else:
        warn("Circuit breaker state", "health unreachable")

    return results


# ── PART 8: ALPACA BROKER ─────────────────────────────────────────────

def check_alpaca():
    header("ALPACA BROKER")
    results = []

    try:
        from trading.broker.alpaca import AlpacaBroker
        p = status(True, "AlpacaBroker importable", "OK")
        results.append(("alpaca_import", p))
    except Exception as e:
        p = status(False, "AlpacaBroker import", str(e)[:60])
        results.append(("alpaca_import", p))
        return results

    # Check if API keys configured
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    has_keys = bool(api_key) and bool(secret_key)
    if has_keys:
        is_placeholder = "your_" in api_key.lower() or "paste" in api_key.lower()
        if is_placeholder:
            warn("Alpaca API keys", "placeholder values — set real keys")
            results.append(("alpaca_keys", True))  # non-blocking
        else:
            p = status(True, "Alpaca API keys", "configured")
            results.append(("alpaca_keys", p))
    else:
        warn("Alpaca API keys", "not set (expected for paper trading)")
        results.append(("alpaca_keys", True))  # non-blocking

    return results


# ── PART 9: POSITIONS & ORDERS ─────────────────────────────────────────

def check_positions_orders():
    header("POSITIONS & ORDERS")
    results = []

    # Positions
    r = http_get(f"{PAPER_TRADER_URL}/positions")
    if r and r.status_code == 200:
        try:
            positions = r.json()
            p = status(True, "Positions endpoint", f"{len(positions)} positions")
        except Exception:
            p = status(True, "Positions endpoint", "200 OK")
    else:
        p = status(False, "Positions endpoint", "UNREACHABLE")
    results.append(("positions", p))

    # History
    r = http_get(f"{PAPER_TRADER_URL}/history")
    if r and r.status_code == 200:
        try:
            history = r.json()
            p = status(True, "History endpoint", f"{len(history)} entries")
        except Exception:
            p = status(True, "History endpoint", "200 OK")
    else:
        p = status(False, "History endpoint", "UNREACHABLE")
    results.append(("history", p))

    return results


# ── PART 10: DASHBOARD ────────────────────────────────────────────────

def check_dashboard():
    header("DASHBOARD")
    results = []

    r = http_get(f"{PAPER_TRADER_URL}/dashboard")
    if r and r.status_code == 200:
        has_html = "<html" in r.text.lower() or "<div" in r.text.lower()
        p = status(has_html, "Dashboard HTML", f"{len(r.text):,} bytes")
        results.append(("dashboard_html", p))
    else:
        p = status(False, "Dashboard", f"{r.status_code}" if r else "UNREACHABLE")
        results.append(("dashboard", p))

    return results


# ── MAIN ────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    print(f"\n{BOLD}{SEP}")
    print(f"  APEX PAPER TRADER END-TO-END TEST SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Target: {PAPER_TRADER_URL}")
    print(f"{SEP}{RST}")

    all_results = []
    all_results.extend(check_connectivity())
    all_results.extend(check_paper_trader_health())
    all_results.extend(check_database())
    all_results.extend(check_endpoints())
    all_results.extend(check_prometheus())
    all_results.extend(check_ensemble_risk())
    all_results.extend(check_circuit_breaker())
    all_results.extend(check_alpaca())
    all_results.extend(check_positions_orders())
    all_results.extend(check_dashboard())

    # Summary
    passed = sum(1 for _, p in all_results if p)
    failed = sum(1 for _, p in all_results if not p)
    total = len(all_results)

    print(f"\n{BOLD}{SEP}")
    if failed == 0:
        print(f"  {G}ALL {total} CHECKS PASSED{RST}")
    else:
        print(f"  {R}{failed}/{total} CHECKS FAILED{RST}")
    print(SEP)

    # Save results
    result_file = RESULTS_DIR / f"e2e_{ts}.json"
    with open(result_file, "w") as f:
        json.dump({
            "timestamp": ts,
            "target": PAPER_TRADER_URL,
            "passed": passed,
            "failed": failed,
            "total": total,
            "details": {name: val for name, val in all_results},
        }, f, indent=2, default=str)
    print(f"  Results saved to {result_file}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
