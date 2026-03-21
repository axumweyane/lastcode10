# APEX v3.0.0 Release Notes

**Release Date**: 2026-03-21
**Codename**: Production Hardening

---

## Summary

APEX v3.0.0 is a production hardening release focused on three areas: data retention and schema registry, security hardening, and fault-tolerant message processing. All 17 audited bugs from the March 2026 audit are now resolved.

### By the Numbers

| Metric | v2.1.0 | v3.0.0 |
|--------|--------|--------|
| Python files | ~145 | ~179 |
| Tests | 114 | 635 |
| Test files | 12 | 30 |
| Bugs remaining | 3 | 0 |
| Kafka consumers with DLQ | 0 | 4 |
| TimescaleDB retention policies | 0 | 4 |
| Continuous aggregates | 0 | 3 |
| Hardcoded passwords | 18 files | 0 |
| Hardcoded paths | 1 file | 0 |
| Leaked API keys | 1 file | 0 |

---

## What's New

### 1. Kafka Broker & Data Retention

Previously, the docker-compose referenced Kafka but did not include a broker service. v3.0.0 adds a full KRaft-mode Kafka broker (no Zookeeper dependency) with configurable log retention.

- **Broker**: Confluent CP 7.6.0, KRaft mode
- **Retention**: 168 hours (7 days), 5GB max per topic, 1GB segments
- **Schema Registry**: Confluent Schema Registry with thread-safe singleton connection cache and exponential backoff on connection failure

### 2. TimescaleDB Hypertables & Aggregates

Four tables converted to TimescaleDB hypertables for time-series optimization. Automatic retention policies drop old data to prevent unbounded storage growth.

**Retention Policies:**
- `ohlcv_bars`: 365 days
- `paper_risk_reports`, `paper_execution_stats`, `paper_signal_analyses`: 90 days

**Continuous Aggregates:**
- `ohlcv_15m` — 15-minute OHLCV rollup (refreshed every hour)
- `ohlcv_1h` — Hourly OHLCV rollup (refreshed every 2 hours)
- `ohlcv_1d` — Daily OHLCV rollup (refreshed daily)

### 3. Dead Letter Queue

Failed Kafka messages are now persisted to PostgreSQL instead of being silently dropped. Each message is retried with exponential backoff until resolved or exhausted.

- **Status lifecycle**: PENDING -> RETRYING -> RESOLVED | EXHAUSTED
- **Backoff**: 1s base, 2x multiplier, max 60s, 0-25% jitter
- **Max retries**: 5 (configurable)
- **Retry worker**: Background daemon thread, polls every 30s
- **Monitoring**: `/dlq` endpoint on paper-trader shows per-service stats

**Integrated services**: sentiment-engine, trading-engine, tft-predictor, orchestrator

### 4. Security Hardening

All hardcoded credentials have been removed from the codebase. The system now fails fast on startup if required credentials are missing or contain placeholder values.

**Changes:**
- 18 Python files: `os.getenv('PASSWORD', 'default')` replaced with `os.environ['PASSWORD']`
- docker-compose: `${VAR:?error_message}` for required credentials
- `setup_postgres.sh`: Real API keys replaced with placeholders
- Training scripts: Hardcoded username `kibrom` and password `***` removed
- `create_dashboard.py`: Hardcoded user home directory path removed

**Startup Validation** (`utils/env_validator.py`):
- Required: `DB_PASSWORD`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- Detects placeholder patterns: `your_*`, `CHANGE_ME`, `changeme`, `xxx`
- Exits with error if required vars missing or contain placeholders

---

## Bug Fixes

Three previously unresolved bugs from the March 2026 audit are now fixed:

| Bug | Issue | Fix |
|-----|-------|-----|
| CF-8 | Alpaca `ClientSession()` had no default timeout despite `_api_call()` using 15s | Timeout aligned |
| HI-5 | `model_trainer.py` hardcoded schedule hours | Hours sourced from env vars |
| HI-8 | Redis had no Docker healthcheck directive | Healthcheck added |

---

## New Files

| File | Purpose |
|------|---------|
| `services/common/dlq.py` | Dead Letter Queue with exponential backoff |
| `services/__init__.py` | Package init |
| `services/common/__init__.py` | Package init |
| `microservices/schema_registry.py` | Schema Registry connection cache |
| `microservices/__init__.py` | Package init |
| `utils/env_validator.py` | Startup environment validation |
| `utils/__init__.py` | Package init |
| `tests/test_production_hardening.py` | 38 tests for retention & registry |
| `tests/test_security_hardening.py` | 22 tests for security |
| `tests/test_dlq.py` | 39 tests for DLQ |

---

## Migration Guide

### From v2.1.0 to v3.0.0

1. **Set required environment variables** — The system no longer has default passwords. Before starting any service, ensure these are set:
   ```bash
   export DB_PASSWORD=your_secure_password
   export ALPACA_API_KEY=your_key
   export ALPACA_SECRET_KEY=your_secret
   export POSTGRES_PASSWORD=your_secure_password  # for train_postgres.py
   ```

2. **Update docker-compose environment** — Set these in your shell or `.env` before `docker-compose up`:
   ```bash
   export POSTGRES_PASSWORD=your_secure_password
   export POSTGRES_USER=your_user
   export GRAFANA_ADMIN_PASSWORD=your_password
   ```

3. **Run schema migration** — The new DLQ table and TimescaleDB extensions are created automatically:
   ```bash
   python postgres_schema.py
   ```

4. **Verify startup** — The paper trader now validates environment on startup. If you see `Missing required environment variables`, set them before retrying.

---

## Test Coverage

| Test Module | Tests | Coverage Area |
|-------------|-------|---------------|
| test_production_hardening.py | 38 | Kafka retention, TimescaleDB, schema registry |
| test_security_hardening.py | 22 | Paths, passwords, secrets, env validator |
| test_dlq.py | 39 | Backoff, persistence, retry, status, integration |
| (existing 30 test files) | 536 | Models, strategies, integration, guardrails, etc. |
| **Total** | **635** | |

---

*APEX v3.0.0 | Production Hardening Release | 2026-03-21*
