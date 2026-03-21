# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

APEX is a multi-strategy algorithmic trading platform built around the Temporal Fusion Transformer (TFT). The main codebase lives in `TFT-main/`. It supports two data backends (legacy SQLite and recommended PostgreSQL), a multi-model ensemble system with **10 models** across 4 asset classes, **12 ensemble strategies**, a production paper-trading execution service with circuit breaker, audit trail, **5 automated safety guardrails**, dead letter queue with exponential backoff retry, and a microservices deployment layer with Kafka-based event streaming, schema registry, and TimescaleDB data retention policies. ~179 Python files, 635 tests across 30 test modules.

## Build & Run Commands

All commands run from `TFT-main/`.

### Initial Setup
```bash
./setup.sh                    # Full setup: venv, deps, dirs, config
pip install -r requirements.txt  # Manual dependency install
python postgres_schema.py     # Create PostgreSQL schema
cp .env.example .env          # Create env config (edit with real credentials)
```

### Training
```bash
# PostgreSQL-based (recommended)
python train_postgres.py --symbols AAPL GOOGL MSFT --start-date 2022-01-01 --target-type returns --max-epochs 50

# Legacy file-based
python train.py --data-source api --symbols AAPL GOOGL MSFT --start-date 2020-01-01 --target-type returns --max-epochs 50
```

### Predictions
```bash
python predict.py --model-path models/tft_model.pth --symbols AAPL GOOGL MSFT --prediction-method quintile --include-portfolio
```

### API Server
```bash
# PostgreSQL API
python -m uvicorn api_postgres:app --host 0.0.0.0 --port 8000 --reload

# Legacy API
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Paper Trader (multi-strategy ensemble)
```bash
cd paper-trader
python -m uvicorn main:app --host 0.0.0.0 --port 8010 --reload
```
The paper trader runs the full daily pipeline: data fetch -> regime detection -> all strategies -> Bayesian ensemble -> portfolio optimization -> Alpaca execution -> PostgreSQL logging -> Discord reports. Runs daily at 10:00 ET Mon-Fri (configurable). Dashboard at `http://localhost:8010/dashboard`.

### Scheduler
```bash
python scheduler.py --mode scheduler       # Automated scheduled runs
python scheduler.py --mode manual --task training  # Manual single task
```

### Docker (microservices)
```bash
docker-compose up              # All services
docker-compose up data-ingestion sentiment-engine  # Specific services
```

### Tests
```bash
pytest                                    # All tests
python test_end_to_end.py                # End-to-end pipeline test
python test_polygon_integration.py       # Polygon.io integration test
python test_service.py                   # Service-level test
./devtools/prompt_runner.sh --test-all   # Copilot prompt validation
```

### Linting
```bash
black .          # Format
flake8 .         # Lint
mypy .           # Type check
```

## Architecture

### Two Parallel Pipelines

The system has a **legacy pipeline** (SQLite/file-based) and a **PostgreSQL pipeline**. Each has its own set of modules:

| Layer | Legacy | PostgreSQL |
|-------|--------|------------|
| Data loading | `data_pipeline.py` (StockDataCollector -> SQLite) | `postgres_data_loader.py` (PostgresDataLoader -> psycopg2) |
| Preprocessing | `data_preprocessing.py` (StockDataPreprocessor) | `postgres_data_pipeline.py` (PostgresDataPipeline) |
| Model | `tft_model.py` (EnhancedTFTModel) | `tft_postgres_model.py` (TFTPostgresModel + AdvancedOptionsModel) |
| Training | `train.py` | `train_postgres.py` |
| API | `api.py` | `api_postgres.py` |

Both pipelines share: `stock_ranking.py` (signal generation + portfolio construction), `scheduler.py`, `config_manager.py`, `predict.py`.

### Multi-Strategy Ensemble System

The ensemble system runs independently from the base TFT pipeline and lives in `strategies/`, `models/`, and `paper-trader/`.

#### Model Layer (`models/`)

10 models managed by `ModelManager` with graceful fallback (missing models return empty predictions):

| # | Model | File | Asset Class | Training | Source |
|---|-------|------|-------------|----------|--------|
| 1 | TFT Stocks | `stocks_adapter.py` | stocks | Required | `models/tft_model.pth` |
| 2 | TFT Forex | `forex_model.py` | forex | Required | `models/tft_forex.pth` |
| 3 | TFT Volatility | `volatility_model.py` | volatility | Required | `models/tft_volatility.pth` |
| 4 | Kronos | `kronos_model.py` | stocks+forex | Pre-trained | HuggingFace `NeoQuasar/Kronos-base` |
| 5 | Deep Surrogates | `deep_surrogate_model.py` | options/vol | Pre-trained | `/opt/deep_surrogate` repo |
| 6 | TDGF | `tdgf_model.py` | options | Light training | `/opt/tdgf` repo |
| 7 | Sentiment | `sentiment_model.py` | cross-asset | Pre-trained | FinBERT (`ProsusAI/finbert`) + VADER fallback |
| 8 | Mean Reversion | `mean_reversion_model.py` | stocks | Statistical | Hurst exponent + OU parameter estimation |
| 9 | Macro Regime | `macro_model.py` | cross-asset | Rule-based | Yield curve + rate trends via yfinance |
| 10 | Microstructure | `microstructure_model.py` | stocks | Statistical | Volume profile + order flow analysis |

Key classes: `BaseTFTModel` (ABC), `ModelManager` (unified loader + prediction), `ModelPrediction` (standardized output).

#### Strategy Layer (`strategies/`)

Each strategy extends `BaseStrategy` (ABC in `strategies/base.py`) and produces `StrategyOutput` containing `List[AlphaScore]` -- z-scored per-symbol alpha signals.

| # | Strategy | File | Asset Class | Model Dependency |
|---|----------|------|-------------|------------------|
| 1 | Cross-Sectional Momentum | `momentum/cross_sectional.py` | stocks | None (factor-based) |
| 2 | Pairs Trading (StatArb) | `statarb/pairs.py` | stocks | None (cointegration) |
| 3 | Mean Reversion | `stocks/mean_reversion.py` | stocks | MeanReversionModel (#8), optionally MicrostructureModel (#10) |
| 4 | Sector Rotation | `stocks/sector_rotation.py` | stocks | MacroRegimeModel (#9) |
| 5 | FX Carry + Trend | `fx/carry_trend.py` | forex | None (carry + trend) |
| 6 | FX Momentum | `fx/momentum.py` | forex | None (multi-lookback trend) |
| 7 | FX Volatility Breakout | `fx/vol_breakout.py` | forex | TFT Volatility for vol forecast |
| 8 | Deep Surrogates | `deep_surrogates/strategy.py` | options/vol | DeepSurrogateModel via ModelManager |
| 9 | TDGF American Options | `tdgf/strategy.py` | options | TDGFModel via ModelManager |
| 10 | Vol Surface Arbitrage | `options/strategies/vol_arb.py` | options | None (IV-RV spread) |
| 11 | Kronos Forecasting | `kronos/strategy.py` | stocks+forex | KronosModel via ModelManager |
| 12 | Sentiment | `sentiment/strategy.py` | cross-asset | SentimentModel (#7) via ModelManager |

Supporting modules:
- `ensemble/combiner.py` -- Bayesian signal fusion with regime-adaptive weighting
- `ensemble/portfolio_optimizer.py` -- Risk-constrained portfolio construction
- `regime/detector.py` -- 4-state market regime classifier (calm/volatile x trending/choppy)
- `risk/portfolio_risk.py` -- Portfolio-level risk management (persistent, fed historical + live data)
- `signals/publisher.py` -- Redis pub/sub signal distribution (additive, not required)
- `validation/walk_forward.py` -- Walk-forward cross-validation engine (rolling window, embargo gap, per-fold normalization)
- `config.py` -- All strategy configs loaded from environment variables (includes `WalkForwardConfig`)

#### Ensemble Flow
1. All enabled strategies call `generate_signals(data)` -> `StrategyOutput`
2. `EnsembleCombiner.combine()` weights signals via Bayesian method (60% Sharpe-based + 40% regime-based)
3. Weights clamped to `[min_weight, max_weight]` and renormalized
4. **Risk assessment**: `PortfolioRiskManager.assess()` -- kill switch halts all trades, killed strategies filtered out, correlated strategies (>0.85) reduced by 50%
5. `PortfolioOptimizer.optimize()` applies position limits, leverage constraints, target vol
6. Paper trader executes via Alpaca, logs to PostgreSQL + `paper_risk_reports`

#### Regime -> Strategy Weight Mapping
Strategies map to 4 regime weight buckets in `combiner.py` -- `[momentum, mean_reversion, pairs, tft]`:
- `cross_sectional_momentum`, `sector_rotation`, `fx_momentum` -> `"momentum"`
- `mean_reversion` -> `"mean_reversion"`
- `pairs_trading`, `fx_vol_breakout`, `deep_surrogates`, `tdgf`, `vol_arb` -> `"pairs"`
- `fx_carry_trend`, `kronos`, `sentiment` -> `"tft"`

#### Strategy Activation
All strategies are **disabled by default**. Enable via `.env`:
```bash
STRATEGY_MOMENTUM_ENABLED=true
STRATEGY_STATARB_ENABLED=true
STRATEGY_MEAN_REVERSION_ENABLED=true
STRATEGY_SECTOR_ROTATION_ENABLED=true
STRATEGY_FX_ENABLED=true               # FX Carry + Trend
STRATEGY_FX_MOMENTUM_ENABLED=true
STRATEGY_FX_VOL_BREAKOUT_ENABLED=true
STRATEGY_KRONOS_ENABLED=true
STRATEGY_DEEP_SURROGATES_ENABLED=true
STRATEGY_TDGF_ENABLED=true
STRATEGY_VOL_ARB_ENABLED=true
STRATEGY_SENTIMENT_ENABLED=true
```

### Safety Guardrails (`trading/safety/guardrails.py`)

Five automated pre-trade safety checks wired into the paper-trader pipeline (added 2026-03-21 after March 10 incident analysis):

| # | Guardrail | Class | Env Var | Default | Pipeline Location |
|---|-----------|-------|---------|---------|-------------------|
| 1 | Signal Variance | `SignalVarianceGuard` | `GUARDRAIL_SIGNAL_MIN_STD` | 0.01 | After ensemble combine, before optimization |
| 2 | Leverage Gate | `LeverageGate` | `GUARDRAIL_MAX_LEVERAGE` | 1.5 | After optimization, before order batch |
| 3 | Calibration Health | `CalibrationHealthCheck` | `GUARDRAIL_CALIBRATION_TOLERANCE` | 1e-6 | At startup + before daily run |
| 4 | Model Promotion | `ModelPromotionGate` | `GUARDRAIL_MIN_PROMOTION_SHARPE` | 0.5 | Before model goes live |
| 5 | Execution Failure | `ExecutionFailureMonitor` | `GUARDRAIL_MAX_EXEC_FAILURE_RATE` / `GUARDRAIL_EXEC_WINDOW_SECONDS` | 0.25 / 3600 | Per-order during execution loop |

**Behavior on failure:**
- Signal Variance: halts pipeline entirely, sends Discord critical alert
- Leverage Gate: skips order batch, continues to snapshot/reporting
- Calibration Health: logs error, skips calibration (does not halt)
- Model Promotion: rejects model, logs warning
- Execution Failure: pauses remaining orders in batch, sends Discord alert

### Paper Trader (`paper-trader/main.py`)

FastAPI service on port 8010. Production-grade daily pipeline with full infrastructure:

0. **Startup: Environment validation** via `utils/env_validator.py` -- required vars checked, placeholder detection
1. Fetch OHLCV via yfinance (stocks + SPY for regime, FX pairs)
2. Circuit breaker check (Redis-backed drawdown limits)
3. Detect market regime
4. Build and run all enabled strategies (up to 12: momentum, pairs, mean reversion, sector rotation, FX carry/momentum/vol breakout, kronos, deep surrogates, TDGF, vol arb, sentiment)
5. Combine via Bayesian ensemble
6. **GUARDRAIL: Signal variance check** -- halt if scores collapse
7. **RISK ASSESSMENT**: `PortfolioRiskManager.assess()` -- kill switch halts all trades, killed strategies filtered, correlated strategies reduced, risk report logged to `paper_risk_reports`
8. Optimize portfolio with risk constraints
9. **GUARDRAIL: Leverage gate** -- skip orders if leverage > limit
10. **GUARDRAIL: Execution failure monitor** -- pause if failure rate spikes
11. Execute trades via production `AlpacaBroker` (from `trading/broker/alpaca.py`)
12. Audit trail logging via `AuditLogger` (from `trading/persistence/audit.py`)
13. Log trades, snapshots, and signals to PostgreSQL (connection pooled)
14. Feed daily return to persistent `PortfolioRiskManager`
15. Publish signals to Redis (optional, fire-and-forget)
16. Send reports via `NotificationManager` (Discord + Email, from `trading/notifications/alerts.py`)
17. LLM signal analysis via `SignalAnalyst` (Ollama, optional)
18. Serve live dashboard at `/dashboard` with panels for all 12 strategies

**Production infrastructure wired in:**
- `AlpacaBroker` (283 lines) replaces the simplified PaperBroker
- `CircuitBreaker` (405 lines) -- Redis-backed drawdown circuit breaker checked before every trade
- `AuditLogger` (277 lines) -- PostgreSQL audit trail for all pipeline events
- `NotificationManager` (202 lines) -- Discord + Email alert system
- `PortfolioRiskManager` (~500 lines) -- Persistent instance, seeded with 30 days historical returns at startup, fed live daily returns. VaR/CVaR, correlation monitoring (>0.85 triggers 50% weight reduction), per-strategy kill switches (drawdown/Sharpe), portfolio-level kill switch. Reports logged to `paper_risk_reports`
- `ThreadedConnectionPool` -- PostgreSQL connection pooling (2-10 connections)
- `SignalVarianceGuard` / `LeverageGate` / `ExecutionFailureMonitor` -- Safety guardrails (`trading/safety/guardrails.py`)
- `DeadLetterQueue` -- PostgreSQL-backed DLQ with exponential backoff retry (`services/common/dlq.py`)
- `EnvValidator` -- Startup environment validation (`utils/env_validator.py`)

- `SignalAnalyst` -- LLM-powered post-pipeline analysis via Ollama (`agents/signal_analyst.py`)
- `PrometheusMetrics` -- Signal, weight, regime, risk, and execution metrics (`monitoring/metrics.py`)
- Signal Provider REST API -- Public API with API key auth, rate limiting, ETag caching (`api/signal_provider.py`)

Endpoints: `/health` (10 models, 12 strategies, infrastructure status), `/run-now` (manual trigger), `/positions`, `/history`, `/weights`, `/dashboard`, `/dlq` (dead letter queue stats), `/metrics` (Prometheus), `/api/v1/signals` (signal provider API).

Database: PostgreSQL on port **5432**, database **`tft_trading`** (tables: `paper_trades`, `paper_daily_snapshots`, `paper_strategy_signals`, `paper_risk_reports`).

### Core Flow (Original TFT Pipeline)
1. **Data ingestion** -- Polygon.io OHLCV, Reddit sentiment, fundamentals -> database
2. **Preprocessing** -- Technical indicators (RSI, MACD, Bollinger Bands), temporal features, normalization
3. **TFT model** -- PyTorch Forecasting `TemporalFusionTransformer` with quantile loss, attention, multi-horizon forecasting
4. **Signal generation** -- `StockRankingSystem` ranks predictions -> `PortfolioConstructor` builds long/short portfolios with risk constraints
5. **API serving** -- FastAPI endpoints for `/predict`, `/train`, `/health`

### Microservices Layer (`microservices/`)

Five FastAPI services coordinated via Kafka topics and Redis caching:

| Service | Port | Role |
|---------|------|------|
| `data-ingestion` | 8001 | Polygon.io + Reddit data -> Kafka |
| `sentiment-engine` | 8002 | NLP sentiment scoring (FinBERT/VADER) |
| `tft-predictor` | 8003 | GPU inference, MLflow model versioning |
| `trading-engine` | 8004 | Alpaca paper/live order execution |
| `orchestrator` | 8005 | Saga-pattern workflow coordination |

Infrastructure: TimescaleDB (PostgreSQL 15), Redis, Kafka (KRaft mode with retention policies), Schema Registry (Confluent), Prometheus, Grafana, MLflow. See `docker-compose.yml`.

All 4 Kafka consumers have DLQ integration -- failed messages are persisted to PostgreSQL with exponential backoff retry (base 1s, 2x multiplier, max 5 retries, 0-25% jitter).

### Key Kafka Topics
`market-data`, `sentiment-scores`, `tft-predictions`, `trading-signals`, `order-updates`, `portfolio-updates`, `system-events`

### Redis Pub/Sub Channels (additive signal layer)
`apex:signals:stock`, `apex:signals:forex`, `apex:signals:options`, `apex:signals:risk`

### Configuration

- Environment: `.env` (copy from `.env.template` for full config, `.env.example` for minimal)
- Model/trading params: `config_manager.py` with `TFTConfig` and `TradingConfig` dataclasses
- Strategy params: `strategies/config.py` with `StrategyMasterConfig.from_env()`
- JSON config: `config/default_config.json` (created by `setup.sh`)

### Key External Dependencies
- **Polygon.io** -- Primary market data API (OHLCV, news, fundamentals, options). Rate limited: 5 req/min on free tier
- **Alpaca** -- Paper/live trading execution
- **Reddit (PRAW)** -- Sentiment data from financial subreddits
- **pytorch-forecasting** -- TFT model implementation
- **MLflow** -- Experiment tracking and model registry
- **Kronos** -- Pre-trained K-line foundation model (`/opt/kronos`, HuggingFace auto-download)
- **DeepSurrogate** -- Neural option pricing surrogate (`/opt/deep_surrogate`)
- **TDGF** -- PDE solver for American options (`/opt/tdgf`)

### Data Directories
`data/` (raw data + SQLite), `models/` (trained `.pth` files + preprocessors), `predictions/`, `logs/`, `reports/`, `output/`

## Known Bugs (audited 2026-03-21)

### All 17 bugs resolved

| Bug | Status | Resolution |
|-----|--------|------------|
| CF-1/CF-2/CF-4 | Fixed | Walk-forward cross-validation system (`strategies/validation/walk_forward.py`) |
| CF-3 | Fixed | Frequency-aware Sharpe annualization |
| CF-5 | Fixed | CVaR-95 (Expected Shortfall) added |
| CF-6 | Fixed | Circuit breaker lifespan + fail-closed on Redis failure |
| CF-7 | Fixed | All 4 Kafka consumers: `enable_auto_commit=False` + explicit commit + DLQ |
| CF-8 | Fixed | Alpaca `ClientSession` timeout aligned with `_api_call()` timeout |
| CF-9 | Fixed | Shutdown `asyncio.wait_for(timeout=30)` + SIGTERM/SIGINT handlers |
| CF-10 | Fixed | Duplicate docker-compose service definitions removed |
| HI-1 | Fixed | `PortfolioRiskManager` wired as persistent global in paper-trader |
| HI-4 | Fixed | Bear regime `exposure_scalar` only scales longs (shorts preserved) |
| HI-5 | Fixed | Scheduler hours sourced from environment variables |
| HI-8 | Fixed | Docker healthcheck directives added for Redis |
| HI-3 | N/A | Platt calibration does not exist in codebase |
| Bug-A | N/A | limits.yaml does not exist in codebase |
| Bug-B | N/A | lean_alpha/signal_engine modules do not exist |

## Production Hardening (v3.0.0)

Added in Phase 04 (2026-03-21):

### Data Retention & Schema Registry
- **Kafka broker** -- KRaft mode (no Zookeeper), 168h log retention, 5GB max per topic, 1GB segments
- **TimescaleDB retention policies** -- 365d for OHLCV/snapshots, 90d for signals/risk/execution
- **TimescaleDB continuous aggregates** -- 15m, 1h, 1d OHLCV rollup views with automatic refresh
- **Schema Registry** -- Confluent Schema Registry with thread-safe singleton connection cache, exponential backoff (max 3 retries)

### Security Hardening
- All hardcoded user home directory paths removed
- All default passwords removed -- replaced with `os.environ[]` (raises on missing)
- Real API keys removed from `setup_postgres.sh` (Polygon, Alpaca, Reddit)
- `utils/env_validator.py` -- startup validation of required env vars with placeholder detection
- docker-compose uses `${VAR:?error}` for required credentials
- `.env.example` has safe placeholders only

### Dead Letter Queue
- `services/common/dlq.py` -- PostgreSQL-persisted DLQ with status lifecycle (PENDING -> RETRYING -> RESOLVED/EXHAUSTED)
- Exponential backoff: base 1s, 2x multiplier, max 5 retries, max 60s delay, 0-25% jitter
- Integrated into all 4 Kafka consumers (sentiment-engine, trading-engine, tft-predictor, orchestrator)
- Background retry worker thread (polls every 30s)
- `/dlq` dashboard endpoint in paper-trader

## Notes

- The docker-compose references `tft_network` as an external network -- create it with `docker network create tft_network` before running.
- PostgreSQL env vars differ between `.env.template` (`DB_HOST`, `DB_PORT`, etc.) and `train_postgres.py` (`POSTGRES_HOST`, `POSTGRES_PORT`, etc.). Check which convention the target module expects. The paper trader uses `DB_HOST`/`DB_PORT` convention.
- The `tft_postgres_model.py` contains an `AdvancedOptionsModel` class with Black-Scholes pricing that is separate from the TFT model itself.
- Kronos, Deep Surrogates, and TDGF models require their repos cloned to `/opt/`. If not present, strategies gracefully return empty signals and the ensemble skips them.
- The paper trader's `/health` endpoint reports which models are loaded and Redis connection status.
- Deep Surrogates includes a `calibrate_heston()` method with multi-start optimization and Feller condition validation.
- All credentials must be set via environment variables -- no defaults exist. The paper trader calls `env_validator.validate(strict=True)` at startup.
- The `dead_letter_queue` table is auto-created by `postgres_schema.py` and by each microservice's DLQ instance on first use.
