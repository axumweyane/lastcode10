# APEX Trading System — Complete Documentation

*Multi-Strategy Algorithmic Trading Platform*
*Last Updated: 2026-03-21 | Version 3.0.0*

---

## System Overview

APEX is a production-grade multi-strategy algorithmic trading platform built around the Temporal Fusion Transformer (TFT). It combines **10 AI/statistical models** across 4 asset classes through **12 trading strategies**, fused via a Bayesian ensemble with regime-adaptive weighting, protected by **5 automated safety guardrails**, and executed through a paper-trading service connected to Alpaca.

### Key Metrics

| Metric | Value |
|--------|-------|
| Python files | ~179 |
| Models | 10 (6 deep learning, 4 statistical/rule-based) |
| Strategies | 12 (stocks, FX, options, cross-asset) |
| Safety guardrails | 5 (pre-trade checks) |
| Tests | 635 across 30 test modules |
| Asset classes | Stocks, Forex, Options/Volatility, Cross-Asset |
| Data retention | TimescaleDB hypertables + continuous aggregates |
| Message recovery | Dead Letter Queue with exponential backoff |
| Security | Env-only credentials, startup validation, no hardcoded secrets |
| Monitoring | Prometheus metrics, Grafana dashboards, LLM signal analysis |
| External API | Signal Provider REST API with API key auth and rate limiting |
| CI/CD | GitHub Actions: lint, test, security scan, Docker build |

---

## Three Independent Layers

### 1. Core TFT Pipeline

Train TFT models, generate predictions, rank stocks, construct portfolios. Two backends: legacy SQLite and PostgreSQL.

| Layer | Legacy | PostgreSQL |
|-------|--------|------------|
| Data loading | `data_pipeline.py` | `postgres_data_loader.py` |
| Preprocessing | `data_preprocessing.py` | `postgres_data_pipeline.py` |
| Model | `tft_model.py` | `tft_postgres_model.py` |
| Training | `train.py` | `train_postgres.py` |
| API | `api.py` (port 8000) | `api_postgres.py` (port 8000) |

### 2. Multi-Strategy Ensemble

11 strategies producing alpha signals, combined via Bayesian weighting, optimized into a risk-constrained portfolio, executed daily via the paper trader.

### 3. Microservices Layer

5 FastAPI services coordinated via Kafka for distributed deployment (Docker/K8s).

---

## Model Layer (10 Models)

All models extend `BaseTFTModel` (ABC in `models/base.py`) and return `List[ModelPrediction]`. `ModelManager` (`models/manager.py`) loads all models with graceful fallback — missing models return empty predictions.

| # | Model | Class | File | Asset Class | Type |
|---|-------|-------|------|-------------|------|
| 1 | TFT Stocks | `TFTStocksAdapter` | `stocks_adapter.py` | stocks | Deep learning |
| 2 | TFT Forex | `TFTForexModel` | `forex_model.py` | forex | Deep learning |
| 3 | TFT Volatility | `TFTVolatilityModel` | `volatility_model.py` | volatility | Deep learning |
| 4 | Kronos | `KronosModel` | `kronos_model.py` | stocks+forex | Pre-trained (HuggingFace) |
| 5 | Deep Surrogates | `DeepSurrogateModel` | `deep_surrogate_model.py` | options/vol | Pre-trained |
| 6 | TDGF | `TDGFModel` | `tdgf_model.py` | options | Light training |
| 7 | Sentiment | `SentimentModel` | `sentiment_model.py` | cross-asset | Pre-trained (FinBERT/VADER) |
| 8 | Mean Reversion | `MeanReversionModel` | `mean_reversion_model.py` | stocks | Statistical |
| 9 | Macro Regime | `MacroRegimeModel` | `macro_model.py` | cross-asset | Rule-based |
| 10 | Microstructure | `MicrostructureModel` | `microstructure_model.py` | stocks | Statistical |

### Model Details (Models 7-10)

**Sentiment Model (#7)** — Tries FinBERT (`ProsusAI/finbert`) first, falls back to VADER. Accepts DataFrame with `symbol` and `text` columns. Returns: sentiment_score, sentiment_momentum, sentiment_dispersion, article_count.

**Mean Reversion Model (#8)** — Computes Hurst exponent via R/S analysis and fits Ornstein-Uhlenbeck parameters (mu, theta, sigma) via OLS on log prices. Returns: hurst_exponent, half_life, deviation_zscore, mr_probability.

**Macro Regime Model (#9)** — Fetches yield curve data (^TNX, ^FVX, ^IRX, UUP) via yfinance. Classifies into 5 regimes: steepening_rising, steepening_falling, flattening_rising, inverted, neutral. Maps to sector tilts via `REGIME_SECTOR_TILTS` dict covering 40+ symbols.

**Microstructure Model (#10)** — Daily-frequency analysis: relative_volume, VWAP deviation, Close Location Value (CLV), buying_pressure, volume_trend, Accumulation/Distribution line. Composite signal = weighted combination.

### Deep Surrogates Tail Risk System

`DeepSurrogateModel` includes a Heston parameter calibration system with multi-start optimization and Feller condition validation. The tail risk index is broadcast via Redis pub/sub channel `apex:signals:risk` and displayed on the paper trader dashboard.

---

## Strategy Layer (12 Strategies)

All strategies extend `BaseStrategy` (ABC in `strategies/base.py`) and produce `StrategyOutput` containing `List[AlphaScore]` — z-scored per-symbol alpha signals.

| # | Strategy | File | Asset Class | Model Dependency |
|---|----------|------|-------------|------------------|
| 1 | Cross-Sectional Momentum | `momentum/cross_sectional.py` | stocks | None |
| 2 | Pairs Trading (StatArb) | `statarb/pairs.py` | stocks | None |
| 3 | Mean Reversion | `stocks/mean_reversion.py` | stocks | MeanReversionModel, MicrostructureModel |
| 4 | Sector Rotation | `stocks/sector_rotation.py` | stocks | MacroRegimeModel |
| 5 | FX Carry + Trend | `fx/carry_trend.py` | forex | None |
| 6 | FX Momentum | `fx/momentum.py` | forex | None |
| 7 | FX Volatility Breakout | `fx/vol_breakout.py` | forex | TFT Volatility |
| 8 | Deep Surrogates | `deep_surrogates/strategy.py` | options/vol | DeepSurrogateModel |
| 9 | TDGF American Options | `tdgf/strategy.py` | options | TDGFModel |
| 10 | Vol Surface Arbitrage | `options/strategies/vol_arb.py` | options | None |
| 11 | Kronos Forecasting | `kronos/strategy.py` | stocks+forex | KronosModel |
| 12 | Sentiment | `sentiment/strategy.py` | cross-asset | SentimentModel (#7) |

**Sentiment Strategy (#12)** — Contrarian/momentum signals from NLP sentiment. When sentiment diverges from price action, applies 1.5x multiplier (contrarian signal). When aligned, uses 0.8x multiplier. Trend window: 5 days. Maps to `"tft"` regime weight bucket.

### Strategy Activation

All strategies disabled by default. Enable via `.env`:

```bash
STRATEGY_MOMENTUM_ENABLED=true
STRATEGY_STATARB_ENABLED=true
STRATEGY_MEAN_REVERSION_ENABLED=true
STRATEGY_SECTOR_ROTATION_ENABLED=true
STRATEGY_FX_ENABLED=true                # FX Carry + Trend
STRATEGY_FX_MOMENTUM_ENABLED=true
STRATEGY_FX_VOL_BREAKOUT_ENABLED=true
STRATEGY_KRONOS_ENABLED=false            # needs /opt/kronos
STRATEGY_DEEP_SURROGATES_ENABLED=false   # needs /opt/deep_surrogate
STRATEGY_TDGF_ENABLED=false              # needs /opt/tdgf
STRATEGY_VOL_ARB_ENABLED=false
STRATEGY_SENTIMENT_ENABLED=true
```

---

## Ensemble System

### Bayesian Combiner (`strategies/ensemble/combiner.py`)

Three weighting methods: **equal**, **sharpe**, **bayesian** (default).

**Regime blending formula:**
```
final_weight = 0.6 x performance_weight + 0.4 x regime_weight
```

Weights clamped to [0.05, 0.40] and renormalized.

### Regime Detection (`strategies/regime/detector.py`)

4-state market classifier:

|  | Trending | Choppy |
|--|----------|--------|
| **Calm** (VIX < 20) | Favor momentum | Favor mean-reversion |
| **Volatile** (VIX > 20) | Reduce exposure | Maximum caution |

### Portfolio Optimization (`strategies/ensemble/portfolio_optimizer.py`)

Score -> raw weights -> inverse-vol adjustment -> regime scaling -> hard constraints (position caps, leverage caps, min position) -> VaR99 check.

---

## Safety Guardrails (`trading/safety/guardrails.py`)

Five automated pre-trade safety checks added 2026-03-21 to prevent repeat of March 10 incident:

| # | Guardrail | Class | Env Var | Default | Failure Action |
|---|-----------|-------|---------|---------|----------------|
| 1 | Signal Variance | `SignalVarianceGuard` | `GUARDRAIL_SIGNAL_MIN_STD` | 0.01 | Halt pipeline, Discord critical alert |
| 2 | Leverage Gate | `LeverageGate` | `GUARDRAIL_MAX_LEVERAGE` | 1.5 | Skip order batch |
| 3 | Calibration Health | `CalibrationHealthCheck` | `GUARDRAIL_CALIBRATION_TOLERANCE` | 1e-6 | Log error, skip calibration |
| 4 | Model Promotion | `ModelPromotionGate` | `GUARDRAIL_MIN_PROMOTION_SHARPE` | 0.5 | Reject model |
| 5 | Execution Failure | `ExecutionFailureMonitor` | `GUARDRAIL_MAX_EXEC_FAILURE_RATE` | 0.25 | Pause orders, Discord alert |

### Calibration Health Check

Verifies Platt scalers and isotonic calibrators are actually fitted (not identity parameters). Checks:
- `check_platt(a, b)` — Detects unfitted (None params) and identity (A=-1, B=0) states
- `check_generic(calibrator)` — sklearn-style fitted check via `classes_`, `calibrators_`, etc.

Runs at startup on all loaded models and can be called before each daily run.

---

## Paper Trader Service

`paper-trader/main.py` — FastAPI service on port 8010.

### Daily Pipeline (18 steps)

1. Fetch OHLCV via yfinance (stocks + SPY + FX pairs, 300 days)
2. Circuit breaker pre-check (Redis-backed)
3. Detect market regime
4. Build and run all enabled strategies (up to 11)
5. Portfolio risk assessment
6. Combine via Bayesian ensemble
7. **GUARDRAIL: Signal variance check**
8. Publish signals to Redis
9. Optimize portfolio with risk constraints
10. **GUARDRAIL: Leverage gate**
11. **GUARDRAIL: Execution failure monitor** (per-order)
12. Execute trades via Alpaca
13. Audit trail logging
14. Log to PostgreSQL
15. Send Discord/Email reports
16. Serve live dashboard

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System status: models, strategies, infrastructure |
| POST | `/run-now` | Manually trigger daily pipeline |
| GET | `/positions` | Current portfolio positions |
| GET | `/history` | Last 30 pipeline run records |
| GET | `/weights` | Current strategy weight distribution |
| GET | `/dashboard` | Live HTML dashboard |
| GET | `/dlq` | Dead letter queue statistics |

### Production Infrastructure

| Component | Class | File | Lines |
|-----------|-------|------|-------|
| Broker | `AlpacaBroker` | `trading/broker/alpaca.py` | 282 |
| Circuit Breaker | `CircuitBreaker` | `trading/risk/circuit_breaker.py` | 404 |
| Audit Logger | `AuditLogger` | `trading/persistence/audit.py` | 276 |
| Notifications | `NotificationManager` | `trading/notifications/alerts.py` | 201 |
| Risk Manager | `PortfolioRiskManager` | `strategies/risk/portfolio_risk.py` | 478 |
| Safety Guardrails | 5 classes | `trading/safety/guardrails.py` | 315 |
| Position Sizing | `PositionSizerFactory` | `trading/risk/position_sizing.py` | — |

---

## Microservices Layer

Five FastAPI services coordinated via Kafka topics and Redis caching:

| Service | Port | Role |
|---------|------|------|
| `data-ingestion` | 8001 | Polygon.io + Reddit data -> Kafka |
| `sentiment-engine` | 8002 | NLP sentiment scoring (FinBERT/VADER) |
| `tft-predictor` | 8003 | GPU inference, MLflow model versioning |
| `trading-engine` | 8004 | Alpaca paper/live order execution |
| `orchestrator` | 8005 | Saga-pattern workflow coordination |

Infrastructure: TimescaleDB (PostgreSQL 15), Redis, Kafka (KRaft mode with retention policies), Schema Registry (Confluent), Prometheus, Grafana, MLflow. All 4 Kafka consumers have DLQ integration with exponential backoff retry. See `docker-compose.yml`.

---

## TFT Model Configuration

```python
# Actual defaults from config_manager.py
max_encoder_length: 30      # ~6 weeks lookback
max_prediction_length: 5    # 5-day forecast
batch_size: 64
learning_rate: 0.001
hidden_size: 64
attention_head_size: 4
dropout: 0.2
hidden_continuous_size: 32
lstm_layers: 2
quantiles: [0.1, 0.5, 0.9]
max_epochs: 50
early_stopping_patience: 10
target_type: "returns"
prediction_horizon: 5
validation_split: 0.15
```

---

## Redis Signal Channels

| Channel | Content |
|---------|---------|
| `apex:signals:stock` | Equity ensemble signals |
| `apex:signals:forex` | FX pair signals |
| `apex:signals:options` | Options/volatility signals |
| `apex:signals:risk` | Composite tail risk index |

---

## Database

### Paper Trader (PostgreSQL)

Database `tft_trading` (default), tables:
- `paper_trades` — Individual trade records
- `paper_daily_snapshots` — End-of-day portfolio state
- `paper_strategy_signals` — Per-strategy signal records

### Core Pipeline (PostgreSQL)

Database `stock_trading_analysis` (default via `.env.template`), tables:
- `ohlcv` — Daily OHLCV (VIEW aggregating intraday bars)
- `fundamentals`, `sentiment`, `earnings`, `symbols`
- `economic_indicators`, `vix_data`

**Note:** Paper trader defaults to `tft_trading` while core pipeline defaults to `stock_trading_analysis`. Set `DB_NAME` in `.env` to unify.

---

## Testing

635 tests across 30 test modules:

| Category | Files | Tests |
|----------|-------|-------|
| Model tests | 4 (sentiment, mean_reversion, macro, microstructure) | 38 |
| Strategy tests | 5 (mean_reversion, sector_rotation, fx_momentum, fx_vol_breakout, vol_arb) | 35 |
| Integration | 2 (full_pipeline, circuit_breaker) | 9 |
| Safety guardrails | 1 (test_guardrails.py) | 33 |
| Production hardening | 1 (Kafka, TimescaleDB, schema registry) | 38 |
| Security hardening | 1 (paths, passwords, secrets, env validator) | 22 |
| Dead Letter Queue | 1 (backoff, persistence, retry, integration) | 39 |
| Prometheus metrics | 1 (needs `prometheus_client`) | 69 |
| Other | 14 (walk-forward, risk, CI, signals, etc.) | 352 |

Run: `pytest tests/ -v`

---

## Production Hardening (v3.0.0)

### Data Retention
- **Kafka**: KRaft broker with 168h log retention, 5GB max per topic, 1GB segments
- **TimescaleDB**: Hypertables with retention policies (365d OHLCV, 90d risk/execution)
- **Continuous Aggregates**: 15m, 1h, 1d OHLCV rollup views with automatic refresh

### Dead Letter Queue
- PostgreSQL-persisted failed message recovery (`services/common/dlq.py`)
- Exponential backoff: base 1s, 2x multiplier, max 5 retries, 0-25% jitter
- Integrated into all 4 Kafka consumers with background retry worker
- `/dlq` endpoint for monitoring

### Security
- All credentials via `os.environ[]` (no defaults, raises on missing)
- Startup validation (`utils/env_validator.py`) with placeholder detection
- No hardcoded paths or secrets in codebase
- `.env.example` has safe placeholders only

### Testing
635 tests across 30 files including production hardening (38), security (22), and DLQ (39) tests.

---

## Monitoring & Observability

### Prometheus Metrics (`monitoring/metrics.py`)

`PrometheusMetrics` class with dedicated `CollectorRegistry`, exposed via ASGI app at `/metrics`.

| Metric | Type | Description |
|--------|------|-------------|
| `apex_signal_score` | Gauge | Per-symbol ensemble signal score |
| `apex_strategy_weight` | Gauge | Per-strategy weight in ensemble |
| `apex_regime_state` | Info | Current market regime classification |
| `apex_ensemble_confidence` | Histogram | Distribution of signal confidence values |
| `apex_execution_slippage_bps` | Histogram | Execution slippage in basis points |
| `apex_pipeline_duration_seconds` | Histogram | Pipeline run duration |
| `apex_risk_*` | Gauges | max_drawdown, var_99, cvar_95 |

### Grafana Dashboard

Pre-configured dashboard at `monitoring/grafana/dashboards/apex_ensemble.json` with panels for signal scores, strategy weights, risk metrics, and pipeline performance.

### LLM Signal Analyst (`agents/signal_analyst.py`)

`SignalAnalyst` class (~500 lines) using local Ollama (default model: `qwen2.5:32b`). Detects patterns: consensus (>80%), conflict (40-60%), weight shifts (>10%), regime changes. Produces `SignalAnalysis` with narrative summary and actionable insights.

---

## Signal Provider REST API (`api/signal_provider.py`)

External signal distribution API mounted at `/api/v1/`. Created via `create_signal_api()` factory.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/signals` | All current ensemble signals |
| GET | `/api/v1/signals/{symbol}` | Per-symbol signal with ETag caching |
| GET | `/api/v1/signals/history/{symbol}` | Historical signals for symbol |
| GET | `/api/v1/signals/regime` | Current regime state |
| GET | `/api/v1/signals/weights` | Current strategy weights |

**Security:** API key authentication via `X-API-Key` header (`SIGNAL_API_KEY` env var). In-memory rate limiting: 100 requests/minute per key. ETag support for conditional requests.

---

## CI/CD Pipeline (`.github/workflows/ci.yml`)

Four-job GitHub Actions pipeline triggered on push and pull requests:

| Job | Steps | Purpose |
|-----|-------|---------|
| `lint` | black --check, flake8, mypy, yamllint | Code quality |
| `test` | pytest with JUnit XML output | Functional verification |
| `security` | pip-audit, detect-secrets | Vulnerability and secret scanning |
| `docker` | Build + verify Docker image | Container readiness (push to main only) |

---

## Environment Variables Summary

### API Keys
```bash
POLYGON_API_KEY=            # Market data
ALPACA_API_KEY=             # Trading
ALPACA_SECRET_KEY=          # Trading
```

### Database
```bash
DB_HOST=localhost           # Paper trader convention
DB_PORT=5432
DB_NAME=tft_trading
POSTGRES_HOST=localhost     # train_postgres.py convention
POSTGRES_DB=stock_trading_analysis
```

### Safety Guardrails
```bash
GUARDRAIL_SIGNAL_MIN_STD=0.01
GUARDRAIL_MAX_LEVERAGE=1.5
GUARDRAIL_CALIBRATION_TOLERANCE=1e-6
GUARDRAIL_MIN_PROMOTION_SHARPE=0.5
GUARDRAIL_MAX_EXEC_FAILURE_RATE=0.25
GUARDRAIL_EXEC_WINDOW_SECONDS=3600
```

### Notifications
```bash
DISCORD_WEBHOOK_URL=
EMAIL_USER=
EMAIL_PASSWORD=
EMAIL_TO=
```

---

*Document Version: 3.0.0 | Classification: Internal Use*
