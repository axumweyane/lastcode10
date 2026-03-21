# Changelog

All notable changes to the APEX Trading System are documented in this file.

---

## [2.1.0] — 2026-03-21

### Added — Safety Guardrails

Five automated pre-trade safety guardrails to prevent repeat of the March 10 signal collapse incident.

**New files:**
- `trading/safety/guardrails.py` (315 lines) — 5 guardrail classes
- `trading/safety/__init__.py`
- `tests/test_guardrails.py` (33 tests)

**Guardrails:**

| # | Guardrail | Purpose | Env Var | Default |
|---|-----------|---------|---------|---------|
| 1 | `SignalVarianceGuard` | Halt pipeline when ensemble scores collapse to near-identical values | `GUARDRAIL_SIGNAL_MIN_STD` | 0.01 |
| 2 | `LeverageGate` | Hard-limit leverage before every order batch | `GUARDRAIL_MAX_LEVERAGE` | 1.5 |
| 3 | `CalibrationHealthCheck` | Verify Platt/isotonic calibrators are fitted, not identity | `GUARDRAIL_CALIBRATION_TOLERANCE` | 1e-6 |
| 4 | `ModelPromotionGate` | Reject models with validation Sharpe below threshold | `GUARDRAIL_MIN_PROMOTION_SHARPE` | 0.5 |
| 5 | `ExecutionFailureMonitor` | Track rolling order failure rate, pause on breach | `GUARDRAIL_MAX_EXEC_FAILURE_RATE` | 0.25 |

**Integration points in `paper-trader/main.py`:**
- Calibration health check runs at startup after model loading
- Signal variance check after `combiner.combine()` — aborts pipeline on failure
- Leverage gate after `optimizer.optimize()` — skips order batch on failure
- Execution failure monitor wraps each `broker.submit_order()` call

**Other changes:**
- `.env.template` — Added 7 guardrail environment variables
- `CLAUDE.md` — Added guardrails section, known bugs, updated pipeline steps (13 → 16)
- `ARCHITECTURE.md` — Added guardrails to diagrams, data flow, component tables, test counts
- `COMPLETE_SYSTEM_DOCUMENTATION.md` — Full rewrite reflecting current 10-model, 11-strategy, 5-guardrail system

### Changed — Documentation

- Updated version 2.0.0 → 2.1.0 across all docs
- Updated test count: 81 → 114 tests across 12 modules
- Updated file count: ~142 → ~145 Python files
- Added daily pipeline steps 7, 10, 11 (guardrail checkpoints)

---

## [2.0.0] — 2026-03-20

### Added — Models 7–10 (58 improvements across 7 files)

Four new models completing the 10-model ensemble:

| # | Model | File | Type |
|---|-------|------|------|
| 7 | Sentiment | `models/sentiment_model.py` | FinBERT + VADER fallback |
| 8 | Mean Reversion | `models/mean_reversion_model.py` | Hurst exponent + OU params |
| 9 | Macro Regime | `models/macro_model.py` | Yield curve + rate classification |
| 10 | Microstructure | `models/microstructure_model.py` | Volume profile + order flow |

**Model details:**
- Sentiment: tries FinBERT (`ProsusAI/finbert`), falls back to VADER; outputs sentiment_score, momentum, dispersion, article_count
- Mean Reversion: R/S analysis for Hurst exponent, OLS for Ornstein-Uhlenbeck (mu, theta, sigma); outputs hurst, half_life, z-score, probability
- Macro Regime: fetches ^TNX, ^FVX, ^IRX, UUP via yfinance; classifies 5 regimes with sector tilts for 40+ symbols
- Microstructure: relative_volume, VWAP deviation, CLV, buying_pressure, volume_trend, A/D line

### Added — Strategies using new models

| # | Strategy | File | Uses Model |
|---|----------|------|------------|
| 3 | Mean Reversion | `strategies/stocks/mean_reversion.py` | MeanReversionModel (#8), MicrostructureModel (#10) |
| 4 | Sector Rotation | `strategies/stocks/sector_rotation.py` | MacroRegimeModel (#9) |

### Added — Tests for new models and strategies

- `tests/test_sentiment_model.py`
- `tests/test_mean_reversion_model.py`
- `tests/test_macro_model.py`
- `tests/test_microstructure_model.py`
- `tests/test_mean_reversion_strategy.py`
- `tests/test_sector_rotation_strategy.py`

### Changed — `models/manager.py`

- Extended `ModelManager` registry from 6 to 10 models
- All new models load with graceful fallback (missing → empty predictions)

### Changed — `strategies/ensemble/combiner.py`

- Added regime weight mappings for `mean_reversion` and `sector_rotation` strategies

### Changed — `strategies/config.py`

- Added `STRATEGY_MEAN_REVERSION_ENABLED` and `STRATEGY_SECTOR_ROTATION_ENABLED` env vars

---

## [1.5.0] — 2026-03-19

### Added — Multi-Strategy Ensemble System

- 11 trading strategies across stocks, FX, and options
- `EnsembleCombiner` with Bayesian weighting (60% Sharpe + 40% regime)
- `RegimeDetector` — 4-state classifier (calm/volatile × trending/choppy)
- `PortfolioOptimizer` — risk-constrained portfolio construction
- Redis pub/sub signal distribution (`apex:signals:*`)

### Added — Paper Trader

- `paper-trader/main.py` — FastAPI service on port 8010
- Daily pipeline: data → regime → strategies → ensemble → optimize → execute → log
- Live HTML dashboard at `/dashboard`
- PostgreSQL logging (paper_trades, paper_daily_snapshots, paper_strategy_signals)

### Added — Production Infrastructure

- `AlpacaBroker` (282 lines) — production order execution
- `CircuitBreaker` (404 lines) — Redis-backed drawdown limits
- `AuditLogger` (276 lines) — PostgreSQL audit trail
- `NotificationManager` (201 lines) — Discord + Email alerts
- `PortfolioRiskManager` (478 lines) — VaR, correlation, kill switches

### Added — Models 1–6

- TFT Stocks (`stocks_adapter.py`), TFT Forex (`forex_model.py`), TFT Volatility (`volatility_model.py`)
- Kronos (`kronos_model.py`) — HuggingFace pre-trained
- Deep Surrogates (`deep_surrogate_model.py`) — Heston calibration, tail risk
- TDGF (`tdgf_model.py`) — PDE solver for American options

### Added — FX and Options Strategies

- FX Carry + Trend, FX Momentum, FX Volatility Breakout
- Deep Surrogates, TDGF American Options, Vol Surface Arbitrage
- Kronos Forecasting

### Added — Options Infrastructure

- Pricing engine, chain fetcher, Greeks calculator, vol surface, vol monitor

### Added — Microservices Layer

- 5 FastAPI services (data-ingestion, sentiment-engine, tft-predictor, trading-engine, orchestrator)
- Kafka topics, Docker/K8s deployment via `docker-compose.yml`

---

## [1.0.0] — 2026-03-19

### Initial Release — Core TFT Pipeline

- Temporal Fusion Transformer model (`tft_model.py`, `tft_postgres_model.py`)
- Dual backends: legacy SQLite and PostgreSQL
- Data ingestion from Polygon.io and Reddit
- Technical indicator preprocessing (RSI, MACD, Bollinger Bands)
- `StockRankingSystem` signal generation + `PortfolioConstructor`
- FastAPI prediction API on port 8000
- Scheduler for automated training/prediction runs
- Cross-Sectional Momentum and Pairs Trading (StatArb) strategies
