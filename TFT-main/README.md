# APEX — Multi-Strategy Algorithmic Trading Platform

![CI](https://github.com/axumweyane/lastcode10/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Strategies](https://img.shields.io/badge/strategies-12-green)
![Models](https://img.shields.io/badge/models-10-orange)
![Tests](https://img.shields.io/badge/tests-635-brightgreen)

A production-grade algorithmic trading platform built around the **Temporal Fusion Transformer (TFT)**. 10 models across 4 asset classes, 12 ensemble strategies, Bayesian regime-adaptive signal fusion, automated paper trading with Alpaca, and a full microservices deployment layer.

> **Live showcase**: [axumweyane.github.io/apex-showcase](https://axumweyane.github.io/apex-showcase)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         APEX TRADING PLATFORM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DATA LAYER                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ yfinance  │  │Polygon.io│  │  Reddit  │  │ FinBERT  │               │
│  │  OHLCV    │  │  OHLCV   │  │ Sentiment│  │   NLP    │               │
│  └─────┬─────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘               │
│        └───────────────┼───────────────┼───────────┘                    │
│                        ▼                                                │
│  ┌─────────────────────────────────────────────────────────┐           │
│  │              TimescaleDB / PostgreSQL                    │           │
│  │   OHLCV │ Trades │ Snapshots │ Signals │ Risk Reports   │           │
│  └─────────────────────┬───────────────────────────────────┘           │
│                        │                                                │
│  MODEL LAYER (10 models)                                               │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐              │
│  │  TFT   │ │  TFT   │ │  TFT   │ │ Kronos │ │  Deep  │              │
│  │ Stocks │ │ Forex  │ │  Vol   │ │ Found. │ │Surrogate│             │
│  │ 0.031  │ │ 0.005  │ │ 0.041  │ │HugFace│ │  Heston│              │
│  └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘              │
│  ┌────┴───┐ ┌────┴───┐ ┌────┴───┐ ┌────┴───┐ ┌────┴───┐              │
│  │  TDGF  │ │Sentimnt│ │MeanRev │ │ Macro  │ │ Micro  │              │
│  │  PDE   │ │FinBERT │ │ Hurst  │ │ Regime │ │  Struct│              │
│  └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘              │
│       └──────────┼──────────┼──────────┼──────────┘                    │
│                  ▼                                                       │
│  STRATEGY LAYER (12 strategies)                                        │
│  ┌────────────────────────────────────────────────────────┐            │
│  │ Momentum │ StatArb │ MeanRev │ SectorRot │ FX Carry   │            │
│  │ FX Mom   │ FX Vol  │ Kronos  │ DeepSurr  │ TDGF       │            │
│  │ VolArb   │Sentiment│         │           │            │            │
│  └─────────────────────┬──────────────────────────────────┘            │
│                        ▼                                                │
│  ENSEMBLE LAYER                                                        │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐            │
│  │   Regime     │  │   Bayesian    │  │   Portfolio       │            │
│  │  Detector    │──│   Combiner    │──│   Optimizer       │            │
│  │ 4-state HMM  │  │ 60/40 weight  │  │ risk-constrained │            │
│  └──────────────┘  └───────────────┘  └────────┬─────────┘            │
│                                                 │                       │
│  SAFETY & EXECUTION                             ▼                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐              │
│  │ Signal   │ │ Leverage │ │Execution │ │   Alpaca     │              │
│  │ Variance │ │   Gate   │ │ Monitor  │ │   Broker     │              │
│  │  Guard   │ │  ≤1.5×   │ │ ≤25% fail│ │  paper/live  │              │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘              │
│                                                                         │
│  INFRASTRUCTURE                                                        │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐              │
│  │  Redis  │ │ Kafka  │ │Promeths│ │Grafana │ │ MLflow │              │
│  │pub/sub  │ │ events │ │metrics │ │ dashbd │ │tracking│              │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Strategies

| # | Strategy | Asset Class | Model Dependency | Description |
|---|----------|-------------|------------------|-------------|
| 1 | Cross-Sectional Momentum | Stocks | None | 12-1 momentum + 5d reversal + quality factor |
| 2 | Pairs Trading (StatArb) | Stocks | None | Engle-Granger cointegration, sector-constrained |
| 3 | Mean Reversion | Stocks | MeanReversionModel | Hurst exponent + OU parameter estimation |
| 4 | Sector Rotation | Stocks | MacroRegimeModel | Yield curve + rate trend driven rotation |
| 5 | FX Carry + Trend | Forex | None | Interest rate differential + trend following |
| 6 | FX Momentum | Forex | None | Multi-lookback trend composite |
| 7 | FX Vol Breakout | Forex | TFT Volatility | Vol-forecast breakout entries |
| 8 | Deep Surrogates | Options | DeepSurrogateModel | Neural option pricing, Heston calibration |
| 9 | TDGF American Options | Options | TDGFModel | PDE solver for American exercise boundary |
| 10 | Vol Surface Arbitrage | Options | None | IV vs RV spread capture |
| 11 | Kronos Forecasting | Multi-asset | KronosModel | Pre-trained K-line foundation model |
| 12 | Sentiment | Cross-asset | SentimentModel | FinBERT + VADER news/social scoring |

---

## Models

| Model | Val Loss | Asset Class | Source |
|-------|----------|-------------|--------|
| TFT Stocks | **0.031** | Stocks | Trained on AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, BAC, XOM |
| TFT Forex | **0.0045** | Forex | Trained on EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF |
| TFT Volatility | **0.041** | Volatility | Trained on 12 symbols + VIX regime context |
| Kronos | Pre-trained | Multi-asset | HuggingFace `NeoQuasar/Kronos-base` |
| Deep Surrogates | Pre-trained | Options/Vol | TensorFlow neural option pricing |
| TDGF | Light train | Options | PyTorch PDE solver (Black-Scholes, Heston, Lifted Heston) |
| Sentiment | Pre-trained | Cross-asset | FinBERT (`ProsusAI/finbert`) + VADER fallback |
| Mean Reversion | Statistical | Stocks | Hurst exponent + Ornstein-Uhlenbeck estimation |
| Macro Regime | Rule-based | Cross-asset | Yield curve + interest rate trends via yfinance |
| Microstructure | Statistical | Stocks | Volume profile + order flow analysis |

---

## Risk Management

### Circuit Breaker (Redis-backed)
- **High Water Mark**: 5.0% drawdown from peak → halt all trading
- **Start of Day**: 3.0% intraday loss → halt for the session
- State persisted in both Redis and PostgreSQL for crash recovery
- Fail-closed: Redis failure = no trading (safe default)

### Safety Guardrails
| Guardrail | Trigger | Action |
|-----------|---------|--------|
| Signal Variance | std < 0.01 | Halt pipeline, Discord critical alert |
| Leverage Gate | leverage > 1.5× | Skip order batch |
| Execution Monitor | >25% failure rate | Pause remaining orders |
| Calibration Health | tolerance > 1e-6 | Skip calibration |
| Model Promotion | Sharpe < 0.5 | Reject model |

### Portfolio Risk Manager
- VaR (99%) and CVaR (95%) monitoring
- Correlation alerts: >0.85 triggers 50% weight reduction
- Per-strategy kill switches (drawdown + Sharpe floors)
- Portfolio-level kill switch with persistent state
- 30-day historical seeding at startup

---

## Infrastructure

| Component | Purpose | Port |
|-----------|---------|------|
| Paper Trader (FastAPI) | Daily ensemble pipeline | 8010 |
| TimescaleDB | OHLCV, trades, signals, risk reports | 15432 |
| Redis | Circuit breaker state, signal pub/sub | 6379 |
| Prometheus | Metrics scraping (8 metric groups) | 9090 |
| Grafana | Trading dashboards | 3000 |
| Kafka (KRaft) | Event streaming (7 topics) | 9092 |
| MLflow | Experiment tracking | 5001 |
| systemd | Auto-start on boot (linger enabled) | — |

### Prometheus Metrics
- `apex_signal_score` — per-symbol ensemble signals
- `apex_strategy_weight` — Bayesian + fixed weights
- `apex_regime_state` — 4-state regime classification
- `apex_risk_drawdown` / `apex_risk_var_99` / `apex_risk_cvar_95`
- `apex_pipeline_duration_seconds` — daily run timing
- `apex_execution_slippage_bps` — per-trade slippage
- `apex_ensemble_confidence` — signal confidence distribution

---

## Tech Stack

**ML/AI**: PyTorch, PyTorch Forecasting, TensorFlow, PyTorch Lightning, Optuna, scikit-learn, FinBERT, HuggingFace Transformers

**Trading**: Alpaca API, yfinance, Polygon.io, PRAW (Reddit)

**Infrastructure**: FastAPI, PostgreSQL/TimescaleDB, Redis, Apache Kafka, Docker, systemd

**Monitoring**: Prometheus, Grafana, MLflow, Discord webhooks

**Data**: pandas, NumPy, statsmodels (cointegration, Hurst), scipy (optimization)

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/axumweyane/lastcode10.git
cd TFT-main
./setup.sh

# Configure
cp .env.example .env
# Edit .env with your Alpaca API keys and database credentials

# Train models
python train_postgres.py --symbols AAPL MSFT GOOGL --start-date 2024-01-01 --max-epochs 30
python -m models.train_forex --epochs 30
python -m models.train_volatility --epochs 30

# Start paper trader
systemctl --user start apex-paper-trader
# Or manually:
cd paper-trader && python -m uvicorn main:app --host 0.0.0.0 --port 8010

# Verify
curl http://localhost:8010/health
# Dashboard: http://localhost:8010/dashboard
# Grafana:   http://localhost:3000/d/apex-paper-trader
```

### Run Backtest
```bash
python run_backtest.py          # 3-strategy real data backtest
python optimize_strategies.py   # Walk-forward parameter optimization
```

### Docker (full microservices stack)
```bash
docker network create tft_network
docker-compose up
```

---

## Project Structure

```
TFT-main/
├── paper-trader/main.py          # FastAPI daily pipeline (port 8010)
├── models/                       # 10 model implementations
│   ├── manager.py                # Unified model loader
│   ├── stocks_adapter.py         # TFT-Stocks wrapper
│   ├── forex_model.py            # TFT-Forex (val_loss=0.0045)
│   ├── volatility_model.py       # TFT-Volatility (val_loss=0.041)
│   ├── kronos_model.py           # Kronos foundation model
│   ├── deep_surrogate_model.py   # Neural option pricing
│   ├── tdgf_model.py             # PDE solver
│   ├── sentiment_model.py        # FinBERT + VADER
│   ├── mean_reversion_model.py   # Hurst + OU estimation
│   ├── macro_model.py            # Yield curve regime
│   └── microstructure_model.py   # Volume profile
├── strategies/                   # 12 strategy implementations
│   ├── ensemble/combiner.py      # Bayesian signal fusion
│   ├── ensemble/portfolio_optimizer.py
│   ├── regime/detector.py        # 4-state market regime
│   ├── risk/portfolio_risk.py    # VaR/CVaR, kill switches
│   ├── validation/walk_forward.py # Walk-forward CV engine
│   └── config.py                 # All strategy configs
├── trading/                      # Production infrastructure
│   ├── broker/alpaca.py          # Alpaca execution (283 lines)
│   ├── risk/circuit_breaker.py   # Redis-backed CB (405 lines)
│   ├── safety/guardrails.py      # 5 automated guardrails
│   ├── persistence/audit.py      # PostgreSQL audit trail
│   └── notifications/alerts.py   # Discord + Email
├── microservices/                # 5 Kafka-connected services
├── monitoring/metrics.py         # Prometheus metrics
├── docker-compose.yml            # Full stack deployment
└── tests/                        # 635 tests across 30 modules
```

---

## Performance

All 3 TFT models trained on 2024-2026 market data:

| Model | Best Val Loss | Epochs | Early Stop |
|-------|-------------|--------|------------|
| TFT-Stocks | 0.031 | 10/30 | Yes (patience=10) |
| TFT-Forex | 0.0045 | 30/30 | No |
| TFT-Volatility | 0.041 | 14/30 | Yes (patience=8) |

---

## License

MIT

---

*Built with PyTorch, FastAPI, and a lot of market data.*
