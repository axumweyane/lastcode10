# APEX Architecture Documentation

**Version**: 3.0.0
**Last Updated**: 2026-03-21
**Total Python Files**: ~179
**Asset Classes**: Stocks, Forex, Options/Volatility, Cross-Asset

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Data Flow: Market Data to Trade Execution](#4-data-flow-market-data-to-trade-execution)
5. [Data Pipelines](#5-data-pipelines)
6. [Model Layer (10 Models)](#6-model-layer-10-models)
7. [Strategy Layer (11 Strategies)](#7-strategy-layer-11-strategies)
8. [Ensemble System](#8-ensemble-system)
9. [Regime Detection](#9-regime-detection)
10. [Portfolio Optimization](#10-portfolio-optimization)
11. [Paper Trader Service](#11-paper-trader-service)
12. [Trading Infrastructure](#12-trading-infrastructure)
12b. [Safety Guardrails](#12b-safety-guardrails)
13. [Database Schema (TimescaleDB)](#13-database-schema-timescaledb)
14. [Redis Usage](#14-redis-usage)
15. [Kafka Messaging](#15-kafka-messaging)
16. [Microservices Layer](#16-microservices-layer)
17. [Configuration & Environment](#17-configuration--environment)
18. [API Endpoints](#18-api-endpoints)
19. [Scheduling](#19-scheduling)
20. [Model Training Pipeline](#20-model-training-pipeline)
21. [Testing](#21-testing)
22. [Deployment Status](#22-deployment-status)

---

## 1. System Overview

APEX is a multi-strategy algorithmic trading platform built around the Temporal Fusion Transformer (TFT). It combines 10 AI/statistical models across 4 asset classes through 11 trading strategies, fused via a Bayesian ensemble with regime-adaptive weighting, and executed through a production paper-trading service connected to Alpaca.

The system has three layers that can operate independently:

1. **Core TFT Pipeline** — Train TFT models, generate predictions, rank stocks, construct portfolios. Two backends: legacy SQLite and PostgreSQL.
2. **Multi-Strategy Ensemble** — 11 strategies producing alpha signals, combined via Bayesian weighting, optimized into a risk-constrained portfolio, executed daily via the paper trader.
3. **Microservices Layer** — 5 FastAPI services coordinated via Kafka for distributed deployment (Docker/K8s).

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APEX Trading Platform                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  Data Layer   │    │  Model Layer  │    │ Strategy Layer│                  │
│  │              │    │  (10 models)  │    │(11 strategies)│                  │
│  │  yfinance    │───>│  TFT Stocks   │───>│  Momentum     │                  │
│  │  Polygon.io  │    │  TFT Forex    │    │  Pairs/StatArb│                  │
│  │  TimescaleDB │    │  TFT Vol      │    │  Mean Revert  │                  │
│  │  OHLCV View  │    │  Kronos       │    │  Sector Rot   │                  │
│  └──────────────┘    │  DeepSurrog   │    │  FX Carry     │                  │
│                      │  TDGF         │    │  FX Momentum  │                  │
│                      │  Sentiment    │    │  FX VolBreak  │                  │
│                      │  MeanRev      │    │  DeepSurrog   │                  │
│                      │  Macro        │    │  TDGF         │                  │
│                      │  Microstr     │    │  Vol Arb      │                  │
│                      └──────────────┘    │  Kronos       │                  │
│                                          └───────┬───────┘                  │
│                                                  │                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────▼───────┐                  │
│  │Regime Detect  │───>│   Ensemble    │<──│  Alpha Scores  │                  │
│  │ 4-state class │    │  Combiner     │    │  per strategy  │                  │
│  │ VIX+Breadth   │    │  Bayesian wts │    └───────────────┘                  │
│  └──────────────┘    └───────┬───────┘                                      │
│                              │                                              │
│                      ┌───────▼───────┐                                      │
│                      │   Portfolio    │                                      │
│                      │  Optimizer     │                                      │
│                      │  Vol-target    │                                      │
│                      │  VaR, leverage │                                      │
│                      └───────┬───────┘                                      │
│                              │                                              │
│  ┌──────────────┐    ┌───────▼───────┐    ┌──────────────┐                  │
│  │Circuit Breaker│───>│ Paper Trader  │───>│ Alpaca Broker │                  │
│  │ Redis-backed  │    │  FastAPI 8010 │    │  Paper/Live   │                  │
│  │ Drawdown chk  │    │  Daily cron   │    │  REST API     │                  │
│  └──────────────┘    └───────┬───────┘    └──────────────┘                  │
│                              │                                              │
│  ┌───────────────────────────┼───────────────────────────┐                  │
│  │         Safety Guardrails (pre-trade checks)          │                  │
│  │  SignalVariance │ LeverageGate │ ExecFailureMonitor   │                  │
│  │  CalibrationHealth │ ModelPromotionGate               │                  │
│  └───────────────────────────┼───────────────────────────┘                  │
│                              │                                              │
│         ┌────────────────────┼────────────────────┐                         │
│         │                    │                    │                         │
│  ┌──────▼──────┐    ┌───────▼───────┐    ┌───────▼──────┐                  │
│  │ PostgreSQL   │    │    Redis      │    │  Discord/    │                  │
│  │ Trades/Snaps │    │  Signals pub  │    │  Email Alerts│                  │
│  │ Audit trail  │    │  CB state     │    └──────────────┘                  │
│  └─────────────┘    └───────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
TFT-main/
├── api.py                          # Legacy REST API (SQLite backend)
├── api_postgres.py                 # PostgreSQL REST API
├── config_manager.py               # TFTConfig, TradingConfig dataclasses
├── data_pipeline.py                # StockDataCollector → SQLite
├── data_preprocessing.py           # StockDataPreprocessor (technical indicators)
├── postgres_data_loader.py         # PostgresDataLoader (psycopg2 queries)
├── postgres_data_pipeline.py       # PostgresDataPipeline (build_dataset, features)
├── postgres_schema.py              # CREATE TABLE statements
├── tft_model.py                    # EnhancedTFTModel (legacy)
├── tft_postgres_model.py           # TFTPostgresModel (PostgreSQL)
├── train.py                        # Legacy training script
├── train_postgres.py               # PostgreSQL training script
├── predict.py                      # Prediction CLI
├── stock_ranking.py                # Signal generation + portfolio construction
├── scheduler.py                    # Automated scheduling (manual/cron)
├── requirements.txt
├── docker-compose.yml
├── .env.template                   # Full environment config template
│
├── models/                         # 10 AI/statistical models
│   ├── base.py                     #   BaseTFTModel ABC, ModelPrediction, ModelInfo
│   ├── manager.py                  #   ModelManager (load_all, predict_*, 10 models)
│   ├── stocks_adapter.py           #   TFTStocksAdapter (wraps EnhancedTFTModel)
│   ├── forex_model.py              #   TFTForexModel (6 currency pairs)
│   ├── volatility_model.py         #   TFTVolatilityModel (VIX/vol surface)
│   ├── kronos_model.py             #   KronosModel (HuggingFace pre-trained)
│   ├── deep_surrogate_model.py     #   DeepSurrogateModel (neural option pricing)
│   ├── tdgf_model.py               #   TDGFModel (PDE American options)
│   ├── sentiment_model.py          #   SentimentModel (FinBERT + VADER)
│   ├── mean_reversion_model.py     #   MeanReversionModel (Hurst + OU)
│   ├── macro_model.py              #   MacroRegimeModel (yield curve)
│   └── microstructure_model.py     #   MicrostructureModel (volume/VWAP)
│
├── strategies/                     # 11 trading strategies
│   ├── base.py                     #   BaseStrategy ABC, AlphaScore, StrategyOutput
│   ├── config.py                   #   StrategyMasterConfig (all configs from env)
│   ├── momentum/
│   │   ├── cross_sectional.py      #   Strategy #1: Cross-Sectional Momentum
│   │   └── features.py             #   Factor computation engine
│   ├── statarb/
│   │   ├── pairs.py                #   Strategy #2: Pairs Trading
│   │   └── scanner.py              #   Cointegration pair scanner
│   ├── stocks/
│   │   ├── mean_reversion.py       #   Strategy #3: Mean Reversion
│   │   └── sector_rotation.py      #   Strategy #4: Sector Rotation
│   ├── fx/
│   │   ├── carry_trend.py          #   Strategy #5: FX Carry + Trend
│   │   ├── momentum.py             #   Strategy #6: FX Momentum
│   │   └── vol_breakout.py         #   Strategy #7: FX Volatility Breakout
│   ├── deep_surrogates/
│   │   └── strategy.py             #   Strategy #8: Deep Surrogates
│   ├── tdgf/
│   │   └── strategy.py             #   Strategy #9: TDGF American Options
│   ├── options/strategies/
│   │   └── vol_arb.py              #   Strategy #10: Vol Surface Arbitrage
│   ├── kronos/
│   │   └── strategy.py             #   Strategy #11: Kronos Forecasting
│   ├── ensemble/
│   │   ├── combiner.py             #   EnsembleCombiner (Bayesian fusion)
│   │   └── portfolio_optimizer.py  #   PortfolioOptimizer (risk constraints)
│   ├── regime/
│   │   └── detector.py             #   RegimeDetector (4-state classifier)
│   ├── risk/
│   │   └── portfolio_risk.py       #   PortfolioRiskManager (VaR, kill switches)
│   └── signals/
│       └── publisher.py            #   SignalPublisher (Redis pub/sub)
│
├── paper-trader/
│   └── main.py                     # FastAPI daily pipeline (port 8010)
│
├── trading/                        # Production infrastructure
│   ├── broker/
│   │   ├── base.py                 #   BaseBroker ABC, OrderRequest, OrderResult
│   │   └── alpaca.py               #   AlpacaBroker (aiohttp, paper/live)
│   ├── risk/
│   │   ├── circuit_breaker.py      #   CircuitBreaker (Redis, drawdown limits)
│   │   └── position_sizing.py      #   Fixed fractional, Kelly, vol-scaled
│   ├── safety/
│   │   └── guardrails.py           #   5 pre-trade safety checks (315 lines)
│   ├── notifications/
│   │   └── alerts.py               #   NotificationManager (Discord + Email)
│   └── persistence/
│       └── audit.py                #   AuditLogger (PostgreSQL audit trail)
│
├── microservices/                  # 5 Kafka-connected services
│   ├── data-ingestion/main.py      #   Port 8001 — Polygon + Reddit → Kafka
│   ├── sentiment-engine/main.py    #   Port 8002 — NLP scoring
│   ├── tft-predictor/main.py       #   Port 8003 — GPU inference
│   ├── trading-engine/main.py      #   Port 8004 — Alpaca execution
│   └── orchestrator/main.py        #   Port 8005 — Saga workflows
│
├── services/                       # Shared infrastructure
│   └── common/
│       └── dlq.py                  #   DeadLetterQueue (PostgreSQL, exp backoff)
│
├── utils/
│   └── env_validator.py            #   Startup environment validation
│
├── tests/                          # 635 tests across 30 files
│   ├── models/                     #   4 model test files (38 tests)
│   ├── strategies/                 #   5 strategy test files (35 tests)
│   ├── integration/                #   2 integration test files (9 tests)
│   ├── test_guardrails.py          #   33 tests (all 5 guardrails)
│   ├── test_production_hardening.py #  38 tests (Kafka, TimescaleDB, schema registry)
│   ├── test_security_hardening.py  #   22 tests (paths, passwords, secrets, env validator)
│   ├── test_dlq.py                 #   39 tests (DLQ persistence, backoff, retry, integration)
│   └── ... (13 more test files)    #   walk-forward, risk, CI, signals, etc.
│
├── options/                        # Options infrastructure
│   ├── infrastructure/             #   chain.py, greeks.py, pricing.py,
│   │                               #   vol_monitor.py, vol_surface.py
│   ├── risk/                       #   options_risk.py
│   └── strategies/                 #   covered_calls, iron_condors,
│                                   #   earnings_plays, gamma_scalping,
│                                   #   protective_puts
│
├── lightning_logs/                 # PyTorch Lightning checkpoints (7 versions)
├── data/                           # Raw data + SQLite
├── models/                         # Trained .pth files + preprocessors
├── predictions/                    # Prediction outputs
├── logs/                           # Application logs
└── reports/                        # Trade reports
```

---

## 4. Data Flow: Market Data to Trade Execution

The daily pipeline executes end-to-end in ~5 seconds:

```
Step 1: DATA FETCH (yfinance)
    │   Stocks: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, BAC, XOM
    │   FX: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF
    │   Market: SPY (for regime detection)
    │   300 trading days lookback
    │
Step 2: CIRCUIT BREAKER PRE-CHECK
    │   Redis GET circuit_breaker:is_tripped → O(1)
    │   If tripped → abort pipeline
    │
Step 3: REGIME DETECTION
    │   Inputs: VIX level, market breadth (% > 50d MA), realized vol
    │   Output: 4-state regime (calm/volatile × trending/choppy)
    │   Sets: strategy weight priors + exposure scalar
    │
Step 4: STRATEGY EXECUTION (parallel-safe, sequential in practice)
    │   Each enabled strategy:
    │     initialize(historical_data) → warm up
    │     generate_signals(data) → StrategyOutput[List[AlphaScore]]
    │
    │   ┌── Cross-Sectional Momentum ── 3 signals (factor-based)
    │   ├── Pairs Trading ──────────── 0 signals (cointegration)
    │   ├── Mean Reversion ─────────── 0 signals (Hurst + OU)
    │   ├── Sector Rotation ────────── 10 signals (macro regime)
    │   ├── FX Carry + Trend ───────── 4 signals (rate differentials)
    │   ├── FX Momentum ────────────── 4 signals (multi-lookback)
    │   ├── FX Vol Breakout ────────── 0 signals (BB squeeze)
    │   ├── (Deep Surrogates) ──────── disabled
    │   ├── (TDGF) ─────────────────── disabled
    │   ├── (Vol Arb) ──────────────── disabled
    │   └── (Kronos) ───────────────── disabled
    │
Step 5: RISK ASSESSMENT
    │   PortfolioRiskManager.assess()
    │   VaR calculation, correlation alerts, strategy kill switches
    │
Step 6: ENSEMBLE COMBINATION
    │   Bayesian weights = 0.6 × Sharpe-based + 0.4 × regime-based
    │   Per-symbol: weighted_score = Σ(strategy_score × weight × confidence)
    │   Output: 13 combined signals (10 long, 3 short)
    │
Step 6a: GUARDRAIL — SIGNAL VARIANCE CHECK
    │   std(scores) > GUARDRAIL_SIGNAL_MIN_STD (default 0.01)
    │   If collapsed → HALT pipeline, Discord critical alert
    │   Prevents repeat of March 10 0.5429 incident
    │
Step 7: SIGNAL PUBLISHING (fire-and-forget)
    │   Redis PUBLISH to apex:signals:{stock,forex,options}
    │   13/13 signals published
    │
Step 8: PORTFOLIO OPTIMIZATION
    │   Inverse-vol weighting → regime exposure scaling
    │   Position caps, gross/net leverage constraints
    │   Target vol: 15%, VaR99 constraint
    │   Output: 12 positions (9L / 3S), gross=0.80, net=0.32
    │
Step 8b: GUARDRAIL — LEVERAGE GATE
    │   gross_leverage <= GUARDRAIL_MAX_LEVERAGE (default 1.5)
    │   If exceeded → skip entire order batch, log warning
    │
Step 9: TRADE EXECUTION
    │   For each target position:
    │     GUARDRAIL: Execution failure rate check (25% in 1hr → pause)
    │     Calculate target_shares = (target_weight × portfolio_value) / price
    │     diff = target_shares - current_holdings
    │     if |diff| >= 1: submit OrderRequest via AlpacaBroker
    │     Record outcome in ExecutionFailureMonitor
    │   Circuit breaker re-check before each trade
    │
Step 10: LOGGING & PERSISTENCE
    │   PostgreSQL: paper_trades, paper_daily_snapshots, paper_strategy_signals
    │   AuditLogger: pipeline events
    │   Redis: signal state
    │
Step 11: NOTIFICATIONS
        Discord webhook + Email (if configured)
```

---

## 5. Data Pipelines

### Two Parallel Pipelines

| Layer | Legacy (SQLite) | PostgreSQL |
|-------|----------------|------------|
| Data Loading | `data_pipeline.py` → `StockDataCollector` | `postgres_data_loader.py` → `PostgresDataLoader` |
| Preprocessing | `data_preprocessing.py` → `StockDataPreprocessor` | `postgres_data_pipeline.py` → `PostgresDataPipeline` |
| Model | `tft_model.py` → `EnhancedTFTModel` | `tft_postgres_model.py` → `TFTPostgresModel` |
| Training | `train.py` | `train_postgres.py` |
| API | `api.py` (port 8000) | `api_postgres.py` (port 8000) |

Both pipelines share: `stock_ranking.py`, `predict.py`, `scheduler.py`, `config_manager.py`.

### OHLCV View (Bridge Layer)

The database stores intraday bars in `ohlcv_bars` (with `time` timestamp column). The training pipeline expects daily OHLCV with a `date` column. A SQL VIEW bridges the two:

```sql
CREATE OR REPLACE VIEW ohlcv AS
SELECT d.date, d.symbol, d.open, d.high, d.low, d.close, d.volume,
       d.open AS adj_open, d.high AS adj_high, d.low AS adj_low,
       d.close AS adj_close, d.volume AS adj_volume
FROM (
    SELECT time::date AS date, symbol,
           (ARRAY_AGG(open::double precision ORDER BY time ASC))[1] AS open,
           MAX(high::double precision) AS high,
           MIN(low::double precision) AS low,
           (ARRAY_AGG(close::double precision ORDER BY time DESC))[1] AS close,
           SUM(volume::double precision) AS volume
    FROM ohlcv_bars
    GROUP BY time::date, symbol
) d;
```

This aggregates intraday bars to daily OHLCV: first open of day, day high/low, last close of day, total volume.

### PostgresDataPipeline.build_dataset() Flow

1. Load OHLCV from `ohlcv` view via `PostgresDataLoader`
2. Add technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, OBV)
3. Add temporal features (day/week/month sin/cos, day-of-week flags)
4. Merge fundamentals (market_cap, PE, EPS, dividend_yield — defaults to 0 if table missing)
5. Merge sentiment scores (defaults to 0 if table missing)
6. Add earnings calendar flags (defaults to 0/999 if table missing)
7. Create target variable: `pct_change(horizon).shift(-horizon)` for forward returns
8. Add time_idx (cumcount per symbol) and group_id
9. Clean dataset: ffill/bfill numerics, drop NaN/inf targets
10. Filter symbols with < 100 observations

### Feature Columns for TFT

| Category | Features |
|----------|----------|
| Static Categoricals | symbol, sector, industry, exchange |
| Static Reals | market_cap, pe_ratio, eps, dividend_yield |
| Time-Varying Known | time_idx, day/week/month sin/cos, is_monday, is_friday, is_month_end, is_quarter_end, earnings_flag, days_to_earnings |
| Time-Varying Unknown | adj_open/high/low/close/volume, sma_5/10/20/50, ema_12/26, rsi, macd/signal/histogram, bb_upper/lower/position, volume_ratio, obv, price_change, high_low_ratio, close_open_ratio, returns_volatility, sentiment_score/magnitude, news_count |

---

## 6. Model Layer (10 Models)

All models extend `BaseTFTModel` (ABC in `models/base.py`) and return `List[ModelPrediction]`.

`ModelManager` (`models/manager.py`) loads all models and provides unified prediction access with graceful fallback — missing models return empty predictions, strategies fall back to non-TFT signals.

| # | Model | Class | Asset Class | Type | Source | Status |
|---|-------|-------|-------------|------|--------|--------|
| 1 | TFT Stocks | `TFTStocksAdapter` | stocks | Deep learning | `models/tft_model.pth` (3.3MB) | Loaded |
| 2 | TFT Forex | `TFTForexModel` | forex | Deep learning | `models/tft_forex.pth` (1.6MB) | Loaded |
| 3 | TFT Volatility | `TFTVolatilityModel` | volatility | Deep learning | `models/tft_volatility.pth` (2.7MB) | Loaded |
| 4 | Kronos | `KronosModel` | stocks+forex | Pre-trained | HuggingFace `NeoQuasar/Kronos-base` | Not loaded (needs `/opt/kronos`) |
| 5 | Deep Surrogates | `DeepSurrogateModel` | options/vol | Pre-trained | `/opt/deep_surrogate` repo | Not loaded (needs repo) |
| 6 | TDGF | `TDGFModel` | options | Light training | `/opt/tdgf` repo | Not loaded (needs repo) |
| 7 | Sentiment | `SentimentModel` | cross-asset | Pre-trained | FinBERT / VADER fallback | Loaded (VADER) |
| 8 | Mean Reversion | `MeanReversionModel` | stocks | Statistical | Hurst exponent + OU estimation | Loaded |
| 9 | Macro Regime | `MacroRegimeModel` | cross-asset | Rule-based | Yield curve + rate trends | Loaded |
| 10 | Microstructure | `MicrostructureModel` | stocks | Statistical | Volume profile + order flow | Loaded |

### ModelPrediction Dataclass

```python
@dataclass
class ModelPrediction:
    symbol: str
    prediction: float          # primary prediction value
    confidence: float          # 0-1
    timestamp: datetime
    horizon_days: int
    metadata: Dict[str, Any]   # model-specific details
```

### Key Model Details

**Mean Reversion Model** — Computes Hurst exponent via R/S analysis and fits Ornstein-Uhlenbeck parameters (mu, theta, sigma) via OLS on log prices. Returns: hurst_exponent, half_life, deviation_zscore, mr_probability.

**Macro Regime Model** — Fetches yield curve data (^TNX, ^FVX, ^IRX, UUP) via yfinance. Classifies into 5 regimes: steepening_rising, steepening_falling, flattening_rising, inverted, neutral. Maps to sector tilts via `REGIME_SECTOR_TILTS` dict covering 40+ symbols.

**Microstructure Model** — Daily-frequency analysis: relative_volume, VWAP deviation, Close Location Value (CLV), buying_pressure, volume_trend, Accumulation/Distribution line. Composite signal = weighted combination.

**Sentiment Model** — Tries FinBERT (`ProsusAI/finbert`) first, falls back to VADER. Accepts DataFrame with `symbol` and `text` columns. Returns sentiment_score, sentiment_momentum, sentiment_dispersion, article_count.

---

## 7. Strategy Layer (11 Strategies)

All strategies extend `BaseStrategy` (ABC in `strategies/base.py`) and produce `StrategyOutput` containing `List[AlphaScore]` — z-scored per-symbol alpha signals.

### BaseStrategy Interface

```python
class BaseStrategy(ABC):
    @property
    def name(self) -> str: ...              # unique identifier
    @property
    def description(self) -> str: ...       # human-readable

    def initialize(self, data: pd.DataFrame) -> None: ...
    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput: ...
    def get_performance(self) -> StrategyPerformance: ...
```

### AlphaScore Dataclass

```python
@dataclass
class AlphaScore:
    symbol: str
    score: float              # z-scored alpha signal
    raw_score: float          # pre-normalization
    confidence: float         # 0-1
    direction: SignalDirection # LONG / SHORT / NEUTRAL
    metadata: Dict[str, Any]
```

### Strategy Details

| # | Strategy | File | Logic | Model Deps |
|---|----------|------|-------|------------|
| 1 | Cross-Sectional Momentum | `momentum/cross_sectional.py` | Multi-factor (momentum, mean-rev, quality z-scores), risk-adjusted, cross-sectional ranking | None |
| 2 | Pairs Trading (StatArb) | `statarb/pairs.py` | Cointegration via Engle-Granger, z-score entry/exit, dynamic hedge ratios. Scanner tests 55 pairs. | None |
| 3 | Mean Reversion | `stocks/mean_reversion.py` | Hurst < threshold AND deviation_zscore > entry → trade. Anti-correlated with momentum by design. | MeanReversionModel (#8), optionally MicrostructureModel (#10) |
| 4 | Sector Rotation | `stocks/sector_rotation.py` | Macro regime → sector tilts. Fallback: 3-month relative strength across sectors. | MacroRegimeModel (#9) |
| 5 | FX Carry + Trend | `fx/carry_trend.py` | Interest rate differentials + trend overlay. 6 major pairs. | None |
| 6 | FX Momentum | `fx/momentum.py` | Multi-lookback returns: 1m(0.4), 3m(0.3), 6m(0.2), 12m(0.1). Cross-sectional z-scoring. | None |
| 7 | FX Vol Breakout | `fx/vol_breakout.py` | Bollinger Band bandwidth percentile. Squeeze detection (bottom 10%). Direction from momentum during squeeze. | TFT Volatility for vol forecast |
| 8 | Deep Surrogates | `deep_surrogates/strategy.py` | Neural option pricing surrogates. Tail risk index broadcasting. | DeepSurrogateModel via ModelManager |
| 9 | TDGF American Options | `tdgf/strategy.py` | PDE-based American option valuation. Mispricing detection. | TDGFModel via ModelManager |
| 10 | Vol Surface Arbitrage | `options/strategies/vol_arb.py` | IV-RV spread analysis, vol surface shape arbitrage. | None (IV-RV spread) |
| 11 | Kronos Forecasting | `kronos/strategy.py` | K-line foundation model predictions for stocks and forex. | KronosModel via ModelManager |

### Strategy Activation (Environment Variables)

All strategies are **disabled by default**. Enable via `.env`:

```bash
STRATEGY_MOMENTUM_ENABLED=true
STRATEGY_STATARB_ENABLED=true
STRATEGY_MEAN_REVERSION_ENABLED=true
STRATEGY_SECTOR_ROTATION_ENABLED=true
STRATEGY_FX_MOMENTUM_ENABLED=true
STRATEGY_FX_VOL_BREAKOUT_ENABLED=true
STRATEGY_KRONOS_ENABLED=false        # needs /opt/kronos
STRATEGY_DEEP_SURROGATES_ENABLED=false  # needs /opt/deep_surrogate
STRATEGY_TDGF_ENABLED=false          # needs /opt/tdgf
STRATEGY_VOL_ARB_ENABLED=false
```

---

## 8. Ensemble System

### EnsembleCombiner (`strategies/ensemble/combiner.py`)

The combiner fuses signals from all active strategies into a single ranked list of alpha scores.

**Weighting Methods** (configured via `ENSEMBLE_WEIGHTING_METHOD`):

1. **Equal** — 1/N per strategy
2. **Sharpe** — weight proportional to max(sharpe_63d, 0)
3. **Bayesian** (default) — prior = 1/N, update = prior + alpha × max(sharpe_63d, 0)

**Regime Blending Formula:**

```
final_weight = 0.6 × performance_weight + 0.4 × regime_weight
```

Weights are clamped to `[min_weight=0.05, max_weight=0.40]` and renormalized.

### Regime → Strategy Weight Mapping

Strategies map to 4 regime weight buckets:

| Bucket | Strategies |
|--------|-----------|
| `momentum` | cross_sectional_momentum, sector_rotation, fx_momentum |
| `mean_reversion` | mean_reversion |
| `pairs` | pairs_trading, fx_vol_breakout, deep_surrogates, tdgf, vol_arb |
| `tft` | fx_carry_trend, kronos |

Each `MarketRegime` state provides different weights for these 4 buckets (configured in `RegimeConfig`).

### Per-Symbol Score Calculation

```
weighted_score(symbol) = Σ_s [ strategy_score_s(symbol) × weight_s × confidence_s ]
                         / Σ_s [ weight_s where strategy_s has signal for symbol ]
```

---

## 9. Regime Detection

### RegimeDetector (`strategies/regime/detector.py`)

Classifies the market into a 2×2 matrix:

```
                    Trending          Choppy
              ┌─────────────────┬─────────────────┐
   Calm       │  CALM_TRENDING  │  CALM_CHOPPY    │
   (VIX < 20) │  Favor momentum │  Favor mean-rev │
              ├─────────────────┼─────────────────┤
   Volatile   │ VOLATILE_TREND  │ VOLATILE_CHOPPY │
   (VIX > 20) │  Reduce exposure│  Max caution    │
              └─────────────────┴─────────────────┘
```

**Inputs:**
- **VIX level** — explicit, column in data, or estimated from realized vol
- **Market breadth** — fraction of stocks with close > 50-day SMA
- **Realized volatility** — annualized std of SPY returns (63-day window)

**Classification Logic:**
- `is_volatile` = (VIX > high_threshold) OR (realized_vol > 25%)
- `is_trending` = (breadth > trending_threshold)

**Outputs:**
- `MarketRegime` enum state
- Strategy weight vector `{momentum, mean_reversion, pairs, tft}`
- `exposure_scalar` — inverse vol targeting, capped [0.3, 1.0]
- `confidence` — distance from regime boundaries

---

## 10. Portfolio Optimization

### PortfolioOptimizer (`strategies/ensemble/portfolio_optimizer.py`)

Converts combined signals into a risk-constrained portfolio target.

**Optimization Steps:**

1. **Score → Raw Weights**: weight ∝ |score| × confidence, normalized to gross=1.0
2. **Volatility Adjustment**: inverse-vol weighting per symbol (target: 20% per-asset vol)
3. **Regime Scaling**: multiply by `exposure_scalar` from regime detector
4. **Hard Constraints**:
   - Per-position cap: `max_gross_leverage / max_positions`
   - Minimum position: 0.5%
   - Max positions: top N by |weight|
   - Gross leverage cap (default: 2.0×)
   - Net leverage cap (default: 1.0×)
5. **Risk Metrics**: portfolio vol (covariance matrix), VaR99 (parametric)

**PortfolioTarget Output:**

```python
@dataclass
class PortfolioTarget:
    positions: List[PortfolioPosition]  # symbol, target_weight, direction
    gross_leverage: float               # sum(|weights|)
    net_leverage: float                 # sum(signed weights)
    expected_volatility: float          # annualized
    var_99: float                       # 1-day 99% VaR
    regime_exposure_scalar: float
```

### PortfolioRiskManager (`strategies/risk/portfolio_risk.py`)

Sits above the circuit breaker. Handles:

1. **VaR Calculation** — Parametric + historical (dual method)
2. **Strategy Correlation Monitoring** — Alert when strategies converge (threshold: 0.6)
3. **Per-Strategy Kill Switches** — Halt individual strategies on drawdown (>20%) or Sharpe (<-1.0)
4. **Dynamic Capital Allocation** — Shift capital toward performing strategies
5. **Portfolio Drawdown Monitoring** — Track peak-to-trough

---

## 11. Paper Trader Service

### Overview

`paper-trader/main.py` — FastAPI service on port 8010. ~760 lines of core logic plus dashboard HTML.

### Configuration (from environment)

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPER_TRADING_SYMBOLS` | AAPL,MSFT,...,XOM | Comma-separated stock tickers |
| `PAPER_FX_PAIRS` | EURUSD,...,USDCHF | Comma-separated FX pairs |
| `PAPER_SCHEDULE_HOUR` | 10 | Daily run hour (ET) |
| `PAPER_SCHEDULE_MINUTE` | 0 | Daily run minute |
| `PAPER_INITIAL_CAPITAL` | 100000 | Starting portfolio value |

### AppState

```python
@dataclass
class AppState:
    portfolio_value: float
    day_count: int
    last_run: Optional[datetime]
    last_pnl: float
    total_return_pct: float
    daily_returns: List[float]
    last_positions: List[Dict]
    last_regime: Optional[str]
    last_weights: Dict[str, float]
    run_log: List[Dict]
    is_running: bool
    enabled_strategies: List[str]
    circuit_breaker_tripped: bool
    scheduler: Optional[AsyncIOScheduler]
```

### Infrastructure Components

| Component | Class | Status |
|-----------|-------|--------|
| Broker | `AlpacaBroker` | Connected (paper account) |
| DB Pool | `ThreadedConnectionPool` | 2-10 connections |
| Circuit Breaker | `CircuitBreaker` | Initialized (Redis-backed) |
| Audit Logger | `AuditLogger` | Initialized (PostgreSQL) |
| Notifications | `NotificationManager` | Disabled (no webhook configured) |
| Signal Publisher | `SignalPublisher` | Connected (Redis) |
| Scheduler | `AsyncIOScheduler` | Active (10:00 ET Mon-Fri) |
| Safety Guardrails | `SignalVarianceGuard`, `LeverageGate`, `ExecutionFailureMonitor` | Active (env-configurable) |

### Daily Pipeline Function

`run_daily_pipeline()` — async function triggered by scheduler or `/run-now`:

1. Fetch OHLCV via yfinance (stocks + SPY + FX pairs, 300 days)
2. Circuit breaker pre-check
3. Detect market regime via `RegimeDetector`
4. Build strategies from `StrategyMasterConfig` via `build_strategies()`
5. Run each strategy: `initialize()` → `generate_signals()`
6. Risk assessment via `PortfolioRiskManager.assess()`
7. Combine via `EnsembleCombiner.combine()`
8. **GUARDRAIL: Signal variance check** — halt if scores collapse
9. Publish signals to Redis (fire-and-forget)
10. Optimize via `PortfolioOptimizer.optimize()`
11. **GUARDRAIL: Leverage gate** — skip orders if over limit
12. **GUARDRAIL: Execution failure monitor** — per-order failure rate check
13. Execute trades via `AlpacaBroker.submit_order()`
14. Log to PostgreSQL (trades, snapshots, signals)
15. Send notifications (Discord/Email)

---

## 12. Trading Infrastructure

### AlpacaBroker (`trading/broker/alpaca.py`)

Async HTTP client for Alpaca Markets REST API v2.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `get_account()` | GET /v2/account | Account info (equity, cash, buying power) |
| `get_positions()` | GET /v2/positions | All open positions |
| `get_position(ticker)` | GET /v2/positions/{ticker} | Single position |
| `submit_order(request)` | POST /v2/orders | Submit market/limit/stop order |
| `cancel_order(order_id)` | DELETE /v2/orders/{id} | Cancel pending order |
| `get_order(order_id)` | GET /v2/orders/{id} | Order status |
| `get_open_orders()` | GET /v2/orders?status=open | List open orders |
| `close_position(ticker)` | DELETE /v2/positions/{ticker} | Close specific position |
| `close_all_positions()` | DELETE /v2/orders + close each | Emergency liquidation (3 retries) |

**Key Design**: Uses `aiohttp.ClientSession` with 15-second timeout. Reads credentials from `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL`.

### CircuitBreaker (`trading/risk/circuit_breaker.py`)

Redis-backed drawdown circuit breaker. 405 lines.

**Drawdown Methods:**
- `HIGH_WATER_MARK` — trip if portfolio drops X% below peak
- `START_OF_DAY` — trip if intraday loss exceeds X%
- `INITIAL_CAPITAL` — trip if total loss from initial exceeds X%

**State Machine:**
```
NORMAL ──[drawdown breach]──> TRIPPED
TRIPPED ──[manual reset]───> NORMAL
```

**Trip Sequence:**
1. Set `circuit_breaker:is_tripped = "1"` in Redis (blocks concurrent trades)
2. Close all positions via `broker.close_all_positions()`
3. Log trip event to PostgreSQL audit trail
4. Log each position closure
5. Log portfolio snapshot
6. Send notifications (fire-and-forget)

**Redis Keys:**
- `circuit_breaker:is_tripped` — "0" or "1"
- `circuit_breaker:state` — JSON blob of `CircuitBreakerState`
- `circuit_breaker:high_water_mark` — float
- `circuit_breaker:start_of_day_value` — float (24h TTL)

**Monitor Loop:** Background asyncio task checks drawdown every `check_interval_seconds` (default: 30).

### AuditLogger (`trading/persistence/audit.py`)

PostgreSQL audit trail for all pipeline events. 277 lines.

**Tables Created:**
- `circuit_breaker_events` — trip/reset events with drawdown details
- `circuit_breaker_closures` — individual position closures during CB events
- `portfolio_snapshots` — periodic/trip/reset/SOD snapshots

### NotificationManager (`trading/notifications/alerts.py`)

Fire-and-forget alert system. 202 lines.

**Senders:**
- `DiscordWebhookSender` — Rich embeds with severity colors (blue/orange/red)
- `EmailSender` — SMTP via Gmail (configurable), with severity prefix in subject

**Alert Types:**
- Circuit breaker trip (critical)
- Circuit breaker reset (warning)
- Custom alerts (info/warning/critical)

### Position Sizing (`trading/risk/position_sizing.py`)

Three sizing strategies via `PositionSizerFactory`:

1. **Fixed Fractional** — shares = (portfolio × risk%) / (price × stop_loss%)
2. **Kelly Criterion** — Half-Kelly: position = 0.5 × (p×b - q)/b, capped
3. **Volatility Scaled** — shares = dollar_risk / (ATR × multiplier)

---

## 12b. Safety Guardrails

### Overview

`trading/safety/guardrails.py` — 5 independent pre-trade safety checks, added 2026-03-21 after March 10 incident analysis. All configurable via environment variables.

### Guardrail Classes

| # | Class | Purpose | Check Point | On Failure |
|---|-------|---------|-------------|------------|
| 1 | `SignalVarianceGuard` | Detect score collapse (all signals identical) | After ensemble combine | Halt pipeline + Discord alert |
| 2 | `LeverageGate` | Hard leverage limit before orders | After portfolio optimization | Skip order batch |
| 3 | `CalibrationHealthCheck` | Verify Platt/isotonic calibrators are fitted | Startup + before daily run | Log error, skip calibration |
| 4 | `ModelPromotionGate` | Min val Sharpe for model promotion | Before model goes live | Reject model |
| 5 | `ExecutionFailureMonitor` | Rolling failure rate monitor | Per-order during execution | Pause remaining orders + alert |

### Environment Variables

```bash
GUARDRAIL_SIGNAL_MIN_STD=0.01          # Min std dev of ensemble scores
GUARDRAIL_MAX_LEVERAGE=1.5             # Max gross leverage
GUARDRAIL_CALIBRATION_TOLERANCE=1e-6   # Identity detection tolerance
GUARDRAIL_MIN_PROMOTION_SHARPE=0.5     # Min val Sharpe for promotion
GUARDRAIL_MAX_EXEC_FAILURE_RATE=0.25   # Max 25% failure rate
GUARDRAIL_EXEC_WINDOW_SECONDS=3600     # 1-hour rolling window
```

### Integration Points in `paper-trader/main.py`

1. **Startup** (lifespan): `CalibrationHealthCheck.check_platt()` / `.check_generic()` on loaded models
2. **Line ~542**: `signal_guard.check(scores)` after `combiner.combine()` — aborts pipeline if variance collapses
3. **Line ~599**: `leverage_gate.check(target.gross_leverage)` after `optimizer.optimize()` — skips orders
4. **Line ~632**: `exec_monitor.check()` before each order — pauses on high failure rate
5. **Line ~685**: `exec_monitor.record(success)` after each `broker.submit_order()`

---

## 13. Database Schema (TimescaleDB)

### Infrastructure

- **Engine**: TimescaleDB (PostgreSQL 15.13)
- **Container**: `apex-timescaledb`
- **Port**: 15432
- **Database**: `apex`
- **Credentials**: `apex_user` / `apex_pass`

### Tables (26 total)

#### Core Data Tables

| Table | Purpose | Row Count |
|-------|---------|-----------|
| `ohlcv_bars` | Intraday price bars (multi-source) | 798,225 |
| `forex_candles` | FX price data | — |
| `forex_cot` | Commitment of Traders data | — |
| `forex_economic_calendar` | Economic events | — |
| `market_raw_minute` | Raw minute bars | — |
| `features` | Computed feature store | — |
| `iv_surface` | Implied volatility surface | — |
| `options_chains` | Options chain data | — |
| `options_flow_features` | Options flow analytics | — |

#### Paper Trading Tables

| Table | Purpose | Row Count |
|-------|---------|-----------|
| `paper_trades` | Individual trade records | 15 |
| `paper_daily_snapshots` | End-of-day portfolio state | 4 |
| `paper_strategy_signals` | Per-strategy signal records | 70 |

#### Trading Infrastructure Tables

| Table | Purpose | Row Count |
|-------|---------|-----------|
| `orders` | Order records | — |
| `positions` | Position tracking | — |
| `forex_positions` | FX position tracking | — |
| `options_positions` | Options position tracking | — |
| `signals` | Raw signal records | — |
| `signals_scored` | Scored signal records | — |
| `signal_attribution` | Signal source attribution | — |

#### Risk & Audit Tables

| Table | Purpose | Row Count |
|-------|---------|-----------|
| `circuit_breaker_events` | CB trip/reset events | 0 |
| `circuit_breaker_closures` | Positions closed during CB events | — |
| `portfolio_snapshots` | Portfolio state snapshots | 0 |
| `risk_events` | Risk event records | — |
| `calibration_snapshots` | Model calibration records | — |
| `decision_records` | Decision audit trail | — |
| `model_performance` | Model performance metrics | — |

---

## 14. Redis Usage

### Connection

- **URL**: `redis://localhost:6379`
- **Client**: `redis.asyncio` for circuit breaker, `redis.Redis` (sync) for signal publisher

### Key Namespaces

| Key/Channel | Type | Purpose |
|-------------|------|---------|
| `circuit_breaker:is_tripped` | String | "0" or "1" — O(1) pre-trade check |
| `circuit_breaker:state` | String (JSON) | Full CB state blob |
| `circuit_breaker:high_water_mark` | String | Peak portfolio value |
| `circuit_breaker:start_of_day_value` | String (24h TTL) | SOD portfolio value |
| `apex:signals:stock` | Pub/Sub channel | Equity ensemble signals |
| `apex:signals:forex` | Pub/Sub channel | FX pair signals |
| `apex:signals:options` | Pub/Sub channel | Options/vol signals |
| `apex:signals:risk` | Pub/Sub channel | Tail risk index broadcasts |

### Signal Payload Schema

```json
{
  "ts": "2026-03-20T21:15:20.243Z",
  "symbol": "AAPL",
  "score": 0.342156,
  "confidence": 0.7512,
  "direction": "long",
  "sources": {
    "cross_sectional_momentum": 0.45,
    "sector_rotation": 0.28
  }
}
```

### Channel Routing Logic

- Symbol in `{EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF}` → `forex`
- Majority contribution from `{deep_surrogates, tdgf}` → `options`
- Everything else → `stock`

---

## 15. Kafka Messaging

### Topics (13 total)

| Topic | Producer | Consumer(s) | Description |
|-------|----------|-------------|-------------|
| `market-data` | data-ingestion | tft-predictor, orchestrator | OHLCV snapshots |
| `reddit-comments` | data-ingestion | sentiment-engine | Reddit posts |
| `earnings-calendar` | data-ingestion | orchestrator | Earnings dates |
| `sentiment-scores` | sentiment-engine | orchestrator, trading-engine | Sentiment values |
| `tft-predictions` | tft-predictor | orchestrator | Model predictions |
| `trading-signals` | orchestrator | trading-engine | Buy/sell signals |
| `order-updates` | trading-engine | orchestrator | Order fills/rejections |
| `portfolio-updates` | trading-engine | orchestrator | Position changes |
| `system-events` | multiple | orchestrator | Errors, warnings |
| `system-health` | all services | monitoring | Health heartbeats |
| `model-updates` | tft-predictor | logging | Model versioning |
| `workflow-events` | orchestrator | audit | Saga state transitions |
| `health-checks` | orchestrator | monitoring | Service health |

Kafka is used by the microservices layer only. The paper trader operates independently via direct API calls.

### Kafka Broker Configuration

KRaft mode (no Zookeeper), Confluent CP 7.6.0:

| Setting | Value | Env Var |
|---------|-------|---------|
| Log retention (hours) | 168 (7 days) | `KAFKA_LOG_RETENTION_HOURS` |
| Log retention (bytes) | 5,368,709,120 (5 GB) | `KAFKA_LOG_RETENTION_BYTES` |
| Log segment size | 1,073,741,824 (1 GB) | `KAFKA_LOG_SEGMENT_BYTES` |

### Dead Letter Queue (`services/common/dlq.py`)

All 4 Kafka consumers persist failed messages to a PostgreSQL `dead_letter_queue` table for retry.

**Status Lifecycle:** `PENDING` -> `RETRYING` -> `RESOLVED` | `EXHAUSTED`

**Exponential Backoff:**
- Base delay: 1s, multiplier: 2x, max delay: 60s, jitter: 0-25%
- Max retries: 5 (configurable per instance)
- Background retry worker polls every 30s (daemon thread)
- Uses `FOR UPDATE SKIP LOCKED` for concurrent safety

**Integration:** `sentiment-engine`, `trading-engine`, `tft-predictor`, `orchestrator` each instantiate their own `DeadLetterQueue` with a service-specific `source_service` tag.

### Schema Registry (`microservices/schema_registry.py`)

Confluent Schema Registry client with thread-safe singleton connection cache:
- `get_schema_registry_client()` — returns cached instance, creates on first call
- Exponential backoff retry on connection failure (max 3 retries, base 1s)
- `reset_client()` for testing

---

## 16. Microservices Layer

Five FastAPI services coordinated via Kafka and Redis. Deployed via `docker-compose.yml`.

| Service | Port | Role | Key Deps |
|---------|------|------|----------|
| `data-ingestion` | 8001 | Polygon.io OHLCV + Reddit data → Kafka | Polygon API, PRAW |
| `sentiment-engine` | 8002 | NLP sentiment scoring | FinBERT, VADER |
| `tft-predictor` | 8003 | GPU inference, model versioning | PyTorch, MLflow |
| `trading-engine` | 8004 | Alpaca order execution | Alpaca API |
| `orchestrator` | 8005 | Saga-pattern workflow coordination | All topics |

### Docker Compose Infrastructure

| Service | Port | Image |
|---------|------|-------|
| PostgreSQL (TimescaleDB) | 5432 | timescale/timescaledb:latest-pg15 |
| Redis | 6379 | redis:7-alpine |
| Kafka Broker | 9092 | confluentinc/cp-kafka:7.6.0 (KRaft) |
| Schema Registry | 8081 | confluentinc/cp-schema-registry:7.6.0 |
| MLflow | 5000 | ghcr.io/mlflow/mlflow |
| Prometheus | 9090 | prom/prometheus |
| Grafana | 3000 | grafana/grafana |
| Kafka UI | 8080 | provectuslabs/kafka-ui |
| Redis Commander | 8082 | rediscommander/redis-commander |

**Note:** Requires external network `tft_network` (`docker network create tft_network`).

---

## 17. Configuration & Environment

### Environment Files

| File | Purpose |
|------|---------|
| `.env.template` | Complete template with all variables and comments |
| `.env.example` | Minimal example for quick start |
| `.env` | Active configuration (gitignored) |

### Key Configuration Groups

**API Keys:**
```bash
POLYGON_API_KEY=            # Market data
ALPACA_API_KEY=             # Paper trading
ALPACA_SECRET_KEY=          # Paper trading
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

**Database (Paper Trader uses DB_ prefix):**
```bash
DB_HOST=localhost
DB_PORT=15432
DB_NAME=apex
DB_USER=apex_user
DB_PASSWORD=apex_pass
```

**Database (train_postgres.py uses POSTGRES_ prefix):**
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=15432
POSTGRES_DB=apex
POSTGRES_USER=apex_user
POSTGRES_PASSWORD=<your_secure_password>
```

**Strategy Configs** — Each strategy has a config dataclass in `strategies/config.py` with `from_env()` classmethod. All use `_env_bool`, `_env_float`, `_env_int` helpers. `StrategyMasterConfig.from_env()` loads all sub-configs.

**Circuit Breaker:**
```bash
CIRCUIT_BREAKER_ENABLED=true
CB_DRAWDOWN_METHODS=high_water_mark:5.0,start_of_day:3.0
CB_CHECK_INTERVAL_SECONDS=30
CB_INITIAL_CAPITAL=100000.0
```

**Notifications:**
```bash
DISCORD_WEBHOOK_URL=        # Discord channel webhook
EMAIL_USER=                 # SMTP username (Gmail)
EMAIL_PASSWORD=             # SMTP password / app password
EMAIL_TO=                   # Recipient address
ALERT_COOLDOWN_MINUTES=15
```

---

## 18. API Endpoints

### Paper Trader (Port 8010)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System status: models, strategies, infrastructure, portfolio |
| POST | `/run-now` | Manually trigger daily pipeline |
| GET | `/positions` | Current portfolio positions with unrealized P&L |
| GET | `/history` | Last 30 pipeline run records |
| GET | `/weights` | Current strategy weight distribution |
| GET | `/dashboard` | Live HTML dashboard with all strategy panels |

### Legacy API (Port 8000, `api.py`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Model/preprocessor status |
| GET | `/model/status` | Training config and metrics |
| POST | `/predict` | Generate predictions for symbols |
| POST | `/train` | Train/retrain model (background) |
| POST | `/predict/batch` | Batch prediction from CSV |

### PostgreSQL API (Port 8000, `api_postgres.py`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API version info |
| GET | `/health` | DB connection status |
| GET | `/symbols` | List available symbols |
| GET | `/symbols/{symbol}/info` | Symbol metadata and date range |
| POST | `/validate-data` | Data quality validation |
| POST | `/predict` | Not implemented (501) |
| DELETE | `/cache/clear` | Clear prediction cache |
| GET | `/cache/stats` | Cache statistics |

---

## 19. Scheduling

### APScheduler Configuration

The paper trader uses `AsyncIOScheduler` from APScheduler, configured in the FastAPI lifespan handler:

```python
scheduler = AsyncIOScheduler()
scheduler.add_job(
    run_daily_pipeline,
    "cron",
    hour=PAPER_SCHEDULE_HOUR,    # default: 10
    minute=PAPER_SCHEDULE_MINUTE, # default: 0
    day_of_week="mon-fri",
    timezone="US/Eastern",
)
scheduler.start()
```

**Schedule**: Daily at 10:00 ET, Monday through Friday.

The scheduler runs inside the FastAPI process. No external cron or systemd service is needed — as long as the paper trader uvicorn process is running, the scheduler fires.

### Manual Trigger

`POST /run-now` triggers `run_daily_pipeline()` immediately via `asyncio.create_task()`. Returns `{"status": "already_running"}` if a pipeline is in progress.

---

## 20. Model Training Pipeline

### TFT Stocks Training Flow

```
1. TFTPostgresModel(db_config, config)
2. model.train(symbols, start_date)
   ├── PostgresDataPipeline.build_dataset()
   │   ├── PostgresDataLoader.load_ohlcv()  → queries ohlcv VIEW
   │   ├── Add technical indicators (SMA, EMA, RSI, MACD, BB, OBV)
   │   ├── Add temporal features (cyclical encoding)
   │   ├── Merge fundamentals (0-filled if missing)
   │   ├── Merge sentiment (0-filled if missing)
   │   ├── Create target: pct_change(5).shift(-5) for 5-day returns
   │   ├── Add time_idx, group_id
   │   ├── Clean: ffill/bfill, drop NaN targets (last 5 rows/symbol)
   │   └── Filter: min 100 observations per symbol
   ├── Split: training_cutoff = max_time_idx × (1 - validation_split)
   ├── Create TimeSeriesDataSet (pytorch-forecasting)
   ├── Create TemporalFusionTransformer.from_dataset()
   │   ├── hidden_size=64, attention_heads=4, lstm_layers=2
   │   ├── QuantileLoss(quantiles=[0.1, 0.5, 0.9])
   │   └── optimizer="adamw"
   ├── Train via lightning.pytorch.Trainer
   │   ├── GPU auto-detect (RTX 5090 available)
   │   ├── EarlyStopping(patience=10, monitor="val_loss")
   │   └── ModelCheckpoint(save_top_k=1)
   └── Save checkpoint dict:
       {config, model_state_dict, training_dataset}
       → models/tft_model.pth (3.3MB)
```

### Checkpoint Format

```python
torch.save({
    'config': dict,                    # model hyperparameters
    'model_state_dict': OrderedDict,   # 1055 parameter tensors
    'training_dataset': TimeSeriesDataSet,  # for architecture reconstruction
}, 'models/tft_model.pth')
```

The `TFTStocksAdapter.load()` handles both formats:
- **Legacy** (`loss_type` in config): Uses `EnhancedTFTModel.create_model()`
- **Postgres** (no `loss_type`): Directly calls `TemporalFusionTransformer.from_dataset()` + `load_state_dict()`

### Training Config Defaults

```python
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

## 21. Testing

### Test Suite: 635 tests, 30 files

```
tests/
├── models/
│   ├── test_sentiment_model.py       # 10 tests (FinBERT/VADER, scoring, edge cases)
│   ├── test_mean_reversion_model.py  # 10 tests (Hurst, OU params, unit tests)
│   ├── test_macro_model.py           #  8 tests (regime classification, sector tilts)
│   └── test_microstructure_model.py  # 10 tests (volume, VWAP, CLV, AD line)
├── strategies/
│   ├── test_mean_reversion.py        #  7 tests (with mean-reverting synthetic data)
│   ├── test_sector_rotation.py       #  6 tests (multi-sector data, fallback)
│   ├── test_fx_momentum.py           #  8 tests (multi-lookback, max_pairs limit)
│   ├── test_fx_vol_breakout.py       #  7 tests (synthetic squeeze data)
│   └── test_vol_arb_integration.py   #  7 tests (BaseStrategy compliance)
├── integration/
│   ├── test_full_pipeline.py         #  4 tests (all strategies through ensemble)
│   └── test_circuit_breaker.py       #  5 tests (import/config tests)
├── test_guardrails.py                # 33 tests (all 5 guardrails)
├── test_production_hardening.py      # 38 tests (Kafka retention, TimescaleDB, schema registry)
├── test_security_hardening.py        # 22 tests (paths, passwords, secrets, env validator)
├── test_dlq.py                       # 39 tests (DLQ persistence, backoff, retry, integration)
├── test_walk_forward.py              # walk-forward cross-validation
├── test_risk_wiring.py               # risk manager integration
├── test_bug_fixes.py                 # bug fix regression tests
├── test_bug_fixes_final.py           # final bug fix regression tests
├── test_ci_pipeline.py               # CI/CD pipeline tests
├── test_bayesian_updater.py          # Bayesian ensemble updater
├── test_sentiment_strategy.py        # sentiment strategy tests
├── test_signal_analyst.py            # signal analysis tests
├── test_signal_provider.py           # signal provider tests
├── test_vwap_execution.py            # VWAP execution tests
└── test_prometheus_metrics.py        # 69 tests (needs prometheus_client)
```

**Run:** `pytest tests/ -v` — 566 tests pass (69 prometheus tests require `pip install prometheus_client`).

---

## 22. Data Retention & TimescaleDB

### Hypertables

Four tables converted to TimescaleDB hypertables for time-series optimization:
- `ohlcv_bars` (partitioned on `time`)
- `paper_risk_reports` (partitioned on `created_at`)
- `paper_execution_stats` (partitioned on `created_at`)
- `paper_signal_analyses` (partitioned on `created_at`)

### Retention Policies

| Table | Retention | Interval |
|-------|-----------|----------|
| `ohlcv_bars` | 365 days | Drop chunks older than 1 year |
| `paper_risk_reports` | 90 days | Drop chunks older than 3 months |
| `paper_execution_stats` | 90 days | Drop chunks older than 3 months |
| `paper_signal_analyses` | 90 days | Drop chunks older than 3 months |

### Continuous Aggregates

Three materialized views with automatic refresh policies:

| View | Aggregation | Refresh Lag | Refresh Window |
|------|-------------|-------------|----------------|
| `ohlcv_15m` | 15-minute OHLCV bars | 1 hour | 2 days |
| `ohlcv_1h` | 1-hour OHLCV bars | 2 hours | 3 days |
| `ohlcv_1d` | Daily OHLCV bars | 1 day | 7 days |

## 22b. Security Hardening

### Credential Management
- All database passwords use `os.environ['VAR']` (raises `KeyError` on missing) — no defaults
- docker-compose uses `${VAR:?error_message}` for required credentials
- `setup_postgres.sh` uses placeholder values only (real keys removed)

### Startup Validation (`utils/env_validator.py`)
- Required vars: `DB_PASSWORD`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- Placeholder detection: rejects `your_*`, `CHANGE_ME`, `changeme`, `xxx` patterns
- Called in paper-trader lifespan with `strict=True` (exits on failure)

### `.env` Security
- `.env` is gitignored
- `.env.example` contains safe placeholders only
- No hardcoded machine paths in codebase

---

## 23. Deployment Status

### Current State (2026-03-21)

#### Running Services

| Service | Status | Details |
|---------|--------|---------|
| Paper Trader | **Running** on port 8010 | v3.0.0 |
| TimescaleDB | **Running** on port 15432 | 798K OHLCV bars, 26 tables |
| Redis | **Running** on port 6379 | Signal publishing active |
| APScheduler | **Active** | Daily 10:00 ET Mon-Fri |

#### Model Status

| Model | Status | Notes |
|-------|--------|-------|
| TFT Stocks | **Loaded** | 3.3MB, trained on 7 symbols |
| TFT Forex | **Loaded** | 1.6MB, pre-trained |
| TFT Volatility | **Loaded** | 2.7MB, pre-trained |
| Sentiment | **Loaded** | VADER fallback (install `transformers` for FinBERT) |
| Mean Reversion | **Loaded** | Statistical, no weights needed |
| Macro Regime | **Loaded** | Rule-based, no weights needed |
| Microstructure | **Loaded** | Statistical, no weights needed |
| Kronos | Not loaded | Requires `/opt/kronos` repo clone |
| Deep Surrogates | Not loaded | Requires `/opt/deep_surrogate` repo |
| TDGF | Not loaded | Requires `/opt/tdgf` repo |

#### Strategy Status

| Strategy | Status | Last Signal Count |
|----------|--------|-------------------|
| Cross-Sectional Momentum | **Active** | 3 (3L/0S) |
| Pairs Trading | **Active** | 0 (10 pairs tracked) |
| FX Carry + Trend | **Active** | 4 (2L/2S) |
| FX Momentum | **Active** | 4 (2L/2S) |
| FX Vol Breakout | **Active** | 0 (no squeeze) |
| Mean Reversion | **Active** | 0 (no triggers) |
| Sector Rotation | **Active** | 10 (8L/2S) |
| Kronos | Disabled | Needs repo |
| Deep Surrogates | Disabled | Needs repo |
| TDGF | Disabled | Needs repo |
| Vol Arb | Disabled | Env var false |

#### Infrastructure Status

| Component | Status | Notes |
|-----------|--------|-------|
| AlpacaBroker | **Connected** | Paper account PA31CZRY8R5V, $89,744 equity |
| Circuit Breaker | **Initialized** | Not tripped, Redis-backed |
| Audit Logger | **Initialized** | PostgreSQL audit trail active |
| DB Pool | **Active** | 2-10 ThreadedConnectionPool |
| Signal Publisher | **Active** | 13 signals published, 0 errors |
| Safety Guardrails | **Active** | 5 checks: signal variance, leverage, calibration, promotion, exec failure |
| Notifications | **Disabled** | No Discord webhook or email configured |

#### What Needs Setup

1. **Discord Notifications** — Set `DISCORD_WEBHOOK_URL` in `.env`
2. **Email Notifications** — Set `EMAIL_USER`, `EMAIL_PASSWORD`, `EMAIL_TO` in `.env`
3. **FinBERT Sentiment** — `pip install transformers` for higher-quality sentiment scoring
4. **External Models** — Clone repos to `/opt/` for Kronos, Deep Surrogates, TDGF
5. **Microservices** — `docker-compose up` for Kafka-based distributed deployment
6. **Monitoring** — Prometheus (9090) and Grafana (3000) available via docker-compose

---

*Version 3.0.0. ~179 Python files, 10 models, 11 strategies, 635 tests, 26+ database tables, 13 Kafka topics, 4 Redis channels.*
