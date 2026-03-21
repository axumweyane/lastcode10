"""
APEX Paper Trading Runner — Daily Ensemble Execution Service.

Runs the full multi-strategy pipeline on a schedule:
  1. Fetch latest market data (yfinance)
  2. Run all enabled strategies (11 total across stocks, FX, options)
  3. Detect market regime
  4. Combine signals via Bayesian ensemble
  5. Risk assessment via PortfolioRiskManager
  6. Optimize portfolio with risk constraints
  7. Circuit breaker check before trade execution
  8. Execute trades via Alpaca (production AlpacaBroker)
  9. Audit trail logging
 10. Log everything to PostgreSQL
 11. Send reports via NotificationManager (Discord + Email)
 12. Serve a live dashboard at /dashboard

Port: 8010
"""

import asyncio
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
from datetime import datetime, date, timezone, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.pool
from psycopg2.extras import Json
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Add parent directory to path for strategy imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Strategies (11 total)
from strategies.momentum.cross_sectional import CrossSectionalMomentum
from strategies.statarb.pairs import PairsTrading
from strategies.fx.carry_trend import FXCarryTrend
from strategies.fx.momentum import FXMomentumStrategy
from strategies.fx.vol_breakout import FXVolBreakoutStrategy
from strategies.kronos.strategy import KronosStrategy
from strategies.deep_surrogates.strategy import DeepSurrogateStrategy
from strategies.tdgf.strategy import TDGFStrategy
from strategies.stocks.mean_reversion import MeanReversionStrategy
from strategies.stocks.sector_rotation import SectorRotationStrategy
from strategies.options.strategies.vol_arb import VolatilityArbitrage
from strategies.sentiment.strategy import SentimentStrategy

# Ensemble & Infrastructure
from strategies.ensemble.bayesian_updater import BayesianWeightUpdater
from strategies.ensemble.combiner import EnsembleCombiner, TFTAdapter
from strategies.ensemble.portfolio_optimizer import PortfolioOptimizer
from strategies.regime.detector import RegimeDetector
from strategies.risk.portfolio_risk import PortfolioRiskManager
from strategies.signals.publisher import SignalPublisher

# Production trading infrastructure
from trading.broker.alpaca import AlpacaBroker
from trading.broker.base import OrderRequest, OrderResult, OrderSide, OrderStatus
from trading.execution.vwap import VWAPExecutionModel, VWAPExecutionResult, VolumeProfileCache
from trading.notifications.alerts import NotificationManager, AlertMessage, DiscordWebhookSender, EmailSender
from trading.persistence.audit import AuditLogger
from trading.risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, DrawdownConfig, DrawdownMethod
from trading.safety.guardrails import (
    SignalVarianceGuard, LeverageGate, CalibrationHealthCheck,
    ExecutionFailureMonitor,
)

# LLM signal analyst
from agents.signal_analyst import SignalAnalyst, SignalAnalysis

# Signal provider API
from api.signal_provider import create_signal_api, SignalCache

# Prometheus metrics
from monitoring.metrics import PrometheusMetrics

# Configuration
from strategies.config import (
    MomentumConfig, StatArbConfig, EnsembleConfig, RegimeConfig, FXConfig,
    KronosConfig, DeepSurrogateConfig, TDGFConfig,
    MeanReversionConfig, SectorRotationConfig, FXMomentumConfig, FXVolBreakoutConfig,
    SentimentConfig, StrategyMasterConfig,
)
from strategies.options.config import VolArbConfig
from strategies.base import StrategyPerformance
from models.manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("paper-trader")

# Configuration from environment
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "tft_trading")
DB_USER = os.getenv("DB_USER", "tft_user")
DB_PASSWORD = os.environ["DB_PASSWORD"]
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
TRADING_SYMBOLS = os.getenv(
    "PAPER_TRADING_SYMBOLS",
    "AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,JPM,BAC,XOM",
).split(",")
FX_PAIRS = os.getenv("PAPER_FX_PAIRS", "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF").split(",")
SCHEDULE_HOUR = int(os.getenv("PAPER_SCHEDULE_HOUR", "10"))  # 10 AM ET
SCHEDULE_MINUTE = int(os.getenv("PAPER_SCHEDULE_MINUTE", "0"))
INITIAL_CAPITAL = float(os.getenv("PAPER_INITIAL_CAPITAL", "100000"))

# ============================================================
# DATABASE SCHEMA
# ============================================================
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id              SERIAL PRIMARY KEY,
    trade_date      DATE NOT NULL,
    symbol          VARCHAR(16) NOT NULL,
    side            VARCHAR(8) NOT NULL,
    quantity        DOUBLE PRECISION,
    price           DOUBLE PRECISION,
    order_id        VARCHAR(64),
    strategy_source VARCHAR(64),
    signal_score    DOUBLE PRECISION,
    signal_confidence DOUBLE PRECISION,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_daily_snapshots (
    id              SERIAL PRIMARY KEY,
    snapshot_date   DATE NOT NULL,
    portfolio_value DOUBLE PRECISION,
    cash            DOUBLE PRECISION,
    equity          DOUBLE PRECISION,
    daily_pnl       DOUBLE PRECISION,
    daily_return_pct DOUBLE PRECISION,
    total_return_pct DOUBLE PRECISION,
    positions_count INTEGER,
    regime          VARCHAR(32),
    strategy_weights JSONB DEFAULT '{}',
    positions       JSONB DEFAULT '{}',
    risk_metrics    JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_strategy_signals (
    id              SERIAL PRIMARY KEY,
    signal_date     DATE NOT NULL,
    strategy_name   VARCHAR(64) NOT NULL,
    symbol          VARCHAR(16) NOT NULL,
    score           DOUBLE PRECISION,
    confidence      DOUBLE PRECISION,
    direction       VARCHAR(16),
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_risk_reports (
    id              SERIAL PRIMARY KEY,
    report_time     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    drawdown        DOUBLE PRECISION,
    var_99          DOUBLE PRECISION,
    cvar_95         DOUBLE PRECISION,
    sharpe_21       DOUBLE PRECISION,
    killed_strategies TEXT[] DEFAULT '{}',
    correlation_alerts JSONB DEFAULT '[]',
    portfolio_breached BOOLEAN DEFAULT FALSE,
    raw_json        JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS bayesian_weight_state (
    strategy_name   VARCHAR(64) PRIMARY KEY,
    alpha           DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    beta            DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    n_updates       INTEGER NOT NULL DEFAULT 0,
    state_json      JSONB DEFAULT '{}',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_execution_stats (
    id              SERIAL PRIMARY KEY,
    execution_time  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ticker          VARCHAR(10) NOT NULL,
    side            VARCHAR(4) NOT NULL,
    total_requested DOUBLE PRECISION NOT NULL,
    total_filled    DOUBLE PRECISION NOT NULL DEFAULT 0,
    expected_price  DOUBLE PRECISION,
    filled_avg_price DOUBLE PRECISION,
    slippage_bps    DOUBLE PRECISION DEFAULT 0,
    fill_rate       DOUBLE PRECISION DEFAULT 0,
    num_slices      INTEGER DEFAULT 0,
    used_fallback   BOOLEAN DEFAULT FALSE,
    adv_capped      BOOLEAN DEFAULT FALSE,
    elapsed_s       DOUBLE PRECISION DEFAULT 0,
    slices_json     JSONB DEFAULT '[]',
    error           TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_pt_date ON paper_trades(trade_date);
CREATE INDEX IF NOT EXISTS idx_pds_date ON paper_daily_snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_pss_date ON paper_strategy_signals(signal_date);
CREATE INDEX IF NOT EXISTS idx_prr_time ON paper_risk_reports(report_time);
CREATE INDEX IF NOT EXISTS idx_exec_stats_time ON paper_execution_stats(execution_time);
CREATE INDEX IF NOT EXISTS idx_exec_stats_ticker ON paper_execution_stats(ticker);

CREATE TABLE IF NOT EXISTS paper_signal_analyses (
    id              SERIAL PRIMARY KEY,
    analysis_time   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    regime          VARCHAR(32),
    summary         TEXT NOT NULL,
    patterns        TEXT,
    confidence      VARCHAR(8),
    flags_json      JSONB DEFAULT '{}',
    top_signals     JSONB DEFAULT '[]',
    model_used      VARCHAR(64),
    latency_s       DOUBLE PRECISION DEFAULT 0,
    raw_json        JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_signal_analyses_time ON paper_signal_analyses(analysis_time);
"""


# ============================================================
# GLOBAL STATE
# ============================================================
class AppState:
    def __init__(self):
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.last_run: Optional[datetime] = None
        self.last_regime: Optional[str] = None
        self.last_weights: Dict[str, float] = {}
        self.last_positions: List[Dict] = []
        self.last_pnl: float = 0.0
        self.total_return_pct: float = 0.0
        self.portfolio_value: float = INITIAL_CAPITAL
        self.day_count: int = 0
        self.daily_returns: List[float] = []
        self.run_log: List[Dict] = []
        self.is_running: bool = False
        self.enabled_strategies: List[str] = []
        self.circuit_breaker_tripped: bool = False


state = AppState()
model_manager: Optional[ModelManager] = None
signal_publisher: Optional[SignalPublisher] = None
notification_manager: Optional[NotificationManager] = None
audit_logger: Optional[AuditLogger] = None
circuit_breaker: Optional[CircuitBreaker] = None
db_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
broker: Optional[AlpacaBroker] = None
risk_manager: Optional[PortfolioRiskManager] = None
bayesian_updater: Optional[BayesianWeightUpdater] = None
USE_BAYESIAN_WEIGHTS = os.getenv("ENSEMBLE_USE_BAYESIAN_WEIGHTS", "false").lower() in ("true", "1", "yes")
USE_VWAP_EXECUTION = os.getenv("EXECUTION_USE_VWAP", "false").lower() in ("true", "1", "yes")
vwap_model: Optional[VWAPExecutionModel] = None
LLM_ANALYST_ENABLED = os.getenv("LLM_ANALYST_ENABLED", "false").lower() in ("true", "1", "yes")
signal_analyst: Optional[SignalAnalyst] = None
SIGNAL_API_ENABLED = os.getenv("SIGNAL_API_ENABLED", "false").lower() in ("true", "1", "yes")
SIGNAL_API_KEY = os.getenv("SIGNAL_API_KEY", "")
signal_cache = SignalCache()
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() in ("true", "1", "yes")
prom_metrics: Optional[PrometheusMetrics] = PrometheusMetrics() if METRICS_ENABLED else None

# Safety guardrails (March 10 incident prevention)
signal_guard = SignalVarianceGuard()
leverage_gate = LeverageGate()
exec_monitor = ExecutionFailureMonitor()


# ============================================================
# DATABASE (connection pooling)
# ============================================================
def get_db_conn():
    if db_pool is not None:
        return db_pool.getconn()
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )


def return_db_conn(conn):
    if db_pool is not None:
        try:
            db_pool.putconn(conn)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
    else:
        try:
            conn.close()
        except Exception:
            pass


def init_db():
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
        return_db_conn(conn)
        logger.info("Database schema initialized")
    except Exception as e:
        logger.warning("Database init failed (will retry on first run): %s", e)


def _safe_db_write(operation_name: str, query: str, params: tuple) -> bool:
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(query, params)
        conn.commit()
        return True
    except Exception as e:
        logger.error("DB %s failed: %s", operation_name, e)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return False
    finally:
        if conn:
            return_db_conn(conn)


def log_trade(trade_date, symbol, side, quantity, price, order_id, strategy, score, confidence, metadata):
    _safe_db_write(
        "log_trade",
        """INSERT INTO paper_trades
           (trade_date, symbol, side, quantity, price, order_id,
            strategy_source, signal_score, signal_confidence, metadata)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (trade_date, symbol, side, quantity, price, order_id,
         strategy, score, confidence, Json(metadata or {})),
    )


def log_daily_snapshot(snapshot):
    _safe_db_write(
        "log_snapshot",
        """INSERT INTO paper_daily_snapshots
           (snapshot_date, portfolio_value, cash, equity, daily_pnl,
            daily_return_pct, total_return_pct, positions_count,
            regime, strategy_weights, positions, risk_metrics)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (snapshot["date"], snapshot["portfolio_value"], snapshot["cash"],
         snapshot["equity"], snapshot["daily_pnl"], snapshot["daily_return_pct"],
         snapshot["total_return_pct"], snapshot["positions_count"],
         snapshot["regime"], Json(snapshot.get("strategy_weights", {})),
         Json(snapshot.get("positions", [])), Json(snapshot.get("risk_metrics", {}))),
    )


def log_signals(signal_date, strategy_name, signals_df):
    if signals_df.empty:
        return
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            for _, row in signals_df.iterrows():
                cur.execute(
                    """INSERT INTO paper_strategy_signals
                       (signal_date, strategy_name, symbol, score, confidence, direction, metadata)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (signal_date, strategy_name, row.get("symbol", ""),
                     row.get("score", 0), row.get("confidence", 0),
                     row.get("direction", ""), Json({})),
                )
        conn.commit()
    except Exception as e:
        logger.error("Failed to log %s signals: %s", strategy_name, e)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
    finally:
        if conn:
            return_db_conn(conn)


def log_risk_report(report) -> bool:
    """Log a RiskReport to the paper_risk_reports table."""
    return _safe_db_write(
        "log_risk_report",
        """INSERT INTO paper_risk_reports
           (report_time, drawdown, var_99, cvar_95, sharpe_21,
            killed_strategies, correlation_alerts, portfolio_breached, raw_json)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (report.timestamp, report.portfolio_drawdown,
         report.var.parametric_var, report.var.cvar_95,
         report.portfolio_sharpe_21d,
         report.killed_strategies or [],
         Json([{"a": a.strategy_a, "b": a.strategy_b, "corr": a.correlation}
               for a in report.correlation_alerts]),
         report.portfolio_breached,
         Json(report.to_dict())),
    )


def load_historical_returns(days: int = 30) -> List[float]:
    """Load recent daily returns from paper_daily_snapshots to seed the risk manager."""
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT daily_return_pct FROM paper_daily_snapshots
                   ORDER BY snapshot_date DESC LIMIT %s""",
                (days,),
            )
            rows = cur.fetchall()
        return_db_conn(conn)
        conn = None
        # Reverse to chronological order, convert from pct to decimal
        return [row[0] / 100.0 for row in reversed(rows)] if rows else []
    except Exception as e:
        logger.warning("Failed to load historical returns: %s", e)
        return []
    finally:
        if conn:
            return_db_conn(conn)


def save_bayesian_state(updater: BayesianWeightUpdater) -> bool:
    """Persist Bayesian weight state to PostgreSQL."""
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            for row in updater.save_to_rows():
                cur.execute(
                    """INSERT INTO bayesian_weight_state
                       (strategy_name, alpha, beta, n_updates, state_json, updated_at)
                       VALUES (%s, %s, %s, %s, %s, NOW())
                       ON CONFLICT (strategy_name) DO UPDATE SET
                           alpha = EXCLUDED.alpha,
                           beta = EXCLUDED.beta,
                           n_updates = EXCLUDED.n_updates,
                           state_json = EXCLUDED.state_json,
                           updated_at = NOW()""",
                    row,
                )
        conn.commit()
        return_db_conn(conn)
        conn = None
        return True
    except Exception as e:
        logger.warning("Failed to save Bayesian state: %s", e)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return False
    finally:
        if conn:
            return_db_conn(conn)


def load_bayesian_state(updater: BayesianWeightUpdater) -> int:
    """Load Bayesian weight state from PostgreSQL. Returns number of strategies loaded."""
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT strategy_name, alpha, beta, n_updates, state_json FROM bayesian_weight_state"
            )
            rows = cur.fetchall()
        return_db_conn(conn)
        conn = None
        if rows:
            updater.load_from_rows(rows)
        return len(rows)
    except Exception as e:
        logger.warning("Failed to load Bayesian state: %s", e)
        return 0
    finally:
        if conn:
            return_db_conn(conn)


def log_execution_stats(result: VWAPExecutionResult) -> bool:
    """Persist VWAP execution result to PostgreSQL."""
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO paper_execution_stats
                   (ticker, side, total_requested, total_filled, expected_price,
                    filled_avg_price, slippage_bps, fill_rate, num_slices,
                    used_fallback, adv_capped, elapsed_s, slices_json, error)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (result.ticker, result.side, result.total_requested,
                 result.total_filled, result.expected_price,
                 result.filled_avg_price, result.slippage_bps, result.fill_rate,
                 len(result.slices), result.used_fallback, result.adv_capped,
                 result.elapsed_s,
                 Json([s.__dict__ for s in result.slices]),
                 result.error),
            )
        conn.commit()
        return_db_conn(conn)
        conn = None
        return True
    except Exception as e:
        logger.warning("Failed to log execution stats: %s", e)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return False
    finally:
        if conn:
            return_db_conn(conn)


def log_signal_analysis(analysis: SignalAnalysis) -> bool:
    """Persist LLM signal analysis to PostgreSQL."""
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO paper_signal_analyses
                   (regime, summary, patterns, confidence, flags_json,
                    top_signals, model_used, latency_s, raw_json)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (analysis.regime, analysis.summary, analysis.patterns,
                 analysis.confidence, Json(analysis.flags.to_dict()),
                 Json(analysis.top_signals), analysis.model_used,
                 analysis.latency_s, Json(analysis.to_dict())),
            )
        conn.commit()
        return_db_conn(conn)
        conn = None
        return True
    except Exception as e:
        logger.warning("Failed to log signal analysis: %s", e)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return False
    finally:
        if conn:
            return_db_conn(conn)


# ============================================================
# DATA FETCHING
# ============================================================
def fetch_stock_data(symbols: List[str], days: int = 300) -> pd.DataFrame:
    import yfinance as yf
    logger.info("Fetching data for %d symbols...", len(symbols))

    all_symbols = symbols + ["SPY"]  # always include SPY for regime
    data = yf.download(all_symbols, period=f"{days}d", group_by="ticker",
                       auto_adjust=True, progress=False)

    rows = []
    for sym in all_symbols:
        try:
            sym_data = data[sym].dropna() if len(all_symbols) > 1 else data.dropna()
            for dt, row in sym_data.iterrows():
                rows.append({
                    "symbol": sym, "timestamp": dt,
                    "open": float(row["Open"]), "high": float(row["High"]),
                    "low": float(row["Low"]), "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                })
        except Exception as e:
            logger.warning("Failed to get data for %s: %s", sym, e)

    df = pd.DataFrame(rows)
    logger.info("Fetched %d rows for %d symbols", len(df), df["symbol"].nunique())
    return df


def fetch_fx_data(pairs: List[str], days: int = 200) -> pd.DataFrame:
    import yfinance as yf

    yf_map = {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
        "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X", "USDCHF": "CHF=X",
    }

    yf_symbols = [yf_map.get(p, f"{p}=X") for p in pairs]
    data = yf.download(yf_symbols, period=f"{days}d", group_by="ticker",
                       auto_adjust=True, progress=False)

    rows = []
    for pair, yf_sym in zip(pairs, yf_symbols):
        try:
            sym_data = data[yf_sym].dropna()
            for dt, row in sym_data.iterrows():
                rows.append({
                    "symbol": pair, "timestamp": dt,
                    "close": float(row["Close"]), "volume": 0,
                })
        except Exception:
            pass

    return pd.DataFrame(rows)


# ============================================================
# STRATEGY RUNNER (dynamic registry)
# ============================================================
def _run_strategy(name: str, strategy, data: pd.DataFrame, today: date) -> Optional[Any]:
    """Run a single strategy with error handling."""
    try:
        output = strategy.generate_signals(data)
        df = output.to_dataframe()
        if not df.empty:
            log_signals(today, name, df)
        logger.info("%s: %d signals", name, len(output.scores))
        return output
    except Exception as e:
        logger.error("%s strategy failed: %s", name, e)
        return None


def build_strategies(config: StrategyMasterConfig, mgr: Optional[ModelManager]) -> Dict[str, Any]:
    """Build all enabled strategies from config."""
    strategies = {}

    # Always-on core strategies
    strategies["cross_sectional_momentum"] = CrossSectionalMomentum(MomentumConfig(
        enabled=True, min_history_days=250, min_avg_dollar_volume=0,
        long_threshold_zscore=1.0, short_threshold_zscore=-1.0,
        max_positions_per_side=5,
    ))
    strategies["pairs_trading"] = PairsTrading(StatArbConfig(
        enabled=True, cointegration_pvalue=0.10,
        same_sector_only=False, max_pairs=10,
    ))

    # FX strategies
    strategies["fx_carry_trend"] = FXCarryTrend(FXConfig(enabled=True))
    if config.fx_momentum.enabled:
        strategies["fx_momentum"] = FXMomentumStrategy(config.fx_momentum)
    if config.fx_vol_breakout.enabled:
        strategies["fx_vol_breakout"] = FXVolBreakoutStrategy(config.fx_vol_breakout)

    # Stock strategies (new)
    if config.mean_reversion.enabled:
        strategies["mean_reversion"] = MeanReversionStrategy(config.mean_reversion, mgr)
    if config.sector_rotation.enabled:
        strategies["sector_rotation"] = SectorRotationStrategy(config.sector_rotation, mgr)

    # Model-dependent strategies
    if config.kronos.enabled and mgr is not None:
        strategies["kronos"] = KronosStrategy(config=config.kronos, manager=mgr)
    if config.deep_surrogates.enabled and mgr is not None:
        strategies["deep_surrogates"] = DeepSurrogateStrategy(config=config.deep_surrogates, manager=mgr)
    if config.tdgf.enabled and mgr is not None:
        strategies["tdgf"] = TDGFStrategy(config=config.tdgf, manager=mgr)

    # Vol arb (integrate existing options strategy)
    vol_arb_enabled = os.getenv("STRATEGY_VOL_ARB_ENABLED", "false").lower() in ("true", "1", "yes")
    if vol_arb_enabled:
        strategies["vol_arb"] = VolatilityArbitrage(VolArbConfig.from_env())

    # Sentiment strategy (contrarian / momentum confirmation)
    if config.sentiment.enabled and mgr is not None:
        strategies["sentiment"] = SentimentStrategy(config=config.sentiment, manager=mgr)

    return strategies


# ============================================================
# MAIN PIPELINE
# ============================================================
async def run_daily_pipeline(is_manual: bool = False):
    """Execute the full ensemble pipeline."""
    if state.is_running:
        logger.warning("Pipeline already running, skipping")
        return

    state.is_running = True
    run_start = datetime.now(timezone.utc)
    today = date.today()

    try:
        logger.info("=" * 60)
        logger.info("DAILY PIPELINE START — %s", today)
        logger.info("=" * 60)

        # Audit: pipeline start
        if audit_logger is not None:
            try:
                audit_logger.log_event("pipeline_start", reason=f"Daily pipeline {today}")
            except Exception:
                pass

        # 1. Fetch data
        stock_data = fetch_stock_data(TRADING_SYMBOLS)
        fx_data = fetch_fx_data(FX_PAIRS)

        if stock_data.empty:
            logger.error("No stock data fetched, aborting")
            return

        # 2. Get current account state
        account = await broker.get_account()
        prev_value = state.portfolio_value
        state.portfolio_value = float(account.portfolio_value) if hasattr(account, 'portfolio_value') else float(account.get("portfolio_value", prev_value) if isinstance(account, dict) else prev_value)
        cash = float(account.cash) if hasattr(account, 'cash') else float(account.get("cash", 0) if isinstance(account, dict) else 0)
        equity = float(account.equity) if hasattr(account, 'equity') else float(account.get("equity", 0) if isinstance(account, dict) else 0)

        # 3. Circuit breaker check before trading (fail-closed)
        if circuit_breaker is not None:
            try:
                is_tripped = await circuit_breaker.is_tripped()
                if is_tripped:
                    state.circuit_breaker_tripped = True
                    logger.warning("CIRCUIT BREAKER TRIPPED — skipping trade execution")
                    if notification_manager is not None:
                        await notification_manager.send(AlertMessage(
                            title="Circuit Breaker Tripped",
                            body="Trading halted due to drawdown limit breach",
                            severity="critical",
                        ))
                    return
                state.circuit_breaker_tripped = False
            except Exception as e:
                logger.critical("Circuit breaker check failed — FAIL CLOSED: %s", e)
                state.circuit_breaker_tripped = True
                return
        elif state.circuit_breaker_tripped:
            logger.critical("Circuit breaker unavailable (Redis down) — FAIL CLOSED, skipping trades")
            return

        # 4. Detect regime
        regime_detector = RegimeDetector(RegimeConfig(enabled=True))
        regime_state = regime_detector.detect(stock_data, vix_value=None)
        state.last_regime = regime_state.regime.value

        logger.info("Regime: %s (exposure_scalar=%.0f%%)",
                     regime_state.regime.value, regime_state.exposure_scalar * 100)

        # 5. Build and run all enabled strategies
        config = StrategyMasterConfig.from_env()
        strategy_map = build_strategies(config, model_manager)
        state.enabled_strategies = list(strategy_map.keys())

        strategy_outputs = []
        fx_strategies = {"fx_carry_trend", "fx_momentum", "fx_vol_breakout"}

        for strat_name, strategy in strategy_map.items():
            # Use FX data for FX strategies, stock data otherwise
            input_data = fx_data if strat_name in fx_strategies and not fx_data.empty else stock_data
            if strat_name in fx_strategies and fx_data.empty:
                continue

            output = _run_strategy(strat_name, strategy, input_data, today)
            if output is not None:
                strategy_outputs.append(output)

                # Publish tail risk for deep surrogates
                if strat_name == "deep_surrogates" and signal_publisher is not None:
                    try:
                        tail_risk_data = strategy.get_all_tail_risk()
                        if tail_risk_data:
                            signal_publisher.publish_tail_risk(tail_risk_data)
                    except Exception:
                        pass

        # 6. Combine via ensemble
        combiner = EnsembleCombiner(
            config=EnsembleConfig(
                enabled=True, weighting_method="bayesian",
                max_total_positions=20, max_gross_leverage=1.5,
                use_bayesian_updater=USE_BAYESIAN_WEIGHTS,
            ),
            bayesian_updater=bayesian_updater,
        )
        combined = combiner.combine(strategy_outputs, regime_state)

        # 6a. GUARDRAIL: Signal variance check (March 10 incident prevention)
        if combined:
            scores = [s.score for s in combined if hasattr(s, 'score')]
            variance_result = signal_guard.check(scores)
            if not variance_result.passed:
                logger.critical("Signal variance guardrail tripped — aborting execution")
                if notification_manager is not None:
                    await notification_manager.send(AlertMessage(
                        title="GUARDRAIL: Signal Variance Collapse",
                        body=variance_result.message,
                        severity="critical",
                    ))
                else:
                    await _send_discord_fallback(
                        f"**GUARDRAIL HALT**\n{variance_result.message}"
                    )
                return

        # 6b. Portfolio risk assessment (HI-1 fix: use persistent risk_manager)
        risk_assessment = None
        try:
            if risk_manager is not None:
                # Feed strategy returns for correlation monitoring
                for output in strategy_outputs:
                    strat_return = sum(s.score for s in output.scores) / max(len(output.scores), 1)
                    risk_manager.record_strategy_return(output.strategy_name, strat_return)

                risk_assessment = risk_manager.assess()
                log_risk_report(risk_assessment)

                # Kill switch: halt ALL trades
                if risk_assessment.kill_switch_triggered:
                    logger.critical("Portfolio risk kill switch triggered: %s", risk_assessment.kill_reason)
                    if notification_manager is not None:
                        await notification_manager.send(AlertMessage(
                            title="RISK KILL SWITCH — All Trading Halted",
                            body=f"Portfolio risk limit breached: {risk_assessment.kill_reason}",
                            severity="critical",
                        ))
                    else:
                        await _send_discord_fallback(
                            f"**RISK KILL SWITCH**\n{risk_assessment.kill_reason}"
                        )
                    return

                # Killed strategies: filter from combined signals
                killed = risk_manager.get_killed_strategies()
                if killed:
                    killed_names = set(killed.keys())
                    logger.warning("Killed strategies excluded: %s", killed_names)
                    combined = [
                        s for s in combined
                        if not any(k in s.contributing_strategies for k in killed_names)
                    ] if combined else combined

                # Correlation-based weight reduction
                if risk_assessment.correlation_alerts and combined:
                    for alert in risk_assessment.correlation_alerts:
                        if alert.correlation > 0.85:
                            # Reduce the younger (later-listed) strategy's signals by 50%
                            younger = alert.strategy_b
                            logger.warning(
                                "Correlation %.3f between %s and %s — reducing %s weight by 50%%",
                                alert.correlation, alert.strategy_a, alert.strategy_b, younger,
                            )
                            for sig in combined:
                                if hasattr(sig, 'contributing_strategies') and younger in sig.contributing_strategies:
                                    sig.combined_score *= 0.5
        except Exception as e:
            logger.warning("Risk assessment failed (continuing): %s", e)

        # Publish combined signals to Redis (fire-and-forget)
        if signal_publisher is not None and combined:
            try:
                signal_publisher.publish_signals(combined)
            except Exception as e:
                logger.warning("Redis signal publish failed (non-critical): %s", e)

        # Track weights
        weight_hist = combiner.get_weight_history(1)
        if weight_hist:
            state.last_weights = {
                name: round(w.final_weight, 3) for name, w in weight_hist[-1].items()
            }

        # 6c. Cache signals for signal provider API
        signal_dicts = [
            {"symbol": s.symbol, "combined_score": s.combined_score,
             "confidence": s.confidence, "direction": s.direction.value,
             "contributing_strategies": s.contributing_strategies}
            for s in combined
        ]
        try:
            bay_w = bayesian_updater.get_weights() if bayesian_updater else None
            bay_s = bayesian_updater.get_state_dicts() if bayesian_updater else None
            regime_detail = {
                "regime": regime_state.regime.value,
                "vix_level": regime_state.vix_level,
                "is_volatile": regime_state.is_volatile,
                "is_trending": regime_state.is_trending,
                "confidence": regime_state.confidence,
                "exposure_scalar": regime_state.exposure_scalar,
            } if regime_state else {}
            signal_cache.refresh(
                signals=signal_dicts,
                weights=state.last_weights,
                regime=state.last_regime,
                regime_detail=regime_detail,
                bayesian_weights=bay_w,
                bayesian_state=bay_s,
            )
        except Exception as e:
            logger.warning("Signal cache refresh failed: %s", e)

        # 6d. LLM signal analysis (after ensemble, before optimization)
        llm_analysis = None
        if signal_analyst is not None:
            try:
                risk_dict = risk_assessment.to_dict() if risk_assessment else {}
                llm_analysis = await signal_analyst.analyze(
                    signals=signal_dicts,
                    weights=state.last_weights,
                    regime=state.last_regime or "unknown",
                    risk_summary=risk_dict,
                    is_manual=is_manual,
                )
                if llm_analysis is not None:
                    log_signal_analysis(llm_analysis)
                    logger.info("LLM analysis: %s (confidence=%s)",
                                llm_analysis.summary[:100], llm_analysis.confidence)
            except Exception as e:
                logger.warning("LLM signal analysis failed (non-critical): %s", e)

        # 7. Optimize portfolio
        optimizer = PortfolioOptimizer(EnsembleConfig(
            enabled=True, max_total_positions=15,
            max_gross_leverage=1.2, target_volatility=0.15,
        ))
        target = optimizer.optimize(combined, stock_data, regime_state)

        logger.info("Target portfolio: %d positions, gross=%.2f, net=%.2f",
                     target.position_count, target.gross_leverage, target.net_leverage)

        # 7b. GUARDRAIL: Leverage gate before execution
        lev_result = leverage_gate.check(target.gross_leverage)
        if not lev_result.passed:
            logger.warning("Leverage guardrail tripped — skipping new orders")
            if notification_manager is not None:
                await notification_manager.send(AlertMessage(
                    title="GUARDRAIL: Leverage Limit Exceeded",
                    body=lev_result.message,
                    severity="warning",
                ))
            else:
                await _send_discord_fallback(
                    f"**GUARDRAIL WARNING**\n{lev_result.message}"
                )
            # Skip execution but continue to snapshot & reporting
            trades_executed = 0
            target.positions = []
            # Jump past the execution block
        # 8. Execute trades (skip if leverage gate tripped)
        if not lev_result.passed:
            trades_executed = 0
        else:
            current_positions = await broker.get_positions()
            if isinstance(current_positions, list) and len(current_positions) > 0:
                if isinstance(current_positions[0], dict):
                    current_holdings = {p["symbol"]: float(p["qty"]) for p in current_positions}
                else:
                    current_holdings = {p.ticker: float(p.quantity) for p in current_positions}
            else:
                current_holdings = {}

            trades_executed = 0
            for pos in target.positions:
                # GUARDRAIL: Execution failure rate check before each order
                exec_health = exec_monitor.check()
                if not exec_health.passed:
                    logger.critical("Execution failure rate guardrail tripped — pausing orders")
                    if notification_manager is not None:
                        await notification_manager.send(AlertMessage(
                            title="GUARDRAIL: Execution Failure Rate",
                            body=exec_health.message,
                            severity="critical",
                        ))
                    else:
                        await _send_discord_fallback(
                            f"**GUARDRAIL HALT**\n{exec_health.message}"
                        )
                    break

                symbol = pos.symbol
                target_value = pos.target_weight * state.portfolio_value
                current_qty = current_holdings.get(symbol, 0)

                # Get latest price
                sym_data = stock_data[stock_data["symbol"] == symbol]
                if sym_data.empty:
                    continue
                price = float(sym_data.sort_values("timestamp")["close"].iloc[-1])
                if price <= 0:
                    continue

                target_shares = int(target_value / price)
                diff = target_shares - current_qty

                if abs(diff) < 1:
                    continue

                side = "buy" if diff > 0 else "sell"
                qty = abs(diff)

                # Circuit breaker re-check before each trade
                if circuit_breaker is not None:
                    try:
                        if await circuit_breaker.is_tripped():
                            logger.warning("Circuit breaker tripped during execution, stopping")
                            break
                    except Exception:
                        pass

                order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

                # VWAP execution or direct market order
                if vwap_model is not None:
                    # Compute average daily volume for ADV cap
                    adv = 0.0
                    if "volume" in sym_data.columns:
                        vol_series = sym_data["volume"].dropna()
                        if len(vol_series) >= 20:
                            adv = float(vol_series.tail(20).mean())
                        elif len(vol_series) > 0:
                            adv = float(vol_series.mean())

                    vwap_result = await vwap_model.execute(
                        ticker=symbol, side=order_side, quantity=qty,
                        current_price=price, adv=adv,
                    )
                    log_execution_stats(vwap_result)
                    if prom_metrics is not None and vwap_result.total_filled > 0:
                        prom_metrics.observe_slippage(symbol, vwap_result.slippage_bps)

                    # Synthesize OrderResult for downstream tracking
                    result = OrderResult(
                        success=(vwap_result.total_filled > 0),
                        order_id=vwap_result.slices[0].order_id if vwap_result.slices else None,
                        status=OrderStatus.FILLED if vwap_result.total_filled > 0 else OrderStatus.CANCELLED,
                        message=f"VWAP: filled {vwap_result.total_filled}/{vwap_result.total_requested}",
                    )
                    qty = vwap_result.total_filled or qty
                    if vwap_result.filled_avg_price > 0:
                        price = vwap_result.filled_avg_price
                else:
                    order_req = OrderRequest(
                        ticker=symbol, side=order_side, quantity=qty,
                    )
                    result = await broker.submit_order(order_req)

                # GUARDRAIL: Record execution outcome for failure rate tracking
                exec_monitor.record(bool(result and result.success))

                if result and result.success:
                    trades_executed += 1
                    order_id = result.order_id or ""
                    log_trade(today, symbol, side, qty, price, order_id,
                              "ensemble", pos.combined_score, pos.confidence,
                              {"target_weight": pos.target_weight})
                    logger.info("  %s %d %s @ $%.2f (order=%s)",
                                side.upper(), qty, symbol, price, str(order_id)[:8])

                    # Audit trail
                    if audit_logger is not None:
                        try:
                            audit_logger.log_event(
                                "trade_executed",
                                reason=f"{side} {qty} {symbol} @ ${price:.2f}",
                                metadata={"order_id": order_id, "strategy": "ensemble"},
                            )
                        except Exception:
                            pass

        # 9. Snapshot
        account = await broker.get_account()
        state.portfolio_value = float(account.portfolio_value) if hasattr(account, 'portfolio_value') else float(account.get("portfolio_value", state.portfolio_value) if isinstance(account, dict) else state.portfolio_value)
        daily_pnl = state.portfolio_value - prev_value
        daily_return = daily_pnl / prev_value if prev_value > 0 else 0
        state.last_pnl = daily_pnl
        state.total_return_pct = (state.portfolio_value / INITIAL_CAPITAL - 1) * 100
        state.daily_returns.append(daily_return)
        state.day_count += 1

        # Feed today's return to persistent risk manager
        if risk_manager is not None:
            risk_manager.record_portfolio_return(daily_return)

        # Update Bayesian weights based on per-strategy profitability
        if bayesian_updater is not None and strategy_outputs:
            outcomes = {}
            for output in strategy_outputs:
                strat_return = sum(s.score * (1 if daily_return > 0 else -1)
                                  for s in output.scores) / max(len(output.scores), 1)
                outcomes[output.strategy_name] = (strat_return > 0)
            bayesian_updater.update(outcomes)
            save_bayesian_state(bayesian_updater)
            logger.info("Bayesian weights updated: %s",
                        {k: f"{v:.3f}" for k, v in bayesian_updater.get_weights().items()})

        positions = await broker.get_positions()
        if isinstance(positions, list) and len(positions) > 0:
            if isinstance(positions[0], dict):
                state.last_positions = [
                    {
                        "symbol": p["symbol"],
                        "qty": float(p["qty"]),
                        "market_value": float(p["market_value"]),
                        "unrealized_pl": float(p["unrealized_pl"]),
                        "unrealized_plpc": float(p.get("unrealized_plpc", 0)),
                    }
                    for p in positions
                ]
            else:
                state.last_positions = [
                    {
                        "symbol": p.ticker,
                        "qty": float(p.quantity),
                        "market_value": float(p.market_value),
                        "unrealized_pl": float(p.unrealized_pnl),
                        "unrealized_plpc": float(p.unrealized_pnl_percent),
                    }
                    for p in positions
                ]
        else:
            state.last_positions = []

        acct_cash = float(account.cash) if hasattr(account, 'cash') else float(account.get("cash", 0) if isinstance(account, dict) else 0)
        acct_equity = float(account.equity) if hasattr(account, 'equity') else float(account.get("equity", 0) if isinstance(account, dict) else 0)

        snapshot = {
            "date": today,
            "portfolio_value": state.portfolio_value,
            "cash": acct_cash,
            "equity": acct_equity,
            "daily_pnl": daily_pnl,
            "daily_return_pct": daily_return * 100,
            "total_return_pct": state.total_return_pct,
            "positions_count": len(state.last_positions),
            "regime": state.last_regime,
            "strategy_weights": state.last_weights,
            "positions": state.last_positions,
            "risk_metrics": {
                "gross_leverage": target.gross_leverage,
                "net_leverage": target.net_leverage,
                "expected_vol": target.expected_volatility,
                "var_99": target.var_99,
            },
        }
        log_daily_snapshot(snapshot)

        state.run_log.append({
            "date": str(today),
            "pnl": round(daily_pnl, 2),
            "return_pct": round(daily_return * 100, 2),
            "trades": trades_executed,
            "regime": state.last_regime,
            "positions": len(state.last_positions),
            "strategies": len(strategy_outputs),
        })

        # 10. Send report via NotificationManager
        sharpe = 0
        if len(state.daily_returns) >= 5:
            rets = np.array(state.daily_returns)
            sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

        strat_summary = " | ".join(
            f"{o.strategy_name}: {len(o.scores)}"
            for o in strategy_outputs if o.scores
        ) or "none"

        # VWAP execution stats for report
        vwap_line = ""
        if vwap_model is not None:
            vs = vwap_model.get_execution_stats()
            if vs["total_executions"] > 0:
                vwap_line = (
                    f"\nVWAP: {vs['total_executions']} orders, "
                    f"avg slip {vs['avg_slippage_bps']:.1f}bps, "
                    f"fill rate {vs['avg_fill_rate']:.1%}, "
                    f"{vs['fallback_count']} fallbacks"
                )

        # LLM analysis for report
        llm_line = ""
        if llm_analysis is not None:
            llm_line = f"\n{llm_analysis.to_report_line()}"

        report = (
            f"**Day {state.day_count}/30**\n"
            f"Portfolio: ${state.portfolio_value:,.2f}\n"
            f"Daily P&L: ${daily_pnl:+,.2f} ({daily_return:+.2%})\n"
            f"Total Return: {state.total_return_pct:+.2f}%\n"
            f"Sharpe (est): {sharpe:.2f}\n"
            f"Regime: {state.last_regime}\n"
            f"Positions: {len(state.last_positions)} | Trades: {trades_executed}\n"
            f"Gross Lev: {target.gross_leverage:.2f} | Net Lev: {target.net_leverage:.2f}\n"
            f"Strategies: {len(strategy_outputs)}/{len(strategy_map)} active ({strat_summary})\n"
            f"Weights: {state.last_weights}"
            f"{vwap_line}"
            f"{llm_line}"
        )

        if notification_manager is not None:
            severity = "info" if daily_pnl >= 0 else "warning"
            await notification_manager.send(AlertMessage(
                title="APEX Paper Trading — Daily Report",
                body=report,
                severity=severity,
            ))
        else:
            await _send_discord_fallback(report)

        # Audit: pipeline complete
        if audit_logger is not None:
            try:
                audit_logger.log_event(
                    "pipeline_complete",
                    reason=f"Day {state.day_count}: P&L ${daily_pnl:+,.2f}",
                    metadata={"trades": trades_executed, "strategies": len(strategy_outputs)},
                )
            except Exception:
                pass

        state.last_run = datetime.now(timezone.utc)
        elapsed = (state.last_run - run_start).total_seconds()
        logger.info("Pipeline complete in %.1fs — P&L: $%+.2f (%+.2f%%)",
                     elapsed, daily_pnl, daily_return * 100)

        # 12. Update Prometheus metrics
        if prom_metrics is not None:
            try:
                prom_metrics.update_signals(signal_dicts)
                prom_metrics.update_weights(
                    state.last_weights,
                    bayesian_weights=bayesian_updater.get_weights() if bayesian_updater else None,
                )
                prom_metrics.update_regime(
                    regime=state.last_regime or "unknown",
                    is_volatile=regime_state.is_volatile if regime_state else False,
                    is_trending=regime_state.is_trending if regime_state else False,
                )
                if risk_assessment is not None:
                    prom_metrics.update_risk(
                        drawdown=risk_assessment.portfolio_drawdown,
                        var_99=risk_assessment.var.parametric_var if risk_assessment.var else 0,
                        cvar_95=risk_assessment.var.cvar_95 if risk_assessment.var else 0,
                    )
                prom_metrics.observe_pipeline_duration(elapsed)
            except Exception as e_metrics:
                logger.warning("Prometheus metrics update failed: %s", e_metrics)

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        error_msg = f"**PIPELINE ERROR**\n{str(e)[:500]}"
        if notification_manager is not None:
            await notification_manager.send(AlertMessage(
                title="APEX Pipeline Error",
                body=error_msg,
                severity="critical",
            ))
        else:
            await _send_discord_fallback(error_msg)
    finally:
        state.is_running = False


async def _send_discord_fallback(report: str):
    """Fallback Discord sender when NotificationManager is not available."""
    if not DISCORD_WEBHOOK_URL or "YOUR_WEBHOOK" in DISCORD_WEBHOOK_URL:
        logger.info("Discord not configured, skipping notification")
        return

    embed = {
        "title": "APEX Paper Trading — Daily Report",
        "description": report,
        "color": 0x4CAF50 if state.last_pnl >= 0 else 0xF44336,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": f"Day {state.day_count}/30 | Paper Trading"},
    }
    payload = {"embeds": [embed]}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(DISCORD_WEBHOOK_URL, json=payload,
                                     timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status in (200, 204):
                    logger.info("Discord report sent")
                else:
                    logger.error("Discord failed: %s", resp.status)
    except Exception as e:
        logger.error("Discord send error: %s", e)


# ============================================================
# FASTAPI APPLICATION
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, signal_publisher, notification_manager, audit_logger
    global circuit_breaker, db_pool, broker, risk_manager, bayesian_updater, vwap_model
    global signal_analyst

    # Validate required environment variables before any connections
    from utils.env_validator import validate
    validate(strict=True)

    # Database connection pool
    try:
        db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2, maxconn=10,
            host=DB_HOST, port=DB_PORT, database=DB_NAME,
            user=DB_USER, password=DB_PASSWORD,
        )
        logger.info("Database connection pool created (2-10 connections)")
    except Exception as e:
        logger.warning("Connection pool failed, falling back to per-query connections: %s", e)
        db_pool = None

    # Startup
    init_db()

    # Production broker
    broker = AlpacaBroker(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        base_url=ALPACA_BASE_URL,
    )
    await broker.connect()

    # Load models (graceful — strategies fall back if models unavailable)
    model_manager = ModelManager()
    model_manager.load_all()

    # GUARDRAIL: Calibration health check at startup
    # Check any loaded calibration models for identity/unfitted state
    for name, model in (model_manager.models if hasattr(model_manager, 'models') else {}).items():
        if hasattr(model, 'calibrator_a') and hasattr(model, 'calibrator_b'):
            cal_result = CalibrationHealthCheck.check_platt(model.calibrator_a, model.calibrator_b)
            if not cal_result.passed:
                logger.error("Startup calibration check failed for '%s': %s", name, cal_result.message)
        elif hasattr(model, 'calibrator'):
            cal_result = CalibrationHealthCheck.check_generic(model.calibrator)
            if not cal_result.passed:
                logger.error("Startup calibration check failed for '%s': %s", name, cal_result.message)

    logger.info("Safety guardrails initialized: signal_variance(min_std=%.4f), "
                "leverage_gate(max=%.2f), exec_monitor(max_fail_rate=%.0f%%)",
                signal_guard.min_std, leverage_gate.max_leverage,
                exec_monitor.max_failure_rate * 100)

    # Notification manager (Discord + Email)
    try:
        senders = []
        if DISCORD_WEBHOOK_URL and "YOUR_WEBHOOK" not in DISCORD_WEBHOOK_URL:
            senders.append(DiscordWebhookSender(DISCORD_WEBHOOK_URL))
        email_user = os.getenv("EMAIL_USER", "")
        email_pass = os.getenv("EMAIL_PASSWORD", "")
        email_to = os.getenv("EMAIL_TO", "")
        if email_user and email_pass and email_to:
            senders.append(EmailSender(email_user, email_pass, email_to))
        if senders:
            notification_manager = NotificationManager(senders)
            logger.info("NotificationManager initialized with %d senders", len(senders))
    except Exception as e:
        logger.info("NotificationManager init failed (using fallback): %s", e)
        notification_manager = None

    # Audit logger
    try:
        audit_logger = AuditLogger(db_config={
            "host": DB_HOST, "port": int(DB_PORT), "database": DB_NAME,
            "user": DB_USER, "password": DB_PASSWORD,
        })
        try:
            audit_logger.create_schema()
        except Exception:
            pass  # tables may already exist with different schema
        logger.info("AuditLogger initialized")
    except Exception as e:
        logger.info("AuditLogger not available: %s", e)
        audit_logger = None

    # Portfolio risk manager (HI-1 fix: persistent instance with historical data)
    risk_manager = PortfolioRiskManager(
        max_portfolio_drawdown=float(os.getenv("RISK_MAX_DRAWDOWN", "0.20")),
        var_confidence=0.99,
        correlation_alert_threshold=float(os.getenv("RISK_CORRELATION_THRESHOLD", "0.85")),
    )
    historical_returns = load_historical_returns(days=30)
    for ret in historical_returns:
        risk_manager.record_portfolio_return(ret)
    logger.info("PortfolioRiskManager initialized with %d historical returns", len(historical_returns))

    # Bayesian weight updater (adaptive ensemble weights)
    if USE_BAYESIAN_WEIGHTS:
        bayesian_updater = BayesianWeightUpdater(
            decay_factor=float(os.getenv("ENSEMBLE_BAYESIAN_DECAY", "0.995")),
        )
        n_loaded = load_bayesian_state(bayesian_updater)
        logger.info("BayesianWeightUpdater initialized (loaded %d strategies from DB)", n_loaded)
    else:
        logger.info("BayesianWeightUpdater disabled (set ENSEMBLE_USE_BAYESIAN_WEIGHTS=true to enable)")

    # VWAP execution model (wraps AlpacaBroker for time-sliced execution)
    if USE_VWAP_EXECUTION:
        vwap_model = VWAPExecutionModel(
            broker=broker,
            num_slices=int(os.getenv("VWAP_NUM_SLICES", "5")),
            slice_interval_s=int(os.getenv("VWAP_SLICE_INTERVAL_S", "60")),
            adv_cap_pct=float(os.getenv("VWAP_ADV_CAP_PCT", "0.10")),
        )
        logger.info("VWAPExecutionModel initialized (%d slices, %ds interval)",
                     vwap_model.num_slices, vwap_model.slice_interval_s)
    else:
        logger.info("VWAP execution disabled (set EXECUTION_USE_VWAP=true to enable)")

    # LLM signal analyst
    if LLM_ANALYST_ENABLED:
        signal_analyst = SignalAnalyst()
        logger.info("SignalAnalyst initialized (model=%s)", signal_analyst.client.model)
    else:
        logger.info("LLM signal analyst disabled (set LLM_ANALYST_ENABLED=true to enable)")

    # Circuit breaker
    try:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()

        cb_config = CircuitBreakerConfig(
            drawdown_configs=[
                DrawdownConfig(method=DrawdownMethod.HIGH_WATER_MARK, threshold_percent=5.0),
                DrawdownConfig(method=DrawdownMethod.START_OF_DAY, threshold_percent=3.0),
            ],
            initial_capital=INITIAL_CAPITAL,
        )
        circuit_breaker = CircuitBreaker(
            config=cb_config,
            broker=broker,
            redis_client=redis_client,
            notifier=notification_manager,
            audit=audit_logger,
        )
        await circuit_breaker.start()
        logger.info("CircuitBreaker initialized and started")
    except Exception as e:
        logger.critical("CircuitBreaker not available (Redis required): %s — FAIL CLOSED, trading disabled", e)
        circuit_breaker = None
        # Fail-closed: trip the circuit breaker state so no trades execute
        state.circuit_breaker_tripped = True

    # Redis signal publisher (optional — fire-and-forget)
    try:
        import redis
        redis_sync = redis.from_url(REDIS_URL, decode_responses=True)
        redis_sync.ping()
        signal_publisher = SignalPublisher(redis_sync)
        logger.info("Redis signal publisher connected")
    except Exception as e:
        logger.info("Redis not available for signal publishing (optional): %s", e)
        signal_publisher = None

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_daily_pipeline,
        "cron",
        hour=SCHEDULE_HOUR,
        minute=SCHEDULE_MINUTE,
        day_of_week="mon-fri",
        timezone="US/Eastern",
    )
    scheduler.start()
    state.scheduler = scheduler
    logger.info("Scheduler started: daily at %d:%02d ET (Mon-Fri)", SCHEDULE_HOUR, SCHEDULE_MINUTE)

    # Register signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler(sig):
        logger.info("Received %s — initiating graceful shutdown", signal.Signals(sig).name)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler, sig)

    yield

    # Shutdown with timeout
    SHUTDOWN_TIMEOUT = 30

    async def _shutdown_sequence():
        logger.info("Starting graceful shutdown (timeout=%ds)...", SHUTDOWN_TIMEOUT)
        scheduler.shutdown(wait=False)
        if circuit_breaker is not None:
            await circuit_breaker.stop()
        await broker.disconnect()
        if db_pool is not None:
            db_pool.closeall()
        logger.info("Shutdown complete")

    try:
        await asyncio.wait_for(_shutdown_sequence(), timeout=SHUTDOWN_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error("Shutdown timed out after %ds — forcing exit", SHUTDOWN_TIMEOUT)
    except Exception as e:
        logger.error("Error during shutdown: %s", e)

app = FastAPI(
    title="APEX Paper Trader",
    description="Multi-strategy paper trading runner with 10 models, 11 strategies, and live dashboard",
    version="2.0.0",
    lifespan=lifespan,
)

# Mount signal provider API
if SIGNAL_API_ENABLED and SIGNAL_API_KEY:
    def _db_query(query: str, params: tuple) -> list:
        """Execute a read-only DB query for the signal API."""
        conn = None
        try:
            conn = get_db_conn()
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            return_db_conn(conn)
            conn = None
            return rows
        except Exception:
            if conn:
                return_db_conn(conn)
            raise

    signal_api = create_signal_api(
        api_key=SIGNAL_API_KEY,
        cache=signal_cache,
        db_query_fn=_db_query,
    )
    app.mount("/api/v1", signal_api)
    logger.info("Signal provider API mounted at /api/v1/")
else:
    if SIGNAL_API_ENABLED and not SIGNAL_API_KEY:
        logger.warning("SIGNAL_API_ENABLED=true but SIGNAL_API_KEY is empty — API not mounted")

# Mount Prometheus /metrics endpoint
if prom_metrics is not None:
    app.mount("/metrics", prom_metrics.get_asgi_app())
    logger.info("Prometheus metrics endpoint mounted at /metrics")


@app.get("/health")
async def health():
    result = {
        "status": "running",
        "version": "2.0.0",
        "day_count": state.day_count,
        "last_run": str(state.last_run) if state.last_run else None,
        "portfolio_value": state.portfolio_value,
        "is_running": state.is_running,
        "regime": state.last_regime,
        "strategy_weights": state.last_weights,
        "enabled_strategies": state.enabled_strategies,
        "circuit_breaker_tripped": state.circuit_breaker_tripped,
        "infrastructure": {
            "broker": "AlpacaBroker" if broker is not None else "None",
            "notification_manager": notification_manager is not None,
            "audit_logger": audit_logger is not None,
            "circuit_breaker": circuit_breaker is not None,
            "db_pool": db_pool is not None,
        },
    }
    if model_manager is not None:
        mgr_status = model_manager.get_status()
        result["models"] = {
            "registered": mgr_status.models_registered,
            "loaded": mgr_status.models_loaded,
            "details": {
                d.name: {"loaded": d.is_loaded, "asset_class": d.asset_class}
                for d in mgr_status.details
            },
        }
    if signal_publisher is not None:
        result["redis"] = signal_publisher.stats
    return result


@app.post("/run-now")
async def run_now():
    if state.is_running:
        return {"status": "already_running"}
    asyncio.create_task(run_daily_pipeline(is_manual=True))
    return {"status": "started"}


@app.get("/positions")
async def positions():
    return state.last_positions


@app.get("/history")
async def history():
    return state.run_log[-30:]


@app.get("/weights")
async def weights():
    return state.last_weights


@app.get("/weights/bayesian")
async def weights_bayesian():
    """Return current Bayesian weight updater state: alpha, beta, weight per strategy."""
    if bayesian_updater is None:
        return {"enabled": False, "message": "Set ENSEMBLE_USE_BAYESIAN_WEIGHTS=true to enable"}
    return {
        "enabled": True,
        "strategies": bayesian_updater.get_state_dicts(),
        "normalized_weights": bayesian_updater.get_weights(),
    }


@app.get("/execution/stats")
async def execution_stats():
    """Return VWAP execution statistics and recent history."""
    if vwap_model is None:
        return {"enabled": False, "message": "Set EXECUTION_USE_VWAP=true to enable"}
    stats = vwap_model.get_execution_stats()
    recent = [r.to_dict() for r in vwap_model.get_execution_history(20)]
    return {"enabled": True, "stats": stats, "recent": recent}


@app.get("/analysis/latest")
async def analysis_latest():
    """Return the most recent LLM signal analysis."""
    if signal_analyst is None:
        return {"enabled": False, "message": "Set LLM_ANALYST_ENABLED=true to enable"}
    analysis = signal_analyst.last_analysis
    if analysis is None:
        return {"enabled": True, "analysis": None, "message": "No analysis yet — waiting for first pipeline run"}
    return {"enabled": True, "analysis": analysis.to_dict()}


@app.get("/dlq")
async def dlq_dashboard():
    """Dead letter queue dashboard: totals by status, recent failures, retry success rate."""
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        from services.common.dlq import DeadLetterQueue
        dlq = DeadLetterQueue(db_url=db_url, service_name="paper-trader")
        return dlq.get_all_stats()
    except Exception as e:
        return {"error": str(e), "by_status": {}, "recent_failures": [], "retry_success_rate_pct": 0}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Live dashboard showing positions, P&L, strategy weights, and all 11 strategies."""
    positions_html = ""
    total_unrealized = 0
    for p in state.last_positions:
        pnl = p.get("unrealized_pl", 0)
        total_unrealized += pnl
        color = "#4CAF50" if pnl >= 0 else "#F44336"
        positions_html += f"""
        <tr>
            <td><strong>{p['symbol']}</strong></td>
            <td>{p['qty']:.0f}</td>
            <td>${p['market_value']:,.2f}</td>
            <td style="color:{color};">${pnl:+,.2f}</td>
            <td style="color:{color};">{p.get('unrealized_plpc',0)*100:+.2f}%</td>
        </tr>"""

    weights_html = ""
    for name, w in state.last_weights.items():
        bar_width = int(w * 400)
        weights_html += f"""
        <div style="margin:4px 0;">
            <span style="display:inline-block;width:220px;">{name}</span>
            <div style="display:inline-block;width:{bar_width}px;height:18px;background:#2196F3;border-radius:3px;"></div>
            <span style="margin-left:8px;">{w:.1%}</span>
        </div>"""

    history_html = ""
    for entry in reversed(state.run_log[-15:]):
        color = "#4CAF50" if entry["pnl"] >= 0 else "#F44336"
        history_html += f"""
        <tr>
            <td>{entry['date']}</td>
            <td style="color:{color};">${entry['pnl']:+,.2f}</td>
            <td style="color:{color};">{entry['return_pct']:+.2f}%</td>
            <td>{entry['trades']}</td>
            <td>{entry['regime']}</td>
            <td>{entry['positions']}</td>
            <td>{entry.get('strategies', 'N/A')}</td>
        </tr>"""

    sharpe = 0
    if len(state.daily_returns) >= 5:
        rets = np.array(state.daily_returns)
        sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

    # Query latest signals from DB for strategy panels
    panel_data = {}
    ds_tail_risk_gauge = 0.0
    conn = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            for strat_name in ("kronos", "deep_surrogates", "tdgf", "mean_reversion",
                               "sector_rotation", "fx_momentum", "fx_vol_breakout", "vol_arb"):
                cur.execute(
                    """SELECT symbol, score, confidence, direction, metadata
                       FROM paper_strategy_signals
                       WHERE strategy_name = %s
                         AND signal_date = (
                             SELECT MAX(signal_date) FROM paper_strategy_signals
                             WHERE strategy_name = %s
                         )
                       ORDER BY ABS(score) DESC
                       LIMIT 10""",
                    (strat_name, strat_name),
                )
                rows = cur.fetchall()
                panel_rows = ""
                for row in rows:
                    sym, sc, conf, dirn, meta = row
                    sc = sc or 0
                    conf = conf or 0
                    color = "#4CAF50" if sc >= 0 else "#F44336"

                    extra = ""
                    if strat_name == "tdgf" and isinstance(meta, dict):
                        mp = meta.get("mispricing_pct") or meta.get("mispricing")
                        if mp is not None:
                            extra = f" ({float(mp):+.1%})"

                    if strat_name == "deep_surrogates" and isinstance(meta, dict):
                        tr = meta.get("tail_risk_index")
                        if tr is not None:
                            ds_tail_risk_gauge = max(ds_tail_risk_gauge, float(tr))

                    panel_rows += f"""
                    <tr>
                        <td><strong>{sym}</strong></td>
                        <td style="color:{color};">{sc:+.3f}{extra}</td>
                        <td>{conf:.1%}</td>
                        <td>{dirn or 'neutral'}</td>
                    </tr>"""
                panel_data[strat_name] = panel_rows
    except Exception:
        pass
    finally:
        if conn:
            return_db_conn(conn)

    # Tail risk gauge color
    if ds_tail_risk_gauge >= 0.7:
        tr_color = "#F44336"
        tr_label = "HIGH"
    elif ds_tail_risk_gauge >= 0.4:
        tr_color = "#F59E0B"
        tr_label = "ELEVATED"
    else:
        tr_color = "#4CAF50"
        tr_label = "LOW"

    pnl_color = "#4CAF50" if state.last_pnl >= 0 else "#F44336"
    total_color = "#4CAF50" if state.total_return_pct >= 0 else "#F44336"

    cb_status = "TRIPPED" if state.circuit_breaker_tripped else ("Active" if circuit_breaker else "N/A")
    cb_color = "#F44336" if state.circuit_breaker_tripped else "#4CAF50"

    def _panel(title, subtitle, strat_name):
        rows = panel_data.get(strat_name, "")
        empty = f'<tr><td colspan="4" style="color:#8B949E;">No signals</td></tr>'
        return f"""
        <div class="card section">
            <div class="section-title">{title}</div>
            <div style="color:#8B949E; font-size:11px; margin-bottom:8px;">{subtitle}</div>
            <table>
                <tr><th>Symbol</th><th>Score</th><th>Conf</th><th>Dir</th></tr>
                {rows if rows else empty}
            </table>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>APEX Paper Trader v2</title>
    <meta http-equiv="refresh" content="60">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #0D1117; color: #C9D1D9; margin: 0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #58A6FF; margin-bottom: 5px; }}
        .subtitle {{ color: #8B949E; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }}
        .card {{ background: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 16px; }}
        .card-title {{ color: #8B949E; font-size: 12px; text-transform: uppercase; }}
        .card-value {{ font-size: 28px; font-weight: bold; margin-top: 5px; }}
        .green {{ color: #4CAF50; }}
        .red {{ color: #F44336; }}
        .blue {{ color: #58A6FF; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ text-align: left; padding: 8px 12px; color: #8B949E; border-bottom: 1px solid #30363D;
              font-size: 12px; text-transform: uppercase; }}
        td {{ padding: 8px 12px; border-bottom: 1px solid #21262D; }}
        .section {{ margin-top: 25px; }}
        .section-title {{ color: #58A6FF; font-size: 18px; margin-bottom: 10px; }}
        .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
                  font-size: 12px; font-weight: bold; }}
        .badge-regime {{ background: #1F2937; color: #F59E0B; }}
        .refresh {{ color: #8B949E; font-size: 12px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>APEX Paper Trader v2</h1>
    <div class="subtitle">
        Day {state.day_count}/30 |
        Regime: <span class="badge badge-regime">{state.last_regime or 'N/A'}</span> |
        Strategies: {len(state.enabled_strategies)}/11 |
        Circuit Breaker: <span style="color:{cb_color};">{cb_status}</span> |
        <span class="refresh">Auto-refreshes every 60s</span>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-title">Portfolio Value</div>
            <div class="card-value blue">${state.portfolio_value:,.2f}</div>
        </div>
        <div class="card">
            <div class="card-title">Today's P&L</div>
            <div class="card-value" style="color:{pnl_color};">${state.last_pnl:+,.2f}</div>
        </div>
        <div class="card">
            <div class="card-title">Total Return</div>
            <div class="card-value" style="color:{total_color};">{state.total_return_pct:+.2f}%</div>
        </div>
        <div class="card">
            <div class="card-title">Est. Sharpe</div>
            <div class="card-value blue">{sharpe:.2f}</div>
        </div>
    </div>

    <div class="grid" style="grid-template-columns: 1fr 1fr;">
        <div class="card section">
            <div class="section-title">Live Positions ({len(state.last_positions)})</div>
            <table>
                <tr><th>Symbol</th><th>Qty</th><th>Value</th><th>P&L</th><th>%</th></tr>
                {positions_html}
                <tr style="border-top:2px solid #30363D;">
                    <td><strong>TOTAL</strong></td><td></td><td></td>
                    <td style="color:{'#4CAF50' if total_unrealized>=0 else '#F44336'};">
                        <strong>${total_unrealized:+,.2f}</strong></td><td></td>
                </tr>
            </table>
        </div>
        <div class="card section">
            <div class="section-title">Strategy Weights</div>
            {weights_html if weights_html else '<p style="color:#8B949E;">No weights yet -- run pipeline first</p>'}
        </div>
    </div>

    <div class="card section">
        <div class="section-title">Daily P&L History</div>
        <table>
            <tr><th>Date</th><th>P&L</th><th>Return</th><th>Trades</th><th>Regime</th><th>Positions</th><th>Strategies</th></tr>
            {history_html if history_html else '<tr><td colspan="7" style="color:#8B949E;">No history yet</td></tr>'}
        </table>
    </div>

    <div class="grid" style="grid-template-columns: 1fr 1fr 1fr;">
        {_panel("Kronos Signals", "Foundation model - stocks + forex forecasts", "kronos")}
        <div class="card section">
            <div class="section-title">Deep Surrogates Tail Risk</div>
            <div style="margin:8px 0;">
                <span style="font-size:11px; color:#8B949E;">Crash Risk: </span>
                <span class="badge" style="background:{tr_color}22; color:{tr_color};">{tr_label} ({ds_tail_risk_gauge:.2f})</span>
            </div>
            <div style="background:#21262D; border-radius:4px; height:8px; margin-bottom:12px;">
                <div style="background:{tr_color}; width:{min(ds_tail_risk_gauge * 100, 100):.0f}%; height:100%; border-radius:4px;"></div>
            </div>
            <table>
                <tr><th>Symbol</th><th>Score</th><th>Conf</th><th>Dir</th></tr>
                {panel_data.get('deep_surrogates', '') or '<tr><td colspan="4" style="color:#8B949E;">No signals</td></tr>'}
            </table>
        </div>
        {_panel("TDGF vs Market", "American option mispricing alpha", "tdgf")}
    </div>

    <div class="grid" style="grid-template-columns: 1fr 1fr 1fr 1fr;">
        {_panel("Mean Reversion", "OU equilibrium deviation signals", "mean_reversion")}
        {_panel("Sector Rotation", "Macro regime sector tilts", "sector_rotation")}
        {_panel("FX Momentum", "Multi-lookback FX trend", "fx_momentum")}
        {_panel("FX Vol Breakout", "Bollinger squeeze breakouts", "fx_vol_breakout")}
    </div>

    <div class="grid" style="grid-template-columns: 1fr;">
        {_panel("Vol Surface Arbitrage", "IV-RV spread trading", "vol_arb")}
    </div>

    <div style="margin-top:20px; color:#8B949E; font-size:12px;">
        Last run: {state.last_run or 'Never'} |
        Schedule: {SCHEDULE_HOUR}:{SCHEDULE_MINUTE:02d} ET Mon-Fri |
        <a href="/run-now" style="color:#58A6FF;">Trigger Manual Run</a> |
        <a href="/health" style="color:#58A6FF;">Health Check</a>
    </div>
</div>
</body>
</html>"""
    return HTMLResponse(content=html)
