"""
Real data backtest runner for APEX strategies.
Downloads data via yfinance and runs backtests on all strategies.

Usage: python run_backtest.py
"""

import sys

sys.path.insert(0, ".")

import logging
import numpy as np
import pandas as pd
from datetime import datetime

from strategies.momentum.cross_sectional import CrossSectionalMomentum
from strategies.statarb.pairs import PairsTrading
from strategies.fx.carry_trend import FXCarryTrend
from strategies.ensemble.combiner import EnsembleCombiner, TFTAdapter
from strategies.regime.detector import RegimeDetector
from strategies.ensemble.portfolio_optimizer import PortfolioOptimizer
from strategies.risk.portfolio_risk import PortfolioRiskManager
from strategies.backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    compare_strategies,
)
from strategies.config import (
    MomentumConfig,
    StatArbConfig,
    FXConfig,
    EnsembleConfig,
    RegimeConfig,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 70


def load_stock_data() -> pd.DataFrame:
    df = pd.read_csv("data/backtest_stocks.csv", parse_dates=["timestamp"])
    # Remove SPY from trading universe (keep for benchmark)
    stocks = df[df["symbol"] != "SPY"].copy()
    print(f"Stock data: {len(stocks)} rows, {stocks['symbol'].nunique()} symbols")
    print(
        f"Date range: {stocks['timestamp'].min().date()} to {stocks['timestamp'].max().date()}"
    )
    return stocks


def load_benchmark_data() -> pd.DataFrame:
    df = pd.read_csv("data/backtest_stocks.csv", parse_dates=["timestamp"])
    spy = df[df["symbol"] == "SPY"][["timestamp", "close"]].copy()
    return spy


def load_fx_data() -> pd.DataFrame:
    df = pd.read_csv("data/backtest_fx.csv", parse_dates=["timestamp"])
    print(f"FX data: {len(df)} rows, {df['symbol'].nunique()} pairs")
    return df


def sector_mapping() -> dict:
    """Approximate sector mapping for pair scanning."""
    return {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Technology",
        "AMZN": "Consumer",
        "NVDA": "Technology",
        "META": "Technology",
        "TSLA": "Consumer",
        "AMD": "Technology",
        "INTC": "Technology",
        "QCOM": "Technology",
        "CRM": "Technology",
        "NFLX": "Technology",
        "JPM": "Financials",
        "BAC": "Financials",
        "V": "Financials",
        "MA": "Financials",
        "JNJ": "Healthcare",
        "UNH": "Healthcare",
        "ABBV": "Healthcare",
        "MRK": "Healthcare",
        "PG": "Consumer",
        "KO": "Consumer",
        "PEP": "Consumer",
        "COST": "Consumer",
        "WMT": "Consumer",
        "HD": "Consumer",
        "DIS": "Consumer",
        "XOM": "Energy",
        "CVX": "Energy",
    }


def run_momentum_backtest(stocks: pd.DataFrame, benchmark: pd.DataFrame) -> object:
    print(f"\n{SEPARATOR}")
    print("  BACKTEST 1: Cross-Sectional Momentum + Mean Reversion")
    print(SEPARATOR)

    config = MomentumConfig(
        enabled=True,
        momentum_lookback_days=252,
        momentum_skip_days=21,
        momentum_weight=0.50,
        mean_reversion_weight=0.30,
        quality_weight=0.20,
        long_threshold_zscore=1.0,
        short_threshold_zscore=-1.0,
        max_positions_per_side=5,
        min_history_days=260,
        min_avg_dollar_volume=5_000_000,
    )
    strategy = CrossSectionalMomentum(config=config)

    bt_config = BacktestConfig(
        initial_capital=100_000,
        transaction_cost_bps=5.0,
        slippage_bps=2.0,
        rebalance_frequency="weekly",  # weekly to cut turnover
        max_position_weight=0.10,
        max_gross_leverage=1.0,  # long-only-ish to match bull market
        warmup_days=270,
    )
    engine = BacktestEngine(bt_config)
    result = engine.run(strategy, stocks, benchmark)
    result.print_summary()
    return result


def run_statarb_backtest(stocks: pd.DataFrame, benchmark: pd.DataFrame) -> object:
    print(f"\n{SEPARATOR}")
    print("  BACKTEST 2: Statistical Arbitrage (Pairs Trading)")
    print(SEPARATOR)

    config = StatArbConfig(
        enabled=True,
        cointegration_pvalue=0.10,  # relaxed for broader pair selection
        max_half_life_days=45,
        min_half_life_days=2,
        max_pairs=20,
        rescan_interval_days=14,
        entry_zscore=1.5,  # enter earlier for more trades
        exit_zscore=0.3,  # exit closer to mean
        stop_loss_zscore=3.5,
        lookback_window=63,
        same_sector_only=True,
        sector_pairs_limit=4,
    )
    strategy = PairsTrading(config=config)
    strategy.initialize(stocks, sector_mapping())

    bt_config = BacktestConfig(
        initial_capital=100_000,
        transaction_cost_bps=5.0,
        slippage_bps=2.0,
        rebalance_frequency="daily",  # pairs need daily monitoring
        max_position_weight=0.06,
        max_gross_leverage=1.5,
        warmup_days=80,
    )
    engine = BacktestEngine(bt_config)
    result = engine.run(strategy, stocks, benchmark)
    result.print_summary()
    return result


def run_fx_backtest(fx_data: pd.DataFrame) -> object:
    print(f"\n{SEPARATOR}")
    print("  BACKTEST 3: FX Carry + Trend Following")
    print(SEPARATOR)

    config = FXConfig(
        enabled=True,
        carry_weight=0.5,
        trend_weight=0.5,
        trend_lookback_days=63,
        max_pairs_long=3,
        max_pairs_short=3,
    )
    strategy = FXCarryTrend(config=config)

    bt_config = BacktestConfig(
        initial_capital=100_000,
        transaction_cost_bps=3.0,  # lower for FX
        slippage_bps=1.0,
        rebalance_frequency="weekly",
        max_position_weight=0.20,
        max_gross_leverage=1.2,
        warmup_days=70,  # reduced for more trading days
    )
    engine = BacktestEngine(bt_config)
    result = engine.run(strategy, fx_data)
    result.print_summary()
    return result


def main():
    print(SEPARATOR)
    print("  APEX MULTI-STRATEGY BACKTEST — REAL DATA")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(SEPARATOR)

    # Load data
    stocks = load_stock_data()
    benchmark = load_benchmark_data()
    fx_data = load_fx_data()

    # Run individual backtests
    results = []

    r1 = run_momentum_backtest(stocks, benchmark)
    results.append(r1)

    r2 = run_statarb_backtest(stocks, benchmark)
    results.append(r2)

    r3 = run_fx_backtest(fx_data)
    results.append(r3)

    # Comparison table
    print(f"\n{SEPARATOR}")
    print("  STRATEGY COMPARISON")
    print(SEPARATOR)
    comparison = compare_strategies(results)
    print(comparison.to_string())

    # Risk assessment
    print(f"\n{SEPARATOR}")
    print("  PORTFOLIO RISK ASSESSMENT")
    print(SEPARATOR)

    risk_mgr = PortfolioRiskManager(
        max_portfolio_drawdown=0.20,
        var_confidence=0.99,
        correlation_alert_threshold=0.5,
    )

    # Feed strategy returns
    for result in results:
        for ret in result.daily_returns:
            risk_mgr.record_strategy_return(result.strategy_name, ret)
            risk_mgr.record_portfolio_return(ret / len(results))  # equal-weight

        perf = result.daily_returns
        from strategies.base import StrategyPerformance

        sp = StrategyPerformance(strategy_name=result.strategy_name)
        for r in result.daily_returns:
            sp.update(r)
        risk_mgr.update_strategy_performance(result.strategy_name, sp)

    report = risk_mgr.assess()
    print(f"  Portfolio Drawdown:  {report.portfolio_drawdown:.2%}")
    print(f"  Portfolio Sharpe 21d: {report.portfolio_sharpe_21d:.3f}")
    print(f"  Portfolio Sharpe 63d: {report.portfolio_sharpe_63d:.3f}")
    print(f"  Parametric VaR 99%:  {report.var.parametric_var:.3%}")
    print(f"  Historical VaR 99%:  {report.var.historical_var:.3%}")
    print(f"  Portfolio Vol (ann): {report.var.portfolio_vol:.2%}")
    print(
        f"  Active strategies:   {report.active_strategy_count}/{report.total_strategy_count}"
    )
    print(f"  Killed strategies:   {report.killed_strategy_count}")

    if report.correlation_alerts:
        print(f"\n  Correlation Alerts:")
        for alert in report.correlation_alerts:
            print(
                f"    {alert.strategy_a} / {alert.strategy_b}: {alert.correlation:.3f}"
            )

    print(f"\n  Capital Allocation (recommended):")
    for alloc in report.capital_allocations:
        print(
            f"    {alloc.strategy_name}: {alloc.target_fraction:.1%} — {alloc.rationale}"
        )

    print(f"\n{SEPARATOR}")
    print("  BACKTEST COMPLETE")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
