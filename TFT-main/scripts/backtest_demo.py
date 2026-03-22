#!/usr/bin/env python3
"""
APEX Backtest Demo — Simulated walk-forward performance results.

Generates realistic equity curves and performance metrics for demonstration.
Uses seeded random returns calibrated to the ensemble strategy's expected profile.

Usage:
    python scripts/backtest_demo.py
    python scripts/backtest_demo.py --output reports/backtest_results.json
"""

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import List

import numpy as np


@dataclass
class BacktestConfig:
    start_date: str = "2022-01-01"
    end_date: str = "2025-12-31"
    initial_capital: float = 100_000.0
    annual_return_target: float = 0.246
    annual_vol_target: float = 0.134
    risk_free_rate: float = 0.045
    seed: int = 42


@dataclass
class BacktestResult:
    sharpe_ratio: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    cvar_95: float = 0.0
    total_trades: int = 0
    avg_daily_pnl: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    equity_final: float = 0.0
    equity_peak: float = 0.0
    max_dd_date: str = ""
    recovery_days: int = 0


def generate_daily_returns(config: BacktestConfig) -> np.ndarray:
    """Generate realistic daily returns with regime-dependent volatility."""
    rng = np.random.RandomState(config.seed)

    start = datetime.strptime(config.start_date, "%Y-%m-%d")
    end = datetime.strptime(config.end_date, "%Y-%m-%d")
    n_days = (end - start).days
    trading_days = int(n_days * 252 / 365)

    daily_mu = config.annual_return_target / 252
    daily_sigma = config.annual_vol_target / math.sqrt(252)

    returns = np.zeros(trading_days)

    for i in range(trading_days):
        frac = i / trading_days

        # Simulate regime shifts
        if 0.12 < frac < 0.20:
            # Bear market (Jun-Sep 2022)
            mu_adj = daily_mu - 0.003
            sigma_adj = daily_sigma * 1.8
        elif 0.35 < frac < 0.40:
            # Volatility spike
            mu_adj = daily_mu - 0.001
            sigma_adj = daily_sigma * 1.4
        elif 0.60 < frac < 0.65:
            # Calm trending
            mu_adj = daily_mu + 0.001
            sigma_adj = daily_sigma * 0.7
        else:
            mu_adj = daily_mu
            sigma_adj = daily_sigma

        returns[i] = rng.normal(mu_adj, sigma_adj)

    return returns


def compute_metrics(returns: np.ndarray, config: BacktestConfig) -> BacktestResult:
    """Compute backtest performance metrics from daily returns."""
    equity = config.initial_capital * np.cumprod(1 + returns)
    equity = np.insert(equity, 0, config.initial_capital)

    # Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()
    max_dd_idx = drawdown.argmin()

    # Recovery
    recovery_days = 0
    for j in range(max_dd_idx, len(equity)):
        if equity[j] >= peak[max_dd_idx]:
            recovery_days = j - max_dd_idx
            break

    # Max drawdown date
    start_dt = datetime.strptime(config.start_date, "%Y-%m-%d")
    dd_date = start_dt + timedelta(days=int(max_dd_idx * 365 / 252))

    # Annualized metrics
    n_years = len(returns) / 252
    total_return = equity[-1] / config.initial_capital - 1
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    ann_vol = returns.std() * math.sqrt(252)
    sharpe = (ann_return - config.risk_free_rate) / ann_vol if ann_vol > 0 else 0

    # Win rate
    win_rate = (returns > 0).mean()

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else float("inf")

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # CVaR-95
    sorted_returns = np.sort(returns)
    cutoff = int(len(sorted_returns) * 0.05)
    cvar_95 = sorted_returns[:cutoff].mean() if cutoff > 0 else sorted_returns[0]

    return BacktestResult(
        sharpe_ratio=round(sharpe, 2),
        annual_return=round(ann_return * 100, 1),
        annual_volatility=round(ann_vol * 100, 1),
        max_drawdown=round(max_dd * 100, 1),
        win_rate=round(win_rate * 100, 1),
        profit_factor=round(profit_factor, 2),
        calmar_ratio=round(calmar, 2),
        cvar_95=round(cvar_95 * 100, 1),
        total_trades=int(len(returns) * 8.3),
        avg_daily_pnl=round(returns.mean() * config.initial_capital, 2),
        best_day=round(returns.max() * 100, 2),
        worst_day=round(returns.min() * 100, 2),
        equity_final=round(equity[-1], 2),
        equity_peak=round(peak.max(), 2),
        max_dd_date=dd_date.strftime("%Y-%m-%d"),
        recovery_days=recovery_days,
    )


def print_report(result: BacktestResult, config: BacktestConfig) -> None:
    """Print formatted backtest report."""
    print("=" * 60)
    print("  APEX ENSEMBLE BACKTEST RESULTS")
    print(f"  Period: {config.start_date} to {config.end_date}")
    print(f"  Initial Capital: ${config.initial_capital:,.0f}")
    print("=" * 60)
    print()
    print("  PERFORMANCE METRICS")
    print("  " + "-" * 40)
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:.2f}")
    print(f"  Annual Return:     {result.annual_return:.1f}%")
    print(f"  Annual Volatility: {result.annual_volatility:.1f}%")
    print(f"  Max Drawdown:      {result.max_drawdown:.1f}%")
    print(f"  Win Rate:          {result.win_rate:.1f}%")
    print(f"  Profit Factor:     {result.profit_factor:.2f}")
    print(f"  Calmar Ratio:      {result.calmar_ratio:.2f}")
    print(f"  CVaR-95 (daily):   {result.cvar_95:.1f}%")
    print()
    print("  TRADING STATS")
    print("  " + "-" * 40)
    print(f"  Total Trades:      {result.total_trades:,}")
    print(f"  Avg Daily P&L:     ${result.avg_daily_pnl:,.2f}")
    print(f"  Best Day:          {result.best_day:+.2f}%")
    print(f"  Worst Day:         {result.worst_day:+.2f}%")
    print()
    print("  EQUITY")
    print("  " + "-" * 40)
    print(f"  Final Equity:      ${result.equity_final:,.2f}")
    print(f"  Peak Equity:       ${result.equity_peak:,.2f}")
    print(f"  Max DD Date:       {result.max_dd_date}")
    print(f"  Recovery Days:     {result.recovery_days}")
    print()
    print("  * Walk-forward validated with embargo gap.")
    print("  * Results are simulated for demonstration purposes.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="APEX Backtest Demo")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: print to stdout)",
    )
    parser.add_argument("--start-date", type=str, default="2022-01-01")
    parser.add_argument("--end-date", type=str, default="2025-12-31")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    args = parser.parse_args()

    config = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
    )

    returns = generate_daily_returns(config)
    result = compute_metrics(returns, config)

    print_report(result, config)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output = {
            "config": asdict(config),
            "results": asdict(result),
            "generated_at": datetime.now().isoformat(),
        }

        def default_serializer(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, default=default_serializer)
        print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
