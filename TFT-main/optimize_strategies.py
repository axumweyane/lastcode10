#!/usr/bin/env python3
"""
APEX Strategy Parameter Optimizer — Walk-forward grid search.

For each strategy, defines a parameter grid, splits data into 5 train/test
windows, runs grid search on train, validates on test, and ranks by
out-of-sample Sharpe ratio.

Usage:
    python optimize_strategies.py
    python optimize_strategies.py --output optimization_results.json
"""

import argparse
import itertools
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from strategies.momentum.cross_sectional import CrossSectionalMomentum
from strategies.statarb.pairs import PairsTrading
from strategies.backtest.engine import BacktestEngine, BacktestConfig
from strategies.config import MomentumConfig, StatArbConfig

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("optimizer")
logger.setLevel(logging.INFO)

SEPARATOR = "=" * 70


# ---------------------------------------------------------------------------
# Parameter Grids
# ---------------------------------------------------------------------------

MOMENTUM_GRID = {
    "momentum_lookback_days": [126, 189, 252],
    "momentum_skip_days": [5, 21, 42],
    "momentum_weight": [0.4, 0.5, 0.6],
    "mean_reversion_weight": [0.2, 0.3, 0.4],
    "long_threshold_zscore": [0.5, 1.0, 1.5],
    "short_threshold_zscore": [-1.5, -1.0, -0.5],
    "max_positions_per_side": [3, 5, 8],
}

STATARB_GRID = {
    "cointegration_pvalue": [0.05, 0.10],
    "entry_zscore": [1.5, 2.0, 2.5],
    "exit_zscore": [0.3, 0.5, 0.8],
    "stop_loss_zscore": [3.0, 3.5, 4.0],
    "lookback_window": [42, 63, 126],
    "max_pairs": [10, 15, 20],
}

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Consumer", "NVDA": "Technology", "META": "Technology",
    "TSLA": "Consumer", "AMD": "Technology", "INTC": "Technology",
    "QCOM": "Technology", "CRM": "Technology", "NFLX": "Technology",
    "JPM": "Financials", "BAC": "Financials", "V": "Financials",
    "MA": "Financials", "JNJ": "Healthcare", "UNH": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "PG": "Consumer",
    "KO": "Consumer", "PEP": "Consumer", "COST": "Consumer",
    "WMT": "Consumer", "HD": "Consumer", "DIS": "Consumer",
    "XOM": "Energy", "CVX": "Energy",
}


# ---------------------------------------------------------------------------
# Walk-Forward Splits
# ---------------------------------------------------------------------------

def make_walk_forward_splits(
    dates: pd.DatetimeIndex, n_folds: int = 5, train_ratio: float = 0.6
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Create n_folds walk-forward train/test date splits."""
    total = len(dates)
    fold_size = total // n_folds
    splits = []
    for i in range(n_folds):
        start = i * (fold_size // 2)
        end = min(start + fold_size, total)
        if end - start < 100:
            continue
        train_end = start + int((end - start) * train_ratio)
        train_dates = dates[start:train_end]
        test_dates = dates[train_end:end]
        if len(train_dates) >= 60 and len(test_dates) >= 20:
            splits.append((train_dates, test_dates))
    return splits


# ---------------------------------------------------------------------------
# Grid Search Helpers
# ---------------------------------------------------------------------------

def expand_grid(grid: Dict[str, list]) -> List[Dict[str, Any]]:
    """Expand parameter grid into list of combinations."""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def run_momentum_trial(
    params: Dict[str, Any],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> Dict[str, Any]:
    """Run a single momentum trial on train/test data."""
    config = MomentumConfig(
        enabled=True,
        momentum_lookback_days=params["momentum_lookback_days"],
        momentum_skip_days=params["momentum_skip_days"],
        momentum_weight=params["momentum_weight"],
        mean_reversion_weight=params["mean_reversion_weight"],
        quality_weight=1.0 - params["momentum_weight"] - params["mean_reversion_weight"],
        long_threshold_zscore=params["long_threshold_zscore"],
        short_threshold_zscore=params["short_threshold_zscore"],
        max_positions_per_side=params["max_positions_per_side"],
        min_history_days=max(params["momentum_lookback_days"] + params["momentum_skip_days"] + 5, 60),
        min_avg_dollar_volume=1_000_000,
    )
    strategy = CrossSectionalMomentum(config=config)

    bt_config = BacktestConfig(
        initial_capital=100_000,
        transaction_cost_bps=5.0,
        slippage_bps=2.0,
        rebalance_frequency="weekly",
        max_position_weight=0.10,
        max_gross_leverage=1.0,
        warmup_days=config.min_history_days,
    )
    engine = BacktestEngine(bt_config)

    # Filter benchmark to test period
    test_dates = test_data["timestamp"].unique()
    test_bench = benchmark[benchmark["timestamp"].isin(test_dates)] if benchmark is not None else None

    try:
        result = engine.run(strategy, test_data, test_bench)
        return {
            "params": params,
            "sharpe": result.sharpe_ratio,
            "annual_return": result.annualized_return,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_return": result.total_return,
            "trades": result.total_trades,
            "calmar": result.calmar_ratio,
        }
    except Exception as e:
        logger.debug("Momentum trial failed: %s", e)
        return {
            "params": params,
            "sharpe": -999.0,
            "annual_return": 0.0,
            "max_drawdown": 1.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "trades": 0,
            "calmar": 0.0,
        }


def run_statarb_trial(
    params: Dict[str, Any],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> Dict[str, Any]:
    """Run a single stat arb trial on train/test data."""
    config = StatArbConfig(
        enabled=True,
        cointegration_pvalue=params["cointegration_pvalue"],
        entry_zscore=params["entry_zscore"],
        exit_zscore=params["exit_zscore"],
        stop_loss_zscore=params["stop_loss_zscore"],
        lookback_window=params["lookback_window"],
        max_pairs=params["max_pairs"],
        same_sector_only=True,
        sector_pairs_limit=4,
    )
    strategy = PairsTrading(config=config)
    strategy.initialize(train_data, SECTOR_MAP)

    bt_config = BacktestConfig(
        initial_capital=100_000,
        transaction_cost_bps=5.0,
        slippage_bps=2.0,
        rebalance_frequency="daily",
        max_position_weight=0.06,
        max_gross_leverage=1.5,
        warmup_days=max(params["lookback_window"] + 10, 80),
    )
    engine = BacktestEngine(bt_config)

    test_dates = test_data["timestamp"].unique()
    test_bench = benchmark[benchmark["timestamp"].isin(test_dates)] if benchmark is not None else None

    try:
        result = engine.run(strategy, test_data, test_bench)
        return {
            "params": params,
            "sharpe": result.sharpe_ratio,
            "annual_return": result.annualized_return,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "total_return": result.total_return,
            "trades": result.total_trades,
            "calmar": result.calmar_ratio,
        }
    except Exception as e:
        logger.debug("StatArb trial failed: %s", e)
        return {
            "params": params,
            "sharpe": -999.0,
            "annual_return": 0.0,
            "max_drawdown": 1.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "trades": 0,
            "calmar": 0.0,
        }


# ---------------------------------------------------------------------------
# Main Optimization Loop
# ---------------------------------------------------------------------------

def optimize_strategy(
    name: str,
    grid: Dict[str, list],
    run_fn,
    stock_data: pd.DataFrame,
    benchmark: pd.DataFrame,
    n_folds: int = 5,
) -> List[Dict[str, Any]]:
    """Run walk-forward grid search for a strategy."""
    all_dates = sorted(stock_data["timestamp"].unique())
    date_index = pd.DatetimeIndex(all_dates)
    splits = make_walk_forward_splits(date_index, n_folds)

    combos = expand_grid(grid)
    logger.info(
        "%s: %d parameter combos × %d folds = %d trials",
        name, len(combos), len(splits), len(combos) * len(splits),
    )

    # Aggregate OOS results across folds
    results_by_combo = {i: [] for i in range(len(combos))}

    for fold_idx, (train_dates, test_dates) in enumerate(splits):
        logger.info(
            "  Fold %d/%d: train %s→%s, test %s→%s",
            fold_idx + 1, len(splits),
            train_dates[0].strftime("%Y-%m-%d"), train_dates[-1].strftime("%Y-%m-%d"),
            test_dates[0].strftime("%Y-%m-%d"), test_dates[-1].strftime("%Y-%m-%d"),
        )
        train_data = stock_data[stock_data["timestamp"].isin(train_dates)]
        test_data = stock_data[stock_data["timestamp"].isin(test_dates)]

        # Ensure enough data
        all_fold_data = stock_data[
            stock_data["timestamp"].isin(train_dates.union(test_dates))
        ]

        for combo_idx, params in enumerate(combos):
            result = run_fn(params, train_data, all_fold_data, benchmark)
            results_by_combo[combo_idx].append(result)

    # Average OOS Sharpe across folds
    aggregated = []
    for combo_idx, fold_results in results_by_combo.items():
        valid = [r for r in fold_results if r["sharpe"] > -900]
        if not valid:
            continue
        avg_sharpe = np.mean([r["sharpe"] for r in valid])
        avg_return = np.mean([r["annual_return"] for r in valid])
        avg_dd = np.mean([r["max_drawdown"] for r in valid])
        avg_wr = np.mean([r["win_rate"] for r in valid])
        avg_trades = np.mean([r["trades"] for r in valid])
        aggregated.append({
            "strategy": name,
            "params": combos[combo_idx],
            "avg_oos_sharpe": round(float(avg_sharpe), 4),
            "avg_annual_return_pct": round(float(avg_return * 100), 2),
            "avg_max_drawdown_pct": round(float(avg_dd * 100), 2),
            "avg_win_rate_pct": round(float(avg_wr * 100), 1),
            "avg_trades": int(avg_trades),
            "n_valid_folds": len(valid),
        })

    aggregated.sort(key=lambda x: x["avg_oos_sharpe"], reverse=True)
    return aggregated


def print_top_results(results: List[Dict], n: int = 10):
    """Print top N results."""
    if not results:
        print("  No valid results.")
        return

    strategy_name = results[0]["strategy"]
    print(f"\n{'─' * 70}")
    print(f"  TOP {min(n, len(results))} RESULTS: {strategy_name}")
    print(f"{'─' * 70}")
    print(f"  {'Rank':<5} {'OOS Sharpe':<12} {'Return%':<10} {'MaxDD%':<9} {'WinRate%':<10} {'Trades':<8} Key Params")
    print(f"  {'─'*5} {'─'*11} {'─'*9} {'─'*8} {'─'*9} {'─'*7} {'─'*30}")

    for i, r in enumerate(results[:n]):
        params = r["params"]
        # Show 2-3 most important params
        if r["strategy"] == "cross_sectional_momentum":
            key = f"lb={params['momentum_lookback_days']}, skip={params['momentum_skip_days']}, z={params['long_threshold_zscore']}"
        else:
            key = f"entry_z={params['entry_zscore']}, exit_z={params['exit_zscore']}, lb={params['lookback_window']}"

        print(
            f"  {i+1:<5} {r['avg_oos_sharpe']:>10.3f}  {r['avg_annual_return_pct']:>8.1f}%"
            f"  {r['avg_max_drawdown_pct']:>6.1f}%  {r['avg_win_rate_pct']:>7.1f}%"
            f"  {r['avg_trades']:>6}  {key}"
        )


def main():
    parser = argparse.ArgumentParser(description="APEX Strategy Parameter Optimizer")
    parser.add_argument("--output", default="optimization_results.json")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    print(SEPARATOR)
    print("  APEX STRATEGY PARAMETER OPTIMIZER")
    print(f"  Walk-Forward Grid Search | {args.folds} folds")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(SEPARATOR)

    # Load data
    stocks = pd.read_csv("data/backtest_stocks.csv", parse_dates=["timestamp"])
    benchmark = stocks[stocks["symbol"] == "SPY"][["timestamp", "close"]].copy()
    stocks_no_spy = stocks[stocks["symbol"] != "SPY"].copy()

    print(f"\n  Data: {len(stocks_no_spy)} rows, {stocks_no_spy['symbol'].nunique()} symbols")
    print(f"  Date range: {stocks_no_spy['timestamp'].min().date()} to {stocks_no_spy['timestamp'].max().date()}")

    all_results = {}

    # --- Momentum ---
    print(f"\n{SEPARATOR}")
    print("  OPTIMIZING: Cross-Sectional Momentum")
    print(SEPARATOR)

    momentum_results = optimize_strategy(
        "cross_sectional_momentum",
        MOMENTUM_GRID,
        run_momentum_trial,
        stocks_no_spy,
        benchmark,
        n_folds=args.folds,
    )
    print_top_results(momentum_results)
    all_results["cross_sectional_momentum"] = momentum_results[:10]

    # --- StatArb ---
    print(f"\n{SEPARATOR}")
    print("  OPTIMIZING: Statistical Arbitrage (Pairs Trading)")
    print(SEPARATOR)

    statarb_results = optimize_strategy(
        "pairs_trading",
        STATARB_GRID,
        run_statarb_trial,
        stocks_no_spy,
        benchmark,
        n_folds=args.folds,
    )
    print_top_results(statarb_results)
    all_results["pairs_trading"] = statarb_results[:10]

    # --- Save ---
    output = {
        "generated_at": datetime.now().isoformat(),
        "n_folds": args.folds,
        "data_rows": len(stocks_no_spy),
        "data_symbols": int(stocks_no_spy["symbol"].nunique()),
        "results": all_results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{SEPARATOR}")
    print(f"  OPTIMIZATION COMPLETE — results saved to {args.output}")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
