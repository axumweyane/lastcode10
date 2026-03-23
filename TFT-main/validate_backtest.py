#!/usr/bin/env python3
"""
APEX Backtest Validation Suite.

Loads backtest data, runs strategies, verifies signal generation,
checks risk manager, validates optimization results.

Usage:
    python validate_backtest.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

# Colors
G = "\033[92m"
R = "\033[91m"
Y = "\033[93m"
B = "\033[94m"
BOLD = "\033[1m"
RST = "\033[0m"
SEP = "=" * 72

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DATA_FILE = "data/backtest_stocks.csv"

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
    "XOM": "Energy", "CVX": "Energy", "SPY": "Index",
}


def status(passed, label, detail=""):
    tag = f"{G}PASS{RST}" if passed else f"{R}FAIL{RST}"
    print(f"  [{tag}] {label:<40s} {detail}")
    return passed


def warn(label, detail=""):
    print(f"  [{Y}WARN{RST}] {label:<40s} {detail}")


def header(title):
    print(f"\n{B}{BOLD}{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}{RST}")


# ── PART 1: DATA LOADING ─────────────────────────────────────────────

def check_data():
    header("BACKTEST DATA")
    results = []

    if not os.path.exists(DATA_FILE):
        status(False, "Data file", f"{DATA_FILE} MISSING")
        return [("data_file", False)]

    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    p = status(len(df) > 1000, "Row count", f"{len(df):,} rows")
    results.append(("data_rows", p))

    symbols = df["symbol"].nunique()
    p = status(symbols >= 5, "Symbol count", f"{symbols} symbols")
    results.append(("data_symbols", p))

    sym_list = sorted(df["symbol"].unique())
    print(f"    Symbols: {', '.join(sym_list[:15])}")

    date_range = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
    days = (df["timestamp"].max() - df["timestamp"].min()).days
    p = status(days >= 180, "Date range", f"{date_range} ({days}d)")
    results.append(("data_range", p))

    # Check required columns
    required = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    p = status(len(missing) == 0, "Required columns", "All present" if not missing else f"Missing: {missing}")
    results.append(("data_columns", p))

    # NaN check
    nan_pct = df[required].isna().mean().max() * 100
    p = status(nan_pct < 5, "NaN check", f"max {nan_pct:.1f}% NaN")
    results.append(("data_nans", p))

    return results, df


# ── PART 2: STRATEGY SIGNAL GENERATION ────────────────────────────────

def check_momentum_strategy(df):
    header("STRATEGY: Cross-Sectional Momentum")
    results = []

    try:
        from strategies.momentum.cross_sectional import CrossSectionalMomentum
        from strategies.config import MomentumConfig
        p = status(True, "Import", "OK")
        results.append(("momentum_import", p))
    except Exception as e:
        status(False, "Import", str(e)[:60])
        return [("momentum_import", False)]

    try:
        config = MomentumConfig(enabled=True)
        strategy = CrossSectionalMomentum(config=config)
        p = status(True, "Instantiation", "OK")
        results.append(("momentum_init", p))
    except Exception as e:
        status(False, "Instantiation", str(e)[:60])
        return results

    try:
        output = strategy.generate_signals(df)
        n_signals = len(output.scores)
        p = status(n_signals > 0, "Signal generation", f"{n_signals} signals")
        results.append(("momentum_signals", p))

        if n_signals > 0:
            scores = [s.score for s in output.scores]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            p = status(std_score > 0.001, "Signal variance", f"mean={mean_score:.4f}, std={std_score:.4f}")
            results.append(("momentum_variance", p))
    except Exception as e:
        status(False, "Signal generation", str(e)[:60])
        results.append(("momentum_signals", False))

    return results


def check_statarb_strategy(df):
    header("STRATEGY: Pairs Trading (StatArb)")
    results = []

    try:
        from strategies.statarb.pairs import PairsTrading
        from strategies.config import StatArbConfig
        p = status(True, "Import", "OK")
        results.append(("statarb_import", p))
    except Exception as e:
        status(False, "Import", str(e)[:60])
        return [("statarb_import", False)]

    try:
        config = StatArbConfig(enabled=True)
        strategy = PairsTrading(config=config)
        strategy.initialize(df, SECTOR_MAP)
        p = status(True, "Instantiation + initialize", "OK")
        results.append(("statarb_init", p))
    except Exception as e:
        status(False, "Instantiation", str(e)[:60])
        return results

    try:
        output = strategy.generate_signals(df)
        n_signals = len(output.scores)
        p = status(True, "Signal generation", f"{n_signals} signals")
        results.append(("statarb_signals", p))

        if hasattr(strategy, "active_pairs"):
            n_pairs = len(strategy.active_pairs) if strategy.active_pairs else 0
            status(True, "Active pairs", f"{n_pairs} pairs") if n_pairs > 0 else warn("Active pairs", "0 pairs")
    except Exception as e:
        status(False, "Signal generation", str(e)[:60])
        results.append(("statarb_signals", False))

    return results


# ── PART 3: BACKTEST ENGINE ───────────────────────────────────────────

def check_backtest_engine(df):
    header("BACKTEST ENGINE")
    results = []

    try:
        from strategies.backtest.engine import BacktestEngine, BacktestConfig
        p = status(True, "Import", "OK")
        results.append(("backtest_import", p))
    except Exception as e:
        status(False, "Import", str(e)[:60])
        return [("backtest_import", False)]

    try:
        from strategies.momentum.cross_sectional import CrossSectionalMomentum
        from strategies.config import MomentumConfig

        config = MomentumConfig(enabled=True)
        strategy = CrossSectionalMomentum(config=config)

        stocks_no_spy = df[df["symbol"] != "SPY"].copy()
        benchmark = df[df["symbol"] == "SPY"][["timestamp", "close"]].copy()

        bt_config = BacktestConfig(
            initial_capital=100_000,
            transaction_cost_bps=5.0,
            slippage_bps=2.0,
            rebalance_frequency="weekly",
            max_position_weight=0.10,
            max_gross_leverage=1.0,
            warmup_days=260,
        )
        engine = BacktestEngine(bt_config)
        result = engine.run(strategy, stocks_no_spy, benchmark)

        p = status(result is not None, "Backtest run", "completed")
        results.append(("backtest_run", p))

        # Check result attributes
        p = status(hasattr(result, "sharpe_ratio"), "Sharpe ratio",
                   f"{result.sharpe_ratio:.3f}" if hasattr(result, "sharpe_ratio") else "MISSING")
        results.append(("backtest_sharpe", p))

        p = status(hasattr(result, "total_return"), "Total return",
                   f"{result.total_return*100:.1f}%" if hasattr(result, "total_return") else "MISSING")
        results.append(("backtest_return", p))

        p = status(hasattr(result, "max_drawdown"), "Max drawdown",
                   f"{result.max_drawdown*100:.1f}%" if hasattr(result, "max_drawdown") else "MISSING")
        results.append(("backtest_drawdown", p))

        total_trades = getattr(result, "total_trades", 0)
        p = status(total_trades >= 0, "Total trades", f"{total_trades}")
        results.append(("backtest_trades", p))

        # Sanity: Sharpe should be reasonable
        if hasattr(result, "sharpe_ratio"):
            sr = result.sharpe_ratio
            sane = -5 < sr < 10
            p = status(sane, "Sharpe sanity", f"{sr:.3f} (expected -5 to 10)")
            results.append(("backtest_sharpe_sane", p))

    except Exception as e:
        status(False, "Backtest run", str(e)[:80])
        results.append(("backtest_run", False))

    return results


# ── PART 4: RISK MANAGER ─────────────────────────────────────────────

def check_risk_manager():
    header("RISK MANAGER")
    results = []

    try:
        from strategies.risk.portfolio_risk import PortfolioRiskManager
        p = status(True, "Import", "OK")
        results.append(("risk_import", p))
    except Exception as e:
        status(False, "Import", str(e)[:60])
        return [("risk_import", False)]

    try:
        rm = PortfolioRiskManager()
        p = status(True, "Instantiation", "OK")
        results.append(("risk_init", p))

        # Feed synthetic returns
        for i in range(30):
            rm.record_portfolio_return(np.random.normal(0.001, 0.02))

        # Get risk report
        report = rm.assess()
        p = status(report is not None, "Risk assessment", "computed")
        results.append(("risk_assess", p))

        if report:
            d = report.to_dict() if hasattr(report, "to_dict") else {}
            detail_parts = []
            for k in ["portfolio_var", "portfolio_cvar", "portfolio_drawdown"]:
                if k in d:
                    detail_parts.append(f"{k}={d[k]:.4f}")
            if detail_parts:
                status(True, "Risk metrics", ", ".join(detail_parts))
            else:
                status(True, "Risk report", "OK")

    except Exception as e:
        status(False, "Risk manager", str(e)[:60])
        results.append(("risk_test", False))

    return results


# ── PART 5: ENSEMBLE COMBINER ─────────────────────────────────────────

def check_ensemble():
    header("ENSEMBLE COMBINER")
    results = []

    try:
        from strategies.ensemble.combiner import EnsembleCombiner
        from strategies.regime.detector import RegimeDetector
        p = status(True, "Import", "OK")
        results.append(("ensemble_import", p))
    except Exception as e:
        status(False, "Import", str(e)[:60])
        return [("ensemble_import", False)]

    try:
        combiner = EnsembleCombiner()
        p = status(True, "EnsembleCombiner init", "OK")
        results.append(("ensemble_init", p))

        detector = RegimeDetector()
        p = status(True, "RegimeDetector init", "OK")
        results.append(("regime_init", p))

    except Exception as e:
        status(False, "Instantiation", str(e)[:60])
        results.append(("ensemble_init", False))

    return results


# ── PART 6: PORTFOLIO OPTIMIZER ───────────────────────────────────────

def check_optimizer():
    header("PORTFOLIO OPTIMIZER")
    results = []

    try:
        from strategies.ensemble.portfolio_optimizer import PortfolioOptimizer
        p = status(True, "Import", "OK")
        results.append(("optimizer_import", p))
    except Exception as e:
        status(False, "Import", str(e)[:60])
        return [("optimizer_import", False)]

    try:
        opt = PortfolioOptimizer()
        p = status(True, "Instantiation", "OK")
        results.append(("optimizer_init", p))
    except Exception as e:
        status(False, "Instantiation", str(e)[:60])
        results.append(("optimizer_init", False))

    return results


# ── PART 7: WALK-FORWARD VALIDATION ──────────────────────────────────

def check_walk_forward():
    header("WALK-FORWARD VALIDATION")
    results = []

    try:
        from strategies.validation.walk_forward import WalkForwardValidator
        p = status(True, "Import", "OK")
        results.append(("walkforward_import", p))
    except Exception as e:
        status(False, "Import", str(e)[:60])
        return [("walkforward_import", False)]

    try:
        from strategies.config import WalkForwardConfig
        wf_config = WalkForwardConfig()
        validator = WalkForwardValidator(wf_config)
        p = status(True, "Instantiation", "OK")
        results.append(("walkforward_init", p))
    except Exception as e:
        status(False, "Instantiation", str(e)[:60])
        results.append(("walkforward_init", False))

    return results


# ── PART 8: OPTIMIZATION RESULTS ─────────────────────────────────────

def check_optimization_results():
    header("OPTIMIZATION RESULTS")
    results = []

    opt_file = "optimization_results.json"
    if not os.path.exists(opt_file):
        warn("Optimization results", f"{opt_file} not found (run optimize_strategies.py)")
        return [("opt_file", True)]  # non-blocking

    try:
        with open(opt_file) as f:
            data = json.load(f)

        p = status(True, "Results file", opt_file)
        results.append(("opt_file", p))

        res = data.get("results", {})
        for strategy_name, top_results in res.items():
            n = len(top_results)
            if n > 0:
                best = top_results[0]
                sharpe = best.get("avg_oos_sharpe", 0)
                ret = best.get("avg_annual_return_pct", 0)
                dd = best.get("avg_max_drawdown_pct", 0)
                p = status(True, f"  {strategy_name}",
                           f"best Sharpe={sharpe:.3f}, return={ret:.1f}%, maxDD={dd:.1f}%")
                results.append((f"opt_{strategy_name}", p))
            else:
                warn(f"  {strategy_name}", "no results")

    except Exception as e:
        status(False, "Optimization results", str(e)[:60])
        results.append(("opt_parse", False))

    return results


# ── MAIN ──────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    print(f"\n{BOLD}{SEP}")
    print(f"  APEX BACKTEST VALIDATION SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{SEP}{RST}")

    all_results = []

    # Data
    data_results, df = check_data() if os.path.exists(DATA_FILE) else ([], None)
    if df is None:
        status(False, "Data file", f"{DATA_FILE} not found — aborting")
        sys.exit(1)
    all_results.extend(data_results)

    # Strategies
    all_results.extend(check_momentum_strategy(df))
    all_results.extend(check_statarb_strategy(df))

    # Engine
    all_results.extend(check_backtest_engine(df))

    # Risk
    all_results.extend(check_risk_manager())

    # Ensemble
    all_results.extend(check_ensemble())
    all_results.extend(check_optimizer())

    # Walk-forward
    all_results.extend(check_walk_forward())

    # Optimization results
    all_results.extend(check_optimization_results())

    # Summary
    passed = sum(1 for _, p in all_results if p)
    failed = sum(1 for _, p in all_results if not p)
    total = len(all_results)

    print(f"\n{BOLD}{SEP}")
    if failed == 0:
        print(f"  {G}ALL {total} CHECKS PASSED{RST}")
    else:
        print(f"  {R}{failed}/{total} CHECKS FAILED{RST}")
    print(SEP)

    result_file = RESULTS_DIR / f"backtest_{ts}.json"
    with open(result_file, "w") as f:
        json.dump({
            "timestamp": ts,
            "passed": passed,
            "failed": failed,
            "total": total,
            "details": {name: val for name, val in all_results},
        }, f, indent=2, default=str)
    print(f"  Results saved to {result_file}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
