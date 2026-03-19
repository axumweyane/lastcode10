"""
Vectorized backtest engine for APEX strategy validation.

Runs any BaseStrategy (or the full ensemble pipeline) over historical data
and produces performance metrics, equity curves, and comparison reports.

Design choices:
    - Vectorized where possible (no row-by-row loops for P&L calculation)
    - Forward-looking bias protection: signals on day T use only data up to T
    - Transaction costs modeled as fixed bps per trade
    - Supports daily rebalancing with configurable frequency

Usage:
    engine = BacktestEngine(initial_capital=100000)
    result = engine.run(strategy, historical_data)
    result.print_summary()
    result.equity_curve  # pd.Series
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, SignalDirection, StrategyOutput

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest parameters."""
    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 5.0     # 5 bps round trip (retail via Alpaca)
    slippage_bps: float = 2.0             # 2 bps slippage estimate
    rebalance_frequency: str = "daily"    # "daily" or "weekly"
    max_position_weight: float = 0.10     # 10% max per position
    max_gross_leverage: float = 2.0
    warmup_days: int = 280                # days of data before first trade
    benchmark_symbol: Optional[str] = None  # e.g. "SPY" for comparison


@dataclass
class TradeRecord:
    """Single trade for the audit trail."""
    date: datetime
    symbol: str
    direction: str       # "long" or "short"
    weight: float        # portfolio weight
    entry_price: float
    daily_return: float
    pnl_dollar: float
    cost_bps: float


@dataclass
class BacktestResult:
    """Complete backtest output."""
    strategy_name: str
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    trading_days: int

    # Equity
    equity_curve: pd.Series          # daily portfolio value
    daily_returns: pd.Series         # daily portfolio returns
    benchmark_returns: Optional[pd.Series] = None

    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    calmar_ratio: float = 0.0        # ann return / max drawdown
    win_rate: float = 0.0
    profit_factor: float = 0.0       # gross profit / gross loss
    avg_daily_return: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    total_trades: int = 0
    total_costs: float = 0.0

    # Comparison
    alpha: float = 0.0               # vs benchmark
    beta: float = 0.0
    information_ratio: float = 0.0

    # Detail
    monthly_returns: Optional[pd.Series] = None
    trade_log: List[TradeRecord] = field(default_factory=list)

    def print_summary(self) -> str:
        """Formatted performance summary."""
        lines = [
            f"\n{'=' * 60}",
            f"  BACKTEST RESULTS: {self.strategy_name}",
            f"{'=' * 60}",
            f"  Period: {self.start_date.date()} to {self.end_date.date()} ({self.trading_days} days)",
            f"  Initial Capital: ${self.config.initial_capital:,.0f}",
            f"  Final Value: ${self.equity_curve.iloc[-1]:,.0f}",
            f"",
            f"  --- Returns ---",
            f"  Total Return:      {self.total_return:+.2%}",
            f"  Annualized Return: {self.annualized_return:+.2%}",
            f"  Ann. Volatility:   {self.annualized_volatility:.2%}",
            f"  Best Day:          {self.best_day:+.2%}",
            f"  Worst Day:         {self.worst_day:+.2%}",
            f"",
            f"  --- Risk-Adjusted ---",
            f"  Sharpe Ratio:      {self.sharpe_ratio:.3f}",
            f"  Sortino Ratio:     {self.sortino_ratio:.3f}",
            f"  Calmar Ratio:      {self.calmar_ratio:.3f}",
            f"  Max Drawdown:      {self.max_drawdown:.2%}",
            f"  Max DD Duration:   {self.max_drawdown_duration_days} days",
            f"",
            f"  --- Trading ---",
            f"  Win Rate:          {self.win_rate:.1%}",
            f"  Profit Factor:     {self.profit_factor:.2f}",
            f"  Total Trades:      {self.total_trades}",
            f"  Total Costs:       ${self.total_costs:,.0f}",
        ]

        if self.benchmark_returns is not None:
            lines.extend([
                f"",
                f"  --- vs Benchmark ---",
                f"  Alpha:             {self.alpha:+.2%}",
                f"  Beta:              {self.beta:.3f}",
                f"  Info Ratio:        {self.information_ratio:.3f}",
            ])

        lines.append(f"{'=' * 60}")
        report = "\n".join(lines)
        print(report)
        return report


class BacktestEngine:
    """
    Runs a strategy over historical data and produces performance metrics.

    The engine handles the simulation loop:
        1. For each rebalance date:
           a. Slice data up to that date (no forward-looking bias)
           b. Call strategy.generate_signals(data_so_far)
           c. Convert signals to target weights
           d. Calculate P&L vs previous day
           e. Apply transaction costs on weight changes
        2. Compute all metrics from the equity curve
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        strategy: BaseStrategy,
        historical_data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run backtest.

        Args:
            strategy: Any BaseStrategy implementation.
            historical_data: DataFrame with [symbol, timestamp, open, high,
                            low, close, volume].
            benchmark_data: Optional DataFrame with [timestamp, close] for
                           a benchmark (e.g., SPY).

        Returns:
            BacktestResult with full metrics and equity curve.
        """
        logger.info(
            "Starting backtest for '%s' with $%,.0f capital",
            strategy.name, self.config.initial_capital,
        )

        # Sort and get unique dates
        historical_data = historical_data.sort_values(["symbol", "timestamp"])
        all_dates = sorted(historical_data["timestamp"].unique())

        if len(all_dates) <= self.config.warmup_days:
            raise ValueError(
                f"Need > {self.config.warmup_days} days of data, "
                f"got {len(all_dates)}"
            )

        # Rebalance dates (skip warmup)
        trade_dates = all_dates[self.config.warmup_days:]
        if self.config.rebalance_frequency == "weekly":
            # Keep only Mondays (or first day of each week)
            trade_dates_dt = pd.DatetimeIndex(trade_dates)
            week_groups = trade_dates_dt.to_series().groupby(
                trade_dates_dt.isocalendar().week
            ).first()
            trade_dates = sorted(week_groups.values)

        logger.info(
            "Backtest: %d total dates, %d warmup, %d rebalance dates",
            len(all_dates), self.config.warmup_days, len(trade_dates),
        )

        # Initialize strategy with warmup data
        warmup_data = historical_data[
            historical_data["timestamp"] <= all_dates[self.config.warmup_days - 1]
        ]
        strategy.initialize(warmup_data)

        # Build price lookup: {(symbol, date): close}
        price_lookup = dict(
            zip(
                zip(historical_data["symbol"], historical_data["timestamp"]),
                historical_data["close"],
            )
        )

        # Simulation state
        capital = self.config.initial_capital
        current_weights: Dict[str, float] = {}
        equity_values: List[float] = [capital]
        equity_dates: List = [all_dates[self.config.warmup_days - 1]]
        trade_log: List[TradeRecord] = []
        total_costs = 0.0

        # Run through each rebalance date
        for i, date in enumerate(trade_dates):
            # Data up to this date (no look-ahead)
            data_slice = historical_data[
                historical_data["timestamp"] <= date
            ]

            # Generate signals
            try:
                output = strategy.generate_signals(data_slice)
            except Exception as e:
                logger.warning("Signal generation failed on %s: %s", date, e)
                equity_values.append(capital)
                equity_dates.append(date)
                continue

            # Convert signals to target weights
            target_weights = self._signals_to_weights(output)

            # Calculate daily P&L from existing positions
            daily_pnl = 0.0
            symbols_in_universe = data_slice["symbol"].unique()

            for sym, weight in current_weights.items():
                if abs(weight) < 1e-6:
                    continue

                # Get today's return for this symbol
                prev_date = trade_dates[i - 1] if i > 0 else all_dates[self.config.warmup_days - 1]
                price_today = price_lookup.get((sym, date))
                price_prev = price_lookup.get((sym, prev_date))

                if price_today is not None and price_prev is not None and price_prev > 0:
                    sym_return = (price_today - price_prev) / price_prev
                    position_pnl = weight * capital * sym_return
                    daily_pnl += position_pnl

                    trade_log.append(TradeRecord(
                        date=date, symbol=sym, direction="long" if weight > 0 else "short",
                        weight=weight, entry_price=price_prev,
                        daily_return=sym_return, pnl_dollar=position_pnl,
                        cost_bps=0.0,
                    ))

            # Transaction costs on weight changes (turnover)
            turnover_cost = self._compute_turnover_cost(
                current_weights, target_weights, capital,
            )
            total_costs += turnover_cost
            daily_pnl -= turnover_cost

            # Update state
            capital += daily_pnl
            current_weights = target_weights.copy()
            equity_values.append(capital)
            equity_dates.append(date)

            # Update strategy performance for kill switch tracking
            daily_ret = daily_pnl / (capital - daily_pnl) if (capital - daily_pnl) > 0 else 0
            strategy.get_performance().update(daily_ret)

        # Build equity curve
        equity_curve = pd.Series(equity_values, index=equity_dates, name="equity")
        daily_returns = equity_curve.pct_change().dropna()

        # Benchmark
        bench_returns = None
        if benchmark_data is not None:
            bench_returns = self._align_benchmark(benchmark_data, daily_returns)

        # Compute metrics
        result = self._compute_metrics(
            strategy_name=strategy.name,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            benchmark_returns=bench_returns,
            trade_log=trade_log,
            total_costs=total_costs,
        )

        return result

    # ------------------------------------------------------------------
    # Signal conversion
    # ------------------------------------------------------------------

    def _signals_to_weights(self, output: StrategyOutput) -> Dict[str, float]:
        """Convert strategy signals to position weights."""
        weights: Dict[str, float] = {}

        for score in output.scores:
            if score.direction == SignalDirection.NEUTRAL:
                continue

            # Weight proportional to score * confidence
            raw_weight = score.score * score.confidence

            # Cap individual position
            capped = max(
                -self.config.max_position_weight,
                min(self.config.max_position_weight, raw_weight),
            )

            if abs(capped) > 0.005:
                weights[score.symbol] = capped

        # Enforce gross leverage
        gross = sum(abs(w) for w in weights.values())
        if gross > self.config.max_gross_leverage:
            scale = self.config.max_gross_leverage / gross
            weights = {s: w * scale for s, w in weights.items()}

        return weights

    def _compute_turnover_cost(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
        capital: float,
    ) -> float:
        """Calculate transaction costs from portfolio turnover."""
        all_symbols = set(old_weights) | set(new_weights)
        turnover = 0.0

        for sym in all_symbols:
            old_w = old_weights.get(sym, 0.0)
            new_w = new_weights.get(sym, 0.0)
            turnover += abs(new_w - old_w)

        cost_rate = (self.config.transaction_cost_bps + self.config.slippage_bps) / 10_000
        return turnover * capital * cost_rate

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        strategy_name: str,
        equity_curve: pd.Series,
        daily_returns: pd.Series,
        benchmark_returns: Optional[pd.Series],
        trade_log: List[TradeRecord],
        total_costs: float,
    ) -> BacktestResult:
        """Compute all performance metrics from equity curve."""
        n_days = len(daily_returns)
        years = n_days / 252

        # Basic returns
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        ann_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1
        ann_vol = float(daily_returns.std() * np.sqrt(252))

        # Sharpe
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        # Sortino (downside deviation only)
        downside = daily_returns[daily_returns < 0]
        downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 0 else ann_vol
        sortino = ann_return / downside_vol if downside_vol > 0 else 0.0

        # Drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        max_dd = float(-drawdowns.min())

        # Max drawdown duration
        dd_duration = self._max_drawdown_duration(drawdowns)

        # Calmar
        calmar = ann_return / max_dd if max_dd > 0 else 0.0

        # Win rate and profit factor
        positive_days = daily_returns[daily_returns > 0]
        negative_days = daily_returns[daily_returns < 0]
        win_rate = len(positive_days) / n_days if n_days > 0 else 0.0

        gross_profit = float(positive_days.sum()) if len(positive_days) > 0 else 0.0
        gross_loss = float(-negative_days.sum()) if len(negative_days) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Monthly returns
        monthly = daily_returns.resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        ) if hasattr(daily_returns.index, "freq") or isinstance(
            daily_returns.index, pd.DatetimeIndex
        ) else None

        # Benchmark comparison
        alpha, beta, info_ratio = 0.0, 0.0, 0.0
        if benchmark_returns is not None and len(benchmark_returns) > 10:
            alpha, beta, info_ratio = self._benchmark_stats(
                daily_returns, benchmark_returns,
            )

        result = BacktestResult(
            strategy_name=strategy_name,
            config=self.config,
            start_date=equity_curve.index[0],
            end_date=equity_curve.index[-1],
            trading_days=n_days,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            benchmark_returns=benchmark_returns,
            total_return=total_return,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration_days=dd_duration,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_daily_return=float(daily_returns.mean()),
            best_day=float(daily_returns.max()),
            worst_day=float(daily_returns.min()),
            total_trades=len(trade_log),
            total_costs=total_costs,
            alpha=alpha,
            beta=beta,
            information_ratio=info_ratio,
            monthly_returns=monthly,
            trade_log=trade_log,
        )

        return result

    @staticmethod
    def _max_drawdown_duration(drawdowns: pd.Series) -> int:
        """Longest consecutive period in drawdown (days)."""
        in_dd = drawdowns < -1e-6
        if not in_dd.any():
            return 0

        groups = (~in_dd).cumsum()
        dd_groups = in_dd.groupby(groups).sum()
        return int(dd_groups.max())

    @staticmethod
    def _benchmark_stats(
        returns: pd.Series, benchmark: pd.Series,
    ) -> Tuple[float, float, float]:
        """Compute alpha, beta, information ratio vs benchmark."""
        # Align
        common = returns.index.intersection(benchmark.index)
        if len(common) < 10:
            return 0.0, 0.0, 0.0

        r = returns.loc[common].values
        b = benchmark.loc[common].values

        # Beta = cov(r, b) / var(b)
        cov_matrix = np.cov(r, b)
        var_b = cov_matrix[1, 1]
        beta = float(cov_matrix[0, 1] / var_b) if var_b > 0 else 0.0

        # Alpha (annualized) = ann_r - beta * ann_b
        ann_r = float(np.mean(r) * 252)
        ann_b = float(np.mean(b) * 252)
        alpha = ann_r - beta * ann_b

        # Information ratio = mean(excess) / std(excess)
        excess = r - b
        ir = float(np.mean(excess) / np.std(excess) * np.sqrt(252)) if np.std(excess) > 0 else 0.0

        return alpha, beta, ir

    @staticmethod
    def _align_benchmark(
        benchmark_data: pd.DataFrame,
        daily_returns: pd.Series,
    ) -> pd.Series:
        """Align benchmark data to the backtest return dates."""
        if "timestamp" in benchmark_data.columns:
            bench = benchmark_data.set_index("timestamp")["close"]
        elif "close" in benchmark_data.columns:
            bench = benchmark_data["close"]
        else:
            return pd.Series(dtype=float)

        bench_returns = bench.pct_change().dropna()
        bench_returns.index = pd.DatetimeIndex(bench_returns.index)
        return bench_returns


def compare_strategies(results: List[BacktestResult]) -> pd.DataFrame:
    """
    Side-by-side comparison table for multiple backtest results.

    Usage:
        comparison = compare_strategies([result_a, result_b, result_c])
        print(comparison)
    """
    rows = []
    for r in results:
        rows.append({
            "Strategy": r.strategy_name,
            "Total Return": f"{r.total_return:+.1%}",
            "Ann. Return": f"{r.annualized_return:+.1%}",
            "Ann. Vol": f"{r.annualized_volatility:.1%}",
            "Sharpe": f"{r.sharpe_ratio:.3f}",
            "Sortino": f"{r.sortino_ratio:.3f}",
            "Max DD": f"{r.max_drawdown:.1%}",
            "Calmar": f"{r.calmar_ratio:.3f}",
            "Win Rate": f"{r.win_rate:.1%}",
            "Profit Factor": f"{r.profit_factor:.2f}",
            "Trades": r.total_trades,
            "Costs": f"${r.total_costs:,.0f}",
        })

    df = pd.DataFrame(rows).set_index("Strategy")
    return df
