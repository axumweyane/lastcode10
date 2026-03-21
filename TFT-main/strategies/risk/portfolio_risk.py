"""
Portfolio-level risk manager for the APEX multi-strategy system.

Responsibilities:
    1. VaR calculation (parametric + historical)
    2. Strategy correlation monitoring (alert when strategies converge)
    3. Per-strategy kill switches (halt individual strategies on drawdown/Sharpe)
    4. Dynamic capital allocation (shift capital toward performing strategies)
    5. Portfolio-level drawdown monitoring

This sits above the circuit breaker (which handles account-level emergencies).
The risk manager handles strategy-level health and portfolio composition risk.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from strategies.base import StrategyPerformance

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    parametric_var: float      # from normal distribution assumption
    historical_var: float      # from actual return distribution
    cvar_95: float             # Conditional VaR (Expected Shortfall) at 95%
    confidence: float          # e.g. 0.99
    horizon_days: int          # e.g. 1
    portfolio_vol: float       # annualized
    worst_case_daily: float    # worst observed daily return
    method_used: str           # which VaR was binding


@dataclass
class CorrelationAlert:
    """Alert when strategy returns become too correlated."""
    strategy_a: str
    strategy_b: str
    correlation: float
    threshold: float
    window_days: int
    message: str


@dataclass
class KillSwitchEvent:
    """Record of a per-strategy kill switch trigger."""
    strategy_name: str
    reason: str
    drawdown: float
    sharpe_21d: float
    timestamp: datetime


@dataclass
class CapitalAllocation:
    """Target capital allocation per strategy."""
    strategy_name: str
    target_fraction: float     # fraction of total capital
    current_fraction: float    # current allocation
    adjustment: float          # change from current
    rationale: str


@dataclass
class RiskReport:
    """Complete risk assessment snapshot."""
    timestamp: datetime
    var: VaRResult
    correlation_alerts: List[CorrelationAlert]
    kill_events: List[KillSwitchEvent]
    capital_allocations: List[CapitalAllocation]
    portfolio_drawdown: float
    portfolio_sharpe_21d: float
    portfolio_sharpe_63d: float
    total_strategy_count: int
    active_strategy_count: int
    killed_strategy_count: int


class PortfolioRiskManager:
    """
    Monitors portfolio-level risk across all strategies.

    Usage:
        risk_mgr = PortfolioRiskManager(max_drawdown=0.20, var_confidence=0.99)
        risk_mgr.update_strategy_performance("momentum", perf)
        risk_mgr.record_portfolio_return(daily_return)
        report = risk_mgr.assess()
    """

    def __init__(
        self,
        max_portfolio_drawdown: float = 0.20,
        var_confidence: float = 0.99,
        correlation_alert_threshold: float = 0.6,
        kill_max_drawdown: float = 0.20,
        kill_min_sharpe: float = -1.0,
    ):
        self.max_portfolio_drawdown = max_portfolio_drawdown
        self.var_confidence = var_confidence
        self.correlation_alert_threshold = correlation_alert_threshold
        self.kill_max_drawdown = kill_max_drawdown
        self.kill_min_sharpe = kill_min_sharpe

        self._strategy_performances: Dict[str, StrategyPerformance] = {}
        self._strategy_daily_returns: Dict[str, List[float]] = {}
        self._portfolio_returns: List[float] = []
        self._killed_strategies: Dict[str, KillSwitchEvent] = {}
        self._kill_history: List[KillSwitchEvent] = []

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def update_strategy_performance(
        self, strategy_name: str, perf: StrategyPerformance,
    ) -> None:
        """Update stored performance for a strategy."""
        self._strategy_performances[strategy_name] = perf

    def record_strategy_return(
        self, strategy_name: str, daily_return: float,
    ) -> None:
        """Record a single daily return for a strategy."""
        if strategy_name not in self._strategy_daily_returns:
            self._strategy_daily_returns[strategy_name] = []
        self._strategy_daily_returns[strategy_name].append(daily_return)

    def record_portfolio_return(self, daily_return: float) -> None:
        """Record a portfolio-level daily return."""
        self._portfolio_returns.append(daily_return)

    # ------------------------------------------------------------------
    # Risk assessment
    # ------------------------------------------------------------------

    def assess(self) -> RiskReport:
        """Run full risk assessment and return report."""
        var = self._compute_var()
        corr_alerts = self._check_correlations()
        kill_events = self._check_kill_switches()
        allocations = self._compute_capital_allocation()
        dd = self._portfolio_drawdown()
        sharpe_21, sharpe_63 = self._portfolio_sharpe()

        active = len(self._strategy_performances) - len(self._killed_strategies)

        report = RiskReport(
            timestamp=datetime.now(timezone.utc),
            var=var,
            correlation_alerts=corr_alerts,
            kill_events=kill_events,
            capital_allocations=allocations,
            portfolio_drawdown=dd,
            portfolio_sharpe_21d=sharpe_21,
            portfolio_sharpe_63d=sharpe_63,
            total_strategy_count=len(self._strategy_performances),
            active_strategy_count=active,
            killed_strategy_count=len(self._killed_strategies),
        )

        # Log summary
        logger.info(
            "Risk report: DD=%.1f%%, VaR99=%.2f%%, CVaR95=%.2f%%, Sharpe21=%.2f, "
            "%d active / %d killed, %d correlation alerts",
            dd * 100, var.parametric_var * 100, var.cvar_95 * 100, sharpe_21,
            active, len(self._killed_strategies), len(corr_alerts),
        )

        return report

    # ------------------------------------------------------------------
    # VaR
    # ------------------------------------------------------------------

    def _compute_var(self) -> VaRResult:
        """
        Compute Value at Risk using both parametric and historical methods.

        Parametric: assumes normal, VaR = z * daily_vol
        Historical: takes the (1-confidence) percentile of actual returns

        The binding (worse) of the two is reported as the active VaR.
        """
        returns = np.array(self._portfolio_returns)

        if len(returns) < 5:
            return VaRResult(
                parametric_var=0.0, historical_var=0.0, cvar_95=0.0,
                confidence=self.var_confidence, horizon_days=1,
                portfolio_vol=0.0, worst_case_daily=0.0,
                method_used="insufficient_data",
            )

        daily_vol = float(np.std(returns))
        ann_vol = daily_vol * np.sqrt(252)

        # Parametric VaR (normal)
        z = scipy_stats.norm.ppf(self.var_confidence)
        parametric_var = z * daily_vol

        # Historical VaR
        percentile = (1 - self.var_confidence) * 100
        historical_var = float(-np.percentile(returns, percentile))

        # CVaR-95 (Expected Shortfall): mean of the worst 5% of returns
        sorted_returns = np.sort(returns)
        cutoff_index = max(1, int(np.ceil(len(sorted_returns) * 0.05)))
        tail_returns = sorted_returns[:cutoff_index]
        cvar_95 = float(-np.mean(tail_returns))

        worst = float(-np.min(returns)) if len(returns) > 0 else 0.0
        binding = max(parametric_var, historical_var)
        method = "historical" if historical_var >= parametric_var else "parametric"

        return VaRResult(
            parametric_var=parametric_var,
            historical_var=historical_var,
            cvar_95=cvar_95,
            confidence=self.var_confidence,
            horizon_days=1,
            portfolio_vol=ann_vol,
            worst_case_daily=worst,
            method_used=method,
        )

    # ------------------------------------------------------------------
    # Correlation monitoring
    # ------------------------------------------------------------------

    def _check_correlations(self, window: int = 63) -> List[CorrelationAlert]:
        """
        Check pairwise correlation between strategy returns.

        When two strategies become highly correlated (>threshold), the
        diversification benefit disappears and the ensemble is exposed to
        concentrated risk.
        """
        alerts: List[CorrelationAlert] = []
        strat_names = list(self._strategy_daily_returns.keys())

        if len(strat_names) < 2:
            return alerts

        for i, name_a in enumerate(strat_names):
            for name_b in strat_names[i + 1:]:
                ret_a = self._strategy_daily_returns[name_a]
                ret_b = self._strategy_daily_returns[name_b]

                # Use the most recent `window` days
                min_len = min(len(ret_a), len(ret_b), window)
                if min_len < 10:
                    continue

                a = np.array(ret_a[-min_len:])
                b = np.array(ret_b[-min_len:])

                corr = float(np.corrcoef(a, b)[0, 1])

                if abs(corr) > self.correlation_alert_threshold:
                    alerts.append(CorrelationAlert(
                        strategy_a=name_a,
                        strategy_b=name_b,
                        correlation=corr,
                        threshold=self.correlation_alert_threshold,
                        window_days=min_len,
                        message=(
                            f"Strategies {name_a} and {name_b} have {min_len}-day "
                            f"correlation {corr:.3f} (threshold: "
                            f"{self.correlation_alert_threshold}). "
                            f"Diversification benefit reduced."
                        ),
                    ))
                    logger.warning(
                        "CORRELATION ALERT: %s / %s = %.3f",
                        name_a, name_b, corr,
                    )

        return alerts

    # ------------------------------------------------------------------
    # Kill switches
    # ------------------------------------------------------------------

    def _check_kill_switches(self) -> List[KillSwitchEvent]:
        """
        Check per-strategy kill conditions.

        A killed strategy is excluded from the ensemble until manually
        re-enabled. Kill conditions:
            1. Strategy drawdown exceeds limit
            2. 21-day Sharpe falls below floor
        """
        new_kills: List[KillSwitchEvent] = []

        for name, perf in self._strategy_performances.items():
            if name in self._killed_strategies:
                continue  # already killed

            reason = None

            if perf.current_drawdown >= self.kill_max_drawdown:
                reason = (
                    f"Drawdown {perf.current_drawdown:.1%} >= "
                    f"limit {self.kill_max_drawdown:.1%}"
                )

            elif (len(perf.daily_returns) >= 21
                  and perf.sharpe_21d < self.kill_min_sharpe):
                reason = (
                    f"21d Sharpe {perf.sharpe_21d:.2f} < "
                    f"floor {self.kill_min_sharpe:.2f}"
                )

            if reason:
                event = KillSwitchEvent(
                    strategy_name=name,
                    reason=reason,
                    drawdown=perf.current_drawdown,
                    sharpe_21d=perf.sharpe_21d,
                    timestamp=datetime.now(timezone.utc),
                )
                self._killed_strategies[name] = event
                self._kill_history.append(event)
                new_kills.append(event)
                logger.warning("KILL SWITCH: %s — %s", name, reason)

        return new_kills

    def is_strategy_killed(self, strategy_name: str) -> bool:
        """Check if a strategy has been killed."""
        return strategy_name in self._killed_strategies

    def revive_strategy(self, strategy_name: str, operator: str) -> bool:
        """Manually re-enable a killed strategy."""
        if strategy_name in self._killed_strategies:
            logger.info(
                "Strategy %s revived by %s", strategy_name, operator,
            )
            del self._killed_strategies[strategy_name]
            return True
        return False

    def get_killed_strategies(self) -> Dict[str, KillSwitchEvent]:
        """Return all currently killed strategies."""
        return dict(self._killed_strategies)

    def get_kill_history(self) -> List[KillSwitchEvent]:
        """Return full kill switch history."""
        return list(self._kill_history)

    # ------------------------------------------------------------------
    # Dynamic capital allocation
    # ------------------------------------------------------------------

    def _compute_capital_allocation(self) -> List[CapitalAllocation]:
        """
        Allocate capital across strategies based on recent performance.

        Method: risk-parity inspired — each strategy gets capital inversely
        proportional to its realized volatility, with a bonus for higher
        Sharpe. Killed strategies get 0.

        allocation_i ∝ max(sharpe_63d_i, 0.1) / vol_i

        Clamped to [5%, 50%] per strategy, then normalized to sum to 100%.
        """
        active_strats: Dict[str, Tuple[float, float]] = {}  # name -> (sharpe, vol)

        for name, perf in self._strategy_performances.items():
            if name in self._killed_strategies:
                continue

            sharpe = max(perf.sharpe_63d, 0.1)  # floor at 0.1

            # Estimate vol from daily returns
            if len(perf.daily_returns) >= 10:
                vol = float(perf.daily_returns.tail(63).std() * np.sqrt(252))
                vol = max(vol, 0.05)  # floor at 5%
            else:
                vol = 0.20  # default

            active_strats[name] = (sharpe, vol)

        if not active_strats:
            return []

        # Raw allocation: sharpe / vol (risk-adjusted return per unit of risk)
        raw: Dict[str, float] = {}
        for name, (sharpe, vol) in active_strats.items():
            raw[name] = sharpe / vol

        total_raw = sum(raw.values())
        if total_raw <= 0:
            total_raw = 1.0

        # Normalize, clamp, re-normalize
        allocations: Dict[str, float] = {}
        for name, r in raw.items():
            frac = r / total_raw
            frac = max(0.05, min(0.50, frac))
            allocations[name] = frac

        total_alloc = sum(allocations.values())
        allocations = {n: a / total_alloc for n, a in allocations.items()}

        # Build results
        results: List[CapitalAllocation] = []
        for name, target_frac in allocations.items():
            # Current fraction (equal weight as default)
            n_active = len(active_strats)
            current = 1.0 / n_active if n_active > 0 else 0.0

            sharpe, vol = active_strats[name]
            results.append(CapitalAllocation(
                strategy_name=name,
                target_fraction=target_frac,
                current_fraction=current,
                adjustment=target_frac - current,
                rationale=(
                    f"Sharpe63d={sharpe:.2f}, Vol={vol:.1%}, "
                    f"risk-adj={sharpe/vol:.2f}"
                ),
            ))

        # Add killed strategies with 0 allocation
        for name, event in self._killed_strategies.items():
            results.append(CapitalAllocation(
                strategy_name=name,
                target_fraction=0.0,
                current_fraction=0.0,
                adjustment=0.0,
                rationale=f"KILLED: {event.reason}",
            ))

        results.sort(key=lambda a: a.target_fraction, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Portfolio-level metrics
    # ------------------------------------------------------------------

    def _portfolio_drawdown(self) -> float:
        """Current portfolio drawdown from peak."""
        if not self._portfolio_returns:
            return 0.0

        cumulative = np.cumsum(self._portfolio_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = peak - cumulative

        if peak[-1] > 0:
            return float(drawdowns[-1] / peak[-1])
        return 0.0

    def _portfolio_sharpe(self) -> Tuple[float, float]:
        """Return (21-day Sharpe, 63-day Sharpe)."""
        returns = pd.Series(self._portfolio_returns)

        sharpe_21 = 0.0
        sharpe_63 = 0.0

        if len(returns) >= 10:
            r21 = returns.tail(21)
            if r21.std() > 0:
                sharpe_21 = float((r21.mean() / r21.std()) * np.sqrt(252))

        if len(returns) >= 21:
            r63 = returns.tail(63)
            if r63.std() > 0:
                sharpe_63 = float((r63.mean() / r63.std()) * np.sqrt(252))

        return sharpe_21, sharpe_63

    def is_portfolio_breached(self) -> bool:
        """Check if portfolio-level drawdown exceeds limit."""
        return self._portfolio_drawdown() >= self.max_portfolio_drawdown
