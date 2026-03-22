"""
Options-specific risk manager.

Responsibilities:
  1. Portfolio Greeks aggregation and limits (net delta, gamma, vega caps)
  2. Max loss calculation across all options positions
  3. Margin requirement estimation
  4. Per-strategy kill switches for options strategies
  5. VIX term structure analysis for regime integration
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.base import StrategyPerformance
from strategies.options.infrastructure.greeks import (
    GreeksCalculator,
    PortfolioGreeks,
    PositionGreeks,
)

logger = logging.getLogger(__name__)


@dataclass
class OptionsRiskLimits:
    """Configurable risk limits for the options portfolio."""

    max_net_delta_pct: float = 0.30  # net delta < 30% of portfolio value
    max_gross_delta_pct: float = 1.50  # gross delta < 150% of portfolio
    max_net_vega_pct: float = 0.05  # vega < 5% of portfolio per 1% vol move
    max_net_theta_daily_pct: float = 0.005  # daily theta < 0.5% of portfolio
    max_total_premium_pct: float = 0.10  # max 10% of portfolio in option premium
    max_single_position_pct: float = 0.03  # max 3% per single option position
    max_loss_pct: float = 0.20  # max theoretical loss < 20%
    max_positions: int = 30


@dataclass
class OptionsKillEvent:
    strategy_name: str
    reason: str
    timestamp: datetime
    greeks_snapshot: Optional[Dict] = None


@dataclass
class VIXTermStructure:
    """VIX contango/backwardation analysis for regime timing."""

    vix_spot: float
    vix_1m: float
    vix_3m: float
    is_contango: bool  # normal: longer-dated > spot
    is_backwardation: bool  # fear: spot > longer-dated
    contango_ratio: float  # vix_3m / vix_spot (>1 = contango)
    term_slope: float  # annualized slope

    @property
    def regime_signal(self) -> str:
        """Map term structure to options regime signal."""
        if self.is_backwardation and self.vix_spot > 25:
            return "crisis"  # buy puts, don't sell premium
        elif self.is_backwardation:
            return "fear"  # reduce premium selling, widen condor wings
        elif self.contango_ratio > 1.10:
            return "complacent"  # sell premium aggressively
        else:
            return "normal"


@dataclass
class OptionsRiskReport:
    timestamp: datetime
    portfolio_greeks: PortfolioGreeks
    breaches: List[str]
    kill_events: List[OptionsKillEvent]
    margin_usage_pct: float
    vix_term: Optional[VIXTermStructure]
    stress_test: Dict
    is_within_limits: bool


class OptionsRiskManager:
    """
    Monitors all options positions and enforces risk limits.

    Usage:
        risk_mgr = OptionsRiskManager(limits, greeks_calc)
        risk_mgr.update_positions(positions)
        report = risk_mgr.assess(portfolio_value=100000)
    """

    def __init__(
        self,
        limits: Optional[OptionsRiskLimits] = None,
        greeks_calc: Optional[GreeksCalculator] = None,
    ):
        self.limits = limits or OptionsRiskLimits()
        self.greeks_calc = greeks_calc or GreeksCalculator()
        self._positions: List[PositionGreeks] = []
        self._strategy_performances: Dict[str, StrategyPerformance] = {}
        self._killed_strategies: Dict[str, OptionsKillEvent] = {}
        self._kill_history: List[OptionsKillEvent] = []

    def update_positions(self, positions: List[PositionGreeks]) -> None:
        self._positions = positions

    def update_strategy_performance(
        self,
        name: str,
        perf: StrategyPerformance,
    ) -> None:
        self._strategy_performances[name] = perf

    def assess(
        self,
        portfolio_value: float,
        vix_data: Optional[Dict] = None,
    ) -> OptionsRiskReport:
        """Run full options risk assessment."""
        # Aggregate Greeks
        pf_greeks = self.greeks_calc.aggregate(self._positions)

        # Check limits
        breaches = self._check_limits(pf_greeks, portfolio_value)

        # Kill switches
        kill_events = self._check_kill_switches()

        # Margin estimation
        margin_pct = self._estimate_margin(pf_greeks, portfolio_value)

        # VIX term structure
        vix_term = self._analyze_vix_term(vix_data) if vix_data else None

        # Stress test
        stress = self.greeks_calc.stress_test(self._positions)

        report = OptionsRiskReport(
            timestamp=datetime.now(timezone.utc),
            portfolio_greeks=pf_greeks,
            breaches=breaches,
            kill_events=kill_events,
            margin_usage_pct=margin_pct,
            vix_term=vix_term,
            stress_test=stress,
            is_within_limits=len(breaches) == 0,
        )

        if breaches:
            logger.warning(
                "OPTIONS RISK BREACHES: %s",
                "; ".join(breaches),
            )

        return report

    def _check_limits(
        self,
        greeks: PortfolioGreeks,
        portfolio_value: float,
    ) -> List[str]:
        breaches = []
        if portfolio_value <= 0:
            return breaches

        # Net delta check
        net_delta_pct = (
            abs(greeks.net_delta * 100) / portfolio_value
        )  # rough conversion
        if net_delta_pct > self.limits.max_net_delta_pct:
            breaches.append(
                f"Net delta {net_delta_pct:.1%} exceeds limit {self.limits.max_net_delta_pct:.1%}"
            )

        # Max loss check
        max_loss_pct = greeks.total_max_loss / portfolio_value
        if max_loss_pct > self.limits.max_loss_pct:
            breaches.append(
                f"Max theoretical loss {max_loss_pct:.1%} exceeds limit {self.limits.max_loss_pct:.1%}"
            )

        # Position count
        if greeks.position_count > self.limits.max_positions:
            breaches.append(
                f"Position count {greeks.position_count} exceeds limit {self.limits.max_positions}"
            )

        # Net theta
        if portfolio_value > 0:
            theta_pct = abs(greeks.net_theta) / portfolio_value
            if theta_pct > self.limits.max_net_theta_daily_pct:
                breaches.append(
                    f"Daily theta {theta_pct:.3%} exceeds limit {self.limits.max_net_theta_daily_pct:.3%}"
                )

        return breaches

    def _check_kill_switches(self) -> List[OptionsKillEvent]:
        new_kills = []

        kill_thresholds = {
            "covered_calls": (0.15, -1.0),
            "iron_condors": (0.20, -1.0),
            "protective_puts": (0.25, -2.0),
            "vol_arb": (0.20, -1.0),
            "earnings_plays": (0.25, -1.5),
            "gamma_scalping": (0.20, -1.0),
        }

        for name, perf in self._strategy_performances.items():
            if name in self._killed_strategies:
                continue

            max_dd, min_sharpe = kill_thresholds.get(name, (0.20, -1.0))
            reason = None

            if perf.current_drawdown >= max_dd:
                reason = f"Drawdown {perf.current_drawdown:.1%} >= {max_dd:.1%}"
            elif len(perf.daily_returns) >= 21 and perf.sharpe_21d < min_sharpe:
                reason = f"Sharpe21 {perf.sharpe_21d:.2f} < {min_sharpe:.2f}"

            if reason:
                event = OptionsKillEvent(
                    strategy_name=name,
                    reason=reason,
                    timestamp=datetime.now(timezone.utc),
                )
                self._killed_strategies[name] = event
                self._kill_history.append(event)
                new_kills.append(event)
                logger.warning("OPTIONS KILL: %s — %s", name, reason)

        return new_kills

    def is_strategy_killed(self, name: str) -> bool:
        return name in self._killed_strategies

    def revive_strategy(self, name: str) -> bool:
        if name in self._killed_strategies:
            del self._killed_strategies[name]
            return True
        return False

    def _estimate_margin(
        self,
        greeks: PortfolioGreeks,
        portfolio_value: float,
    ) -> float:
        """Rough margin estimate: max_loss + 20% buffer."""
        if portfolio_value <= 0:
            return 0.0
        margin_req = greeks.total_max_loss * 1.2
        return min(margin_req / portfolio_value, 1.0)

    @staticmethod
    def _analyze_vix_term(vix_data: Dict) -> Optional[VIXTermStructure]:
        """
        Analyze VIX term structure from data.

        Expected vix_data keys: "vix_spot", "vix_1m", "vix_3m"
        """
        spot = vix_data.get("vix_spot", 0)
        m1 = vix_data.get("vix_1m", spot)
        m3 = vix_data.get("vix_3m", spot)

        if spot <= 0:
            return None

        contango_ratio = m3 / spot if spot > 0 else 1.0
        is_contango = m3 > spot
        is_backwardation = spot > m3
        term_slope = (m3 - spot) / spot * (12 / 3)  # annualized

        return VIXTermStructure(
            vix_spot=spot,
            vix_1m=m1,
            vix_3m=m3,
            is_contango=is_contango,
            is_backwardation=is_backwardation,
            contango_ratio=contango_ratio,
            term_slope=term_slope,
        )
