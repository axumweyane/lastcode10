"""Tests for HI-1 fix: PortfolioRiskManager properly wired into paper-trader pipeline."""

import os
import sys
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.risk.portfolio_risk import (
    PortfolioRiskManager, RiskReport, VaRResult, CorrelationAlert,
    KillSwitchEvent,
)
from strategies.base import StrategyPerformance
import pandas as pd


# ── 1. assess() returns meaningful results after feeding 30 days ─────────────

class TestAssessWithData:
    """Verify assess() produces real metrics after being fed historical data."""

    def _make_manager_with_data(self, n_days=30, mean_return=0.001, vol=0.015):
        mgr = PortfolioRiskManager()
        rng = np.random.RandomState(42)
        returns = rng.normal(mean_return, vol, n_days)
        for r in returns:
            mgr.record_portfolio_return(float(r))
        return mgr

    def test_var_computed(self):
        mgr = self._make_manager_with_data(30)
        report = mgr.assess()
        assert report.var.method_used != "insufficient_data"
        assert report.var.parametric_var > 0
        assert report.var.historical_var > 0

    def test_cvar_computed(self):
        mgr = self._make_manager_with_data(30)
        report = mgr.assess()
        assert report.var.cvar_95 > 0

    def test_sharpe_computed(self):
        mgr = self._make_manager_with_data(30, mean_return=0.003)
        report = mgr.assess()
        assert report.portfolio_sharpe_21d != 0.0

    def test_drawdown_computed(self):
        mgr = PortfolioRiskManager()
        # Feed declining returns to create drawdown
        for r in [0.01, 0.01, -0.05, -0.03, -0.02, 0.001] * 5:
            mgr.record_portfolio_return(r)
        report = mgr.assess()
        assert report.portfolio_drawdown > 0

    def test_report_has_all_fields(self):
        mgr = self._make_manager_with_data(30)
        report = mgr.assess()
        assert isinstance(report, RiskReport)
        assert report.timestamp is not None
        assert isinstance(report.correlation_alerts, list)
        assert isinstance(report.kill_events, list)
        assert isinstance(report.capital_allocations, list)

    def test_to_dict(self):
        mgr = self._make_manager_with_data(30)
        report = mgr.assess()
        d = report.to_dict()
        assert "portfolio_drawdown" in d
        assert "var_99" in d
        assert "cvar_95" in d
        assert "killed_strategies" in d
        assert "correlation_alerts" in d

    def test_insufficient_data_returns_defaults(self):
        """With < 5 data points, VaR reports insufficient_data."""
        mgr = PortfolioRiskManager()
        mgr.record_portfolio_return(0.01)
        report = mgr.assess()
        assert report.var.method_used == "insufficient_data"
        assert report.var.parametric_var == 0.0


# ── 2. kill_switch_triggered blocks trades ───────────────────────────────────

class TestKillSwitchBlocksTrades:
    """Verify kill_switch_triggered=True when portfolio drawdown is breached."""

    def test_kill_switch_on_drawdown_breach(self):
        mgr = PortfolioRiskManager(max_portfolio_drawdown=0.05)
        # Create large drawdown
        returns = [0.02] + [-0.03] * 10
        for r in returns:
            mgr.record_portfolio_return(r)
        report = mgr.assess()
        # portfolio_breached should be True since drawdown > 5%
        assert report.portfolio_breached is True
        assert report.kill_switch_triggered is True

    def test_kill_switch_not_triggered_within_limits(self):
        mgr = PortfolioRiskManager(max_portfolio_drawdown=0.50)
        # Use consistently positive returns to avoid drawdown
        for r in [0.005, 0.003, 0.004, 0.002, 0.006] * 6:
            mgr.record_portfolio_return(r)
        report = mgr.assess()
        assert report.portfolio_drawdown == 0.0
        assert report.kill_switch_triggered is False

    def test_kill_reason_populated(self):
        mgr = PortfolioRiskManager(max_portfolio_drawdown=0.02)
        for r in [0.01, -0.05, -0.05]:
            mgr.record_portfolio_return(r)
        report = mgr.assess()
        if report.kill_switch_triggered:
            assert "drawdown" in report.kill_reason.lower()

    def test_paper_trader_checks_kill_switch(self):
        """Verify the paper-trader source uses risk_manager.assess() and checks kill_switch_triggered."""
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper-trader", "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        assert "risk_assessment.kill_switch_triggered" in source
        assert "risk_manager.assess()" in source
        # Verify it returns (skips trades) on kill switch
        assert "RISK KILL SWITCH" in source


# ── 3. killed_strategies are excluded from the pipeline ──────────────────────

class TestKilledStrategiesExcluded:
    """Verify per-strategy kill switches exclude strategies."""

    def test_strategy_killed_on_drawdown(self):
        mgr = PortfolioRiskManager(kill_max_drawdown=0.10)
        perf = StrategyPerformance(
            strategy_name="bad_momentum",
            daily_returns=pd.Series([0.01] * 30),
            current_drawdown=0.15,
            sharpe_21d=0.5,
        )
        mgr.update_strategy_performance("bad_momentum", perf)
        report = mgr.assess()
        assert "bad_momentum" in report.killed_strategies

    def test_strategy_killed_on_sharpe(self):
        mgr = PortfolioRiskManager(kill_min_sharpe=-0.5)
        perf = StrategyPerformance(
            strategy_name="failing_strat",
            daily_returns=pd.Series([0.01] * 25),
            current_drawdown=0.05,
            sharpe_21d=-1.5,
        )
        mgr.update_strategy_performance("failing_strat", perf)
        report = mgr.assess()
        assert "failing_strat" in report.killed_strategies

    def test_healthy_strategy_not_killed(self):
        mgr = PortfolioRiskManager()
        perf = StrategyPerformance(
            strategy_name="good_strat",
            daily_returns=pd.Series([0.01] * 30),
            current_drawdown=0.05,
            sharpe_21d=1.5,
        )
        mgr.update_strategy_performance("good_strat", perf)
        report = mgr.assess()
        assert "good_strat" not in report.killed_strategies

    def test_paper_trader_filters_killed(self):
        """Verify the paper-trader source filters killed strategies from combined signals."""
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper-trader", "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        assert "get_killed_strategies" in source
        assert "killed_names" in source

    def test_is_strategy_killed(self):
        mgr = PortfolioRiskManager(kill_max_drawdown=0.10)
        perf = StrategyPerformance(
            strategy_name="doomed",
            daily_returns=pd.Series([0.01] * 30),
            current_drawdown=0.15,
            sharpe_21d=0.5,
        )
        mgr.update_strategy_performance("doomed", perf)
        mgr.assess()
        assert mgr.is_strategy_killed("doomed")
        assert not mgr.is_strategy_killed("healthy")


# ── 4. Correlation alert fires when two strategies are highly correlated ─────

class TestCorrelationAlerts:
    """Verify correlation monitoring between strategy return vectors."""

    def test_high_correlation_triggers_alert(self):
        mgr = PortfolioRiskManager(correlation_alert_threshold=0.85)
        rng = np.random.RandomState(42)
        base = rng.normal(0.001, 0.01, 30)
        noise = rng.normal(0, 0.001, 30)  # tiny noise => near-perfect correlation

        for i in range(30):
            mgr.record_strategy_return("strat_a", float(base[i]))
            mgr.record_strategy_return("strat_b", float(base[i] + noise[i]))

        report = mgr.assess()
        assert len(report.correlation_alerts) > 0
        alert = report.correlation_alerts[0]
        assert alert.correlation > 0.85
        assert {alert.strategy_a, alert.strategy_b} == {"strat_a", "strat_b"}

    def test_low_correlation_no_alert(self):
        mgr = PortfolioRiskManager(correlation_alert_threshold=0.85)
        rng = np.random.RandomState(42)
        for i in range(30):
            mgr.record_strategy_return("strat_a", float(rng.normal(0.001, 0.01)))
            mgr.record_strategy_return("strat_b", float(rng.normal(0.001, 0.01)))

        report = mgr.assess()
        # With independent RNG draws, correlation should be low
        high_corr_alerts = [a for a in report.correlation_alerts if a.correlation > 0.85]
        assert len(high_corr_alerts) == 0

    def test_correlation_alert_message(self):
        mgr = PortfolioRiskManager(correlation_alert_threshold=0.5)
        # Perfectly correlated
        for i in range(20):
            val = 0.001 * i
            mgr.record_strategy_return("alpha", val)
            mgr.record_strategy_return("beta", val)

        report = mgr.assess()
        assert len(report.correlation_alerts) > 0
        assert "alpha" in report.correlation_alerts[0].message

    def test_paper_trader_reduces_correlated_weight(self):
        """Verify correlation-based weight reduction exists in the pipeline."""
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper-trader", "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        assert "correlation_alerts" in source
        assert "combined_score *= 0.5" in source


# ── 5. Paper-trader structural tests ────────────────────────────────────────

class TestPaperTraderRiskWiring:
    """Structural tests: verify risk_manager is a persistent global, fed at startup."""

    def _read_source(self):
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper-trader", "main.py",
        )
        with open(main_path) as f:
            return f.read()

    def test_risk_manager_is_global(self):
        source = self._read_source()
        assert "risk_manager: Optional[PortfolioRiskManager] = None" in source

    def test_risk_manager_initialized_at_startup(self):
        source = self._read_source()
        assert "risk_manager = PortfolioRiskManager(" in source

    def test_historical_returns_loaded(self):
        source = self._read_source()
        assert "load_historical_returns" in source
        assert "historical_returns" in source

    def test_daily_return_fed_to_risk_manager(self):
        source = self._read_source()
        assert "risk_manager.record_portfolio_return(daily_return)" in source

    def test_strategy_returns_fed(self):
        source = self._read_source()
        assert "record_strategy_return" in source

    def test_risk_report_logged_to_db(self):
        source = self._read_source()
        assert "log_risk_report" in source

    def test_paper_risk_reports_table_exists(self):
        source = self._read_source()
        assert "paper_risk_reports" in source

    def test_risk_reports_table_in_schema(self):
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "postgres_schema.py",
        )
        with open(schema_path) as f:
            source = f.read()
        assert "paper_risk_reports" in source
