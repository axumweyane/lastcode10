"""Tests for trading/safety/guardrails.py — all 5 safety guardrails."""

import time
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.safety.guardrails import (
    SignalVarianceGuard,
    LeverageGate,
    CalibrationHealthCheck,
    ModelPromotionGate,
    ExecutionFailureMonitor,
)

# ── 1. Signal Variance Guard ────────────────────────────────────────────────


class TestSignalVarianceGuard:
    def test_passes_with_diverse_signals(self):
        guard = SignalVarianceGuard(min_std=0.01)
        result = guard.check([0.1, 0.5, 0.9, -0.3, 0.7])
        assert result.passed is True
        assert result.std_dev > 0.01

    def test_fails_when_all_scores_identical(self):
        """Simulates the March 10 0.5429 incident."""
        guard = SignalVarianceGuard(min_std=0.01)
        result = guard.check([0.5429] * 10)
        assert result.passed is False
        assert result.std_dev < 1e-10
        assert "collapsed" in result.message.lower()

    def test_fails_when_scores_nearly_identical(self):
        guard = SignalVarianceGuard(min_std=0.01)
        result = guard.check([0.5429, 0.5430, 0.5429, 0.5428])
        assert result.passed is False
        assert result.std_dev < 0.01

    def test_passes_with_single_signal(self):
        """Single signal should pass (can't check variance)."""
        guard = SignalVarianceGuard(min_std=0.01)
        result = guard.check([0.5])
        assert result.passed is True

    def test_passes_with_empty_signals(self):
        guard = SignalVarianceGuard(min_std=0.01)
        result = guard.check([])
        assert result.passed is True

    def test_configurable_threshold(self):
        guard = SignalVarianceGuard(min_std=0.5)
        result = guard.check([0.1, 0.2, 0.3])
        assert result.passed is False  # std ~0.08 < 0.5

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("GUARDRAIL_SIGNAL_MIN_STD", "0.05")
        guard = SignalVarianceGuard()
        assert guard.min_std == 0.05


# ── 2. Leverage Gate ────────────────────────────────────────────────────────


class TestLeverageGate:
    def test_passes_under_limit(self):
        gate = LeverageGate(max_leverage=1.5)
        result = gate.check(1.2)
        assert result.passed is True

    def test_fails_over_limit(self):
        gate = LeverageGate(max_leverage=1.5)
        result = gate.check(1.8)
        assert result.passed is False
        assert result.current_leverage == 1.8

    def test_passes_at_exact_limit(self):
        gate = LeverageGate(max_leverage=1.5)
        result = gate.check(1.5)
        assert result.passed is True

    def test_passes_zero_leverage(self):
        gate = LeverageGate(max_leverage=1.5)
        result = gate.check(0.0)
        assert result.passed is True

    def test_configurable_threshold(self):
        gate = LeverageGate(max_leverage=2.0)
        result = gate.check(1.8)
        assert result.passed is True

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("GUARDRAIL_MAX_LEVERAGE", "2.5")
        gate = LeverageGate()
        assert gate.max_leverage == 2.5


# ── 3. Calibration Health Check ─────────────────────────────────────────────


class TestCalibrationHealthCheck:
    def test_platt_passes_normal_params(self):
        result = CalibrationHealthCheck.check_platt(a=-2.5, b=0.3)
        assert result.passed is True
        assert result.is_fitted is True
        assert result.is_identity is False

    def test_platt_fails_none_params(self):
        result = CalibrationHealthCheck.check_platt(a=None, b=None)
        assert result.passed is False
        assert result.is_fitted is False

    def test_platt_fails_identity_params(self):
        result = CalibrationHealthCheck.check_platt(a=-1.0, b=0.0)
        assert result.passed is False
        assert result.is_identity is True

    def test_platt_passes_near_but_not_identity(self):
        result = CalibrationHealthCheck.check_platt(a=-1.1, b=0.05)
        assert result.passed is True

    def test_generic_fails_none(self):
        result = CalibrationHealthCheck.check_generic(None)
        assert result.passed is False
        assert result.is_fitted is False

    def test_generic_fails_unfitted(self):
        class UnfittedCalibrator:
            pass

        result = CalibrationHealthCheck.check_generic(UnfittedCalibrator())
        assert result.passed is False

    def test_generic_passes_fitted(self):
        class FittedCalibrator:
            classes_ = [0, 1]

        result = CalibrationHealthCheck.check_generic(FittedCalibrator())
        assert result.passed is True


# ── 4. Model Promotion Sharpe Gate ──────────────────────────────────────────


class TestModelPromotionGate:
    def test_passes_above_threshold(self):
        gate = ModelPromotionGate(min_sharpe=0.5)
        result = gate.check("tft_v3", val_sharpe=1.2)
        assert result.passed is True
        assert result.model_name == "tft_v3"

    def test_fails_below_threshold(self):
        gate = ModelPromotionGate(min_sharpe=0.5)
        result = gate.check("tft_v3", val_sharpe=0.3)
        assert result.passed is False
        assert "REJECTED" in result.message

    def test_fails_negative_sharpe(self):
        gate = ModelPromotionGate(min_sharpe=0.5)
        result = gate.check("bad_model", val_sharpe=-0.5)
        assert result.passed is False

    def test_passes_at_exact_threshold(self):
        gate = ModelPromotionGate(min_sharpe=0.5)
        result = gate.check("border_model", val_sharpe=0.5)
        assert result.passed is True

    def test_configurable_threshold(self):
        gate = ModelPromotionGate(min_sharpe=1.0)
        result = gate.check("strict_model", val_sharpe=0.8)
        assert result.passed is False

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("GUARDRAIL_MIN_PROMOTION_SHARPE", "0.8")
        gate = ModelPromotionGate()
        assert gate.min_sharpe == 0.8


# ── 5. Execution Failure Monitor ────────────────────────────────────────────


class TestExecutionFailureMonitor:
    def test_passes_all_success(self):
        monitor = ExecutionFailureMonitor(max_failure_rate=0.25, window_seconds=3600)
        for _ in range(10):
            monitor.record(True)
        result = monitor.check()
        assert result.passed is True
        assert result.failure_rate == 0.0

    def test_fails_high_failure_rate(self):
        monitor = ExecutionFailureMonitor(max_failure_rate=0.25, window_seconds=3600)
        # 4 failures out of 10 = 40%
        for _ in range(6):
            monitor.record(True)
        for _ in range(4):
            monitor.record(False)
        result = monitor.check()
        assert result.passed is False
        assert result.failure_rate == pytest.approx(0.4)
        assert result.failed_orders == 4
        assert result.total_orders == 10

    def test_passes_just_under_threshold(self):
        monitor = ExecutionFailureMonitor(max_failure_rate=0.25, window_seconds=3600)
        # 1 failure out of 4 = 25% (at threshold, should pass)
        for _ in range(3):
            monitor.record(True)
        monitor.record(False)
        result = monitor.check()
        assert result.passed is True
        assert result.failure_rate == pytest.approx(0.25)

    def test_passes_empty_window(self):
        monitor = ExecutionFailureMonitor(max_failure_rate=0.25, window_seconds=3600)
        result = monitor.check()
        assert result.passed is True
        assert result.total_orders == 0

    def test_window_expiry(self):
        """Events outside the window should be pruned."""
        monitor = ExecutionFailureMonitor(max_failure_rate=0.25, window_seconds=1)
        # Record failures
        for _ in range(5):
            monitor.record(False)
        # Wait for window to expire
        time.sleep(1.1)
        # Record a single success
        monitor.record(True)
        result = monitor.check()
        assert result.passed is True
        assert result.total_orders == 1

    def test_reset(self):
        monitor = ExecutionFailureMonitor(max_failure_rate=0.25, window_seconds=3600)
        for _ in range(10):
            monitor.record(False)
        monitor.reset()
        result = monitor.check()
        assert result.passed is True
        assert result.total_orders == 0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("GUARDRAIL_MAX_EXEC_FAILURE_RATE", "0.10")
        monkeypatch.setenv("GUARDRAIL_EXEC_WINDOW_SECONDS", "1800")
        monitor = ExecutionFailureMonitor()
        assert monitor.max_failure_rate == 0.10
        assert monitor.window_seconds == 1800
