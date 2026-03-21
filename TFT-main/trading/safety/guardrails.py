"""
Safety guardrails to prevent catastrophic trading failures.

Five independent checks that can be wired into the paper-trader pipeline:

1. SignalVarianceGuard   – halt if ensemble scores collapse to a single value
2. LeverageGate         – block new orders when leverage exceeds hard limit
3. CalibrationHealthCheck – verify calibration models are fitted, not identity
4. ModelPromotionGate    – reject models whose val Sharpe is below threshold
5. ExecutionFailureMonitor – pause trading when order failure rate spikes

All thresholds are configurable via environment variables.
"""

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── 1. Signal Variance Guard ────────────────────────────────────────────────

@dataclass
class SignalVarianceResult:
    passed: bool
    std_dev: float
    num_signals: int
    message: str


class SignalVarianceGuard:
    """Halt trading when ensemble signal scores collapse to near-identical values."""

    def __init__(self, min_std: Optional[float] = None):
        self.min_std = min_std if min_std is not None else float(
            os.getenv("GUARDRAIL_SIGNAL_MIN_STD", "0.01")
        )

    def check(self, scores: List[float]) -> SignalVarianceResult:
        if len(scores) < 2:
            return SignalVarianceResult(
                passed=True, std_dev=0.0, num_signals=len(scores),
                message="Too few signals to check variance",
            )

        std = float(np.std(scores))
        passed = std > self.min_std

        if not passed:
            msg = (
                f"SIGNAL VARIANCE HALT: std={std:.6f} <= threshold {self.min_std}. "
                f"All {len(scores)} scores collapsed (possible 0.5429 incident repeat)."
            )
            logger.critical(msg)
        else:
            msg = f"Signal variance OK: std={std:.4f} ({len(scores)} signals)"
            logger.info(msg)

        return SignalVarianceResult(
            passed=passed, std_dev=std, num_signals=len(scores), message=msg,
        )


# ── 2. Leverage Gate ────────────────────────────────────────────────────────

@dataclass
class LeverageCheckResult:
    passed: bool
    current_leverage: float
    max_leverage: float
    message: str


class LeverageGate:
    """Hard-limit leverage check before every order batch."""

    def __init__(self, max_leverage: Optional[float] = None):
        self.max_leverage = max_leverage if max_leverage is not None else float(
            os.getenv("GUARDRAIL_MAX_LEVERAGE", "1.5")
        )

    def check(self, gross_leverage: float) -> LeverageCheckResult:
        passed = gross_leverage <= self.max_leverage

        if not passed:
            msg = (
                f"LEVERAGE GATE BLOCKED: gross_leverage={gross_leverage:.3f} "
                f"> max={self.max_leverage:.3f}. Skipping new orders."
            )
            logger.warning(msg)
        else:
            msg = f"Leverage OK: {gross_leverage:.3f} <= {self.max_leverage:.3f}"
            logger.info(msg)

        return LeverageCheckResult(
            passed=passed, current_leverage=gross_leverage,
            max_leverage=self.max_leverage, message=msg,
        )


# ── 3. Calibration Health Check ─────────────────────────────────────────────

@dataclass
class CalibrationCheckResult:
    passed: bool
    is_fitted: bool
    is_identity: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class CalibrationHealthCheck:
    """Verify that calibration models (Platt scaler, isotonic, etc.) are
    actually fitted and not silently passing raw scores through."""

    # A Platt scaler with these params is effectively identity (sigmoid(1*x + 0) ≈ x for small x)
    IDENTITY_A = 1.0
    IDENTITY_B = 0.0
    TOLERANCE = float(os.getenv("GUARDRAIL_CALIBRATION_TOLERANCE", "1e-6"))

    @staticmethod
    def check_platt(a: Optional[float], b: Optional[float]) -> CalibrationCheckResult:
        """Check sigmoid calibration: P = 1 / (1 + exp(A*f + B)).
        Identity when A ≈ -1, B ≈ 0  OR  when params are None (unfitted)."""
        if a is None or b is None:
            msg = "CALIBRATION ERROR: Platt scaler not fitted (params are None). Skipping calibration."
            logger.error(msg)
            return CalibrationCheckResult(
                passed=False, is_fitted=False, is_identity=False, message=msg,
            )

        tol = CalibrationHealthCheck.TOLERANCE
        # Platt: P = 1/(1+exp(A*f + B)).  Identity-like when A ≈ -1, B ≈ 0
        is_identity = abs(a - (-1.0)) < tol and abs(b - 0.0) < tol
        if is_identity:
            msg = (
                f"CALIBRATION WARNING: Platt params are identity-like (A={a}, B={b}). "
                f"Calibration will pass raw scores through unchanged."
            )
            logger.error(msg)
            return CalibrationCheckResult(
                passed=False, is_fitted=True, is_identity=True, message=msg,
                details={"A": a, "B": b},
            )

        msg = f"Calibration OK: Platt A={a:.4f}, B={b:.4f}"
        logger.info(msg)
        return CalibrationCheckResult(
            passed=True, is_fitted=True, is_identity=False, message=msg,
            details={"A": a, "B": b},
        )

    @staticmethod
    def check_generic(calibrator: Any) -> CalibrationCheckResult:
        """Check any sklearn-style calibrator that has a fitted indicator."""
        if calibrator is None:
            msg = "CALIBRATION ERROR: Calibrator is None. Skipping calibration."
            logger.error(msg)
            return CalibrationCheckResult(
                passed=False, is_fitted=False, is_identity=False, message=msg,
            )

        # sklearn convention: fitted estimators have attributes ending in _
        is_fitted = (
            hasattr(calibrator, "classes_")
            or hasattr(calibrator, "calibrators_")
            or hasattr(calibrator, "a_")
            or hasattr(calibrator, "b_")
            or hasattr(calibrator, "is_fitted") and calibrator.is_fitted
        )

        if not is_fitted:
            msg = "CALIBRATION ERROR: Calibrator appears unfitted. Skipping calibration."
            logger.error(msg)
            return CalibrationCheckResult(
                passed=False, is_fitted=False, is_identity=False, message=msg,
            )

        msg = "Calibration OK: calibrator appears fitted"
        logger.info(msg)
        return CalibrationCheckResult(
            passed=True, is_fitted=True, is_identity=False, message=msg,
        )


# ── 4. Model Promotion Sharpe Gate ──────────────────────────────────────────

@dataclass
class ModelPromotionResult:
    passed: bool
    val_sharpe: float
    min_sharpe: float
    model_name: str
    message: str


class ModelPromotionGate:
    """Reject models whose validation Sharpe is below the promotion threshold."""

    def __init__(self, min_sharpe: Optional[float] = None):
        self.min_sharpe = min_sharpe if min_sharpe is not None else float(
            os.getenv("GUARDRAIL_MIN_PROMOTION_SHARPE", "0.5")
        )

    def check(self, model_name: str, val_sharpe: float) -> ModelPromotionResult:
        passed = val_sharpe >= self.min_sharpe

        if not passed:
            msg = (
                f"MODEL REJECTED: '{model_name}' val_sharpe={val_sharpe:.3f} "
                f"< min_sharpe={self.min_sharpe:.3f}. Not promoted to live."
            )
            logger.warning(msg)
        else:
            msg = (
                f"Model promoted: '{model_name}' val_sharpe={val_sharpe:.3f} "
                f">= {self.min_sharpe:.3f}"
            )
            logger.info(msg)

        return ModelPromotionResult(
            passed=passed, val_sharpe=val_sharpe, min_sharpe=self.min_sharpe,
            model_name=model_name, message=msg,
        )


# ── 5. Execution Failure Rate Monitor ───────────────────────────────────────

@dataclass
class ExecutionHealthResult:
    passed: bool
    failure_rate: float
    max_failure_rate: float
    total_orders: int
    failed_orders: int
    message: str


class ExecutionFailureMonitor:
    """Track order success/failure ratio in a rolling window.
    Pause trading when failure rate exceeds threshold."""

    def __init__(
        self,
        max_failure_rate: Optional[float] = None,
        window_seconds: Optional[int] = None,
    ):
        self.max_failure_rate = max_failure_rate if max_failure_rate is not None else float(
            os.getenv("GUARDRAIL_MAX_EXEC_FAILURE_RATE", "0.25")
        )
        self.window_seconds = window_seconds if window_seconds is not None else int(
            os.getenv("GUARDRAIL_EXEC_WINDOW_SECONDS", "3600")
        )
        # Deque of (timestamp, success_bool)
        self._events: Deque[Tuple[float, bool]] = deque()

    def record(self, success: bool) -> None:
        """Record an order outcome."""
        self._events.append((time.monotonic(), success))
        self._prune()

    def _prune(self) -> None:
        """Remove events outside the rolling window."""
        cutoff = time.monotonic() - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def check(self) -> ExecutionHealthResult:
        """Check if current failure rate is within limits."""
        self._prune()

        total = len(self._events)
        if total == 0:
            return ExecutionHealthResult(
                passed=True, failure_rate=0.0,
                max_failure_rate=self.max_failure_rate,
                total_orders=0, failed_orders=0,
                message="No orders in window",
            )

        failed = sum(1 for _, s in self._events if not s)
        rate = failed / total

        passed = rate <= self.max_failure_rate

        if not passed:
            msg = (
                f"EXECUTION HALT: failure rate {rate:.1%} ({failed}/{total}) "
                f"> threshold {self.max_failure_rate:.0%} in last "
                f"{self.window_seconds}s window. Pausing trading."
            )
            logger.critical(msg)
        else:
            msg = (
                f"Execution health OK: failure rate {rate:.1%} "
                f"({failed}/{total}) <= {self.max_failure_rate:.0%}"
            )
            logger.info(msg)

        return ExecutionHealthResult(
            passed=passed, failure_rate=rate,
            max_failure_rate=self.max_failure_rate,
            total_orders=total, failed_orders=failed, message=msg,
        )

    def reset(self) -> None:
        """Clear all events (e.g. after alert is acknowledged)."""
        self._events.clear()
