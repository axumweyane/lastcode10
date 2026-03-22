"""
Prometheus metrics for the APEX ensemble trading system.

Exposes gauges, histograms, and info metrics covering signals, strategy
weights, regime state, execution quality, pipeline duration, and risk.

Usage::

    from monitoring.metrics import PrometheusMetrics
    metrics = PrometheusMetrics()
    metrics.update_signals(signals, weights, regime, risk, duration)

Mount the ``/metrics`` endpoint via ``metrics.get_asgi_app()``.
"""

import logging
from typing import Any, Dict, List, Optional

from prometheus_client import (
    CollectorRegistry,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    make_asgi_app,
)

logger = logging.getLogger(__name__)

# Histogram bucket definitions
CONFIDENCE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SLIPPAGE_BUCKETS = (-50, -20, -10, -5, -2, 0, 2, 5, 10, 20, 50, 100, 200)
DURATION_BUCKETS = (1, 5, 10, 30, 60, 120, 300, 600)


class PrometheusMetrics:
    """
    Central metrics registry for the APEX paper trader.

    All metrics use a dedicated ``CollectorRegistry`` so they don't
    collide with the default process-level metrics from prometheus_client.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # ── Signal metrics ────────────────────────────────────────────────
        self.signal_score = Gauge(
            "apex_signal_score",
            "Current ensemble signal score per symbol",
            labelnames=["symbol", "direction"],
            registry=self.registry,
        )

        # ── Weight metrics ────────────────────────────────────────────────
        self.strategy_weight = Gauge(
            "apex_strategy_weight",
            "Strategy weight in the ensemble",
            labelnames=["strategy_name", "weight_type"],
            registry=self.registry,
        )

        # ── Regime metrics ────────────────────────────────────────────────
        self.regime_state = Info(
            "apex_regime_state",
            "Current market regime classification",
            registry=self.registry,
        )

        # ── Confidence histogram ──────────────────────────────────────────
        self.ensemble_confidence = Histogram(
            "apex_ensemble_confidence",
            "Distribution of ensemble signal confidence values",
            buckets=CONFIDENCE_BUCKETS,
            registry=self.registry,
        )

        # ── Execution slippage ────────────────────────────────────────────
        self.execution_slippage = Histogram(
            "apex_execution_slippage_bps",
            "VWAP execution slippage in basis points",
            labelnames=["symbol"],
            buckets=SLIPPAGE_BUCKETS,
            registry=self.registry,
        )

        # ── Pipeline duration ─────────────────────────────────────────────
        self.pipeline_duration = Histogram(
            "apex_pipeline_duration_seconds",
            "Duration of daily pipeline run in seconds",
            buckets=DURATION_BUCKETS,
            registry=self.registry,
        )

        # ── Risk metrics ──────────────────────────────────────────────────
        self.risk_drawdown = Gauge(
            "apex_risk_drawdown",
            "Current portfolio drawdown (fraction)",
            registry=self.registry,
        )
        self.risk_var_99 = Gauge(
            "apex_risk_var_99",
            "99th percentile Value at Risk",
            registry=self.registry,
        )
        self.risk_cvar_95 = Gauge(
            "apex_risk_cvar_95",
            "95th percentile Conditional VaR (Expected Shortfall)",
            registry=self.registry,
        )

    # ── Bulk update methods ───────────────────────────────────────────────

    def update_signals(self, signals: List[Dict[str, Any]]) -> None:
        """Update signal score gauge and confidence histogram."""
        # Clear existing signal labels to avoid stale symbols
        self.signal_score._metrics.clear()
        for sig in signals:
            symbol = sig.get("symbol", "")
            direction = sig.get("direction", "neutral")
            score = sig.get("combined_score", 0.0)
            confidence = sig.get("confidence", 0.0)
            self.signal_score.labels(symbol=symbol, direction=direction).set(score)
            self.ensemble_confidence.observe(confidence)

    def update_weights(
        self,
        fixed_weights: Dict[str, float],
        bayesian_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update strategy weight gauges."""
        self.strategy_weight._metrics.clear()
        for name, w in fixed_weights.items():
            self.strategy_weight.labels(strategy_name=name, weight_type="fixed").set(w)
        if bayesian_weights:
            for name, w in bayesian_weights.items():
                self.strategy_weight.labels(
                    strategy_name=name, weight_type="bayesian"
                ).set(w)

    def update_regime(
        self,
        regime: str,
        is_volatile: bool = False,
        is_trending: bool = False,
    ) -> None:
        """Update regime info metric."""
        self.regime_state.info(
            {
                "regime": regime,
                "volatility": "volatile" if is_volatile else "calm",
                "trend": "trending" if is_trending else "choppy",
            }
        )

    def update_risk(
        self,
        drawdown: float = 0.0,
        var_99: float = 0.0,
        cvar_95: float = 0.0,
    ) -> None:
        """Update risk gauge metrics."""
        self.risk_drawdown.set(drawdown)
        self.risk_var_99.set(var_99)
        self.risk_cvar_95.set(cvar_95)

    def observe_slippage(self, symbol: str, slippage_bps: float) -> None:
        """Record a single execution slippage observation."""
        self.execution_slippage.labels(symbol=symbol).observe(slippage_bps)

    def observe_pipeline_duration(self, seconds: float) -> None:
        """Record pipeline run duration."""
        self.pipeline_duration.observe(seconds)

    # ── Export ────────────────────────────────────────────────────────────

    def generate(self) -> bytes:
        """Generate Prometheus exposition format bytes."""
        return generate_latest(self.registry)

    def get_asgi_app(self):
        """Return an ASGI app that serves /metrics."""
        return make_asgi_app(registry=self.registry)
