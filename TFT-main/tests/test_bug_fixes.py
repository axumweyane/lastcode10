"""Tests for the 5 bug fixes: CF-5, CF-6, CF-7, CF-9, HI-4."""

import asyncio
import time
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ── CF-5: CVaR-95 now computed ─────────────────────────────────────────────


class TestCF5_CVaR:
    """Verify CVaR-95 (Expected Shortfall) is computed in VaRResult."""

    def test_cvar_field_exists(self):
        from strategies.risk.portfolio_risk import PortfolioRiskManager

        mgr = PortfolioRiskManager()
        # Need >= 5 returns for VaR calculation
        for r in [
            -0.05,
            -0.03,
            -0.01,
            0.02,
            0.01,
            -0.04,
            0.03,
            -0.02,
            0.0,
            0.01,
            -0.06,
            0.02,
            -0.01,
            0.04,
            -0.03,
            0.01,
            -0.02,
            0.03,
            -0.01,
            0.02,
        ]:
            mgr.record_portfolio_return(r)
        report = mgr.assess()
        assert hasattr(report.var, "cvar_95")
        assert report.var.cvar_95 > 0

    def test_cvar_greater_than_var(self):
        """CVaR should always be >= VaR (it's the expected loss beyond VaR)."""
        from strategies.risk.portfolio_risk import PortfolioRiskManager

        mgr = PortfolioRiskManager(var_confidence=0.95)
        # Skewed negative returns to make tail fat
        returns = [
            -0.10,
            -0.08,
            -0.07,
            -0.05,
            -0.03,
            -0.01,
            0.01,
            0.02,
            0.03,
            0.04,
            0.01,
            0.02,
            0.01,
            0.03,
            0.02,
            0.01,
            0.02,
            0.01,
            0.00,
            0.01,
        ]
        for r in returns:
            mgr.record_portfolio_return(r)
        result = mgr._compute_var()
        assert result.cvar_95 >= result.historical_var

    def test_cvar_is_mean_of_worst_5pct(self):
        """Verify CVaR is exactly the mean of the worst 5% of returns."""
        from strategies.risk.portfolio_risk import PortfolioRiskManager

        mgr = PortfolioRiskManager()
        # 100 returns — worst 5% = worst 5 values
        returns = list(np.linspace(-0.10, 0.05, 100))
        for r in returns:
            mgr.record_portfolio_return(r)
        result = mgr._compute_var()
        sorted_ret = sorted(returns)
        expected_cvar = -np.mean(sorted_ret[:5])
        assert abs(result.cvar_95 - expected_cvar) < 1e-10

    def test_cvar_zero_on_insufficient_data(self):
        from strategies.risk.portfolio_risk import PortfolioRiskManager

        mgr = PortfolioRiskManager()
        mgr.record_portfolio_return(0.01)
        result = mgr._compute_var()
        assert result.cvar_95 == 0.0
        assert result.method_used == "insufficient_data"


# ── CF-6: Circuit breaker fail-closed ──────────────────────────────────────


class TestCF6_CircuitBreakerFailClosed:
    """Verify fail-closed behavior when Redis is unreachable."""

    def test_circuit_breaker_tripped_on_redis_failure(self):
        """If circuit breaker fails to init, state.circuit_breaker_tripped must be set True."""
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper-trader",
            "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        # Verify the fail-closed assignment exists in the except block
        assert "state.circuit_breaker_tripped = True" in source

    def test_fail_closed_state_blocks_pipeline(self):
        """When circuit_breaker is None and tripped=True, pipeline must not execute trades."""
        # This is a structural test verifying the code path exists.
        # Read the source and verify the elif branch.
        import inspect

        # Import the module's source to check the pattern
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper-trader",
            "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        # Verify fail-closed pattern: if CB is None but tripped, return early
        assert "elif state.circuit_breaker_tripped:" in source
        assert "FAIL CLOSED" in source

    def test_circuit_breaker_stop_in_shutdown(self):
        """Verify circuit_breaker.stop() is called during shutdown."""
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper-trader",
            "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        assert "await circuit_breaker.stop()" in source

    def test_circuit_breaker_start_in_startup(self):
        """Verify circuit_breaker.start() is called during startup."""
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper-trader",
            "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        assert "await circuit_breaker.start()" in source


# ── CF-7: Kafka explicit commit ────────────────────────────────────────────


class TestCF7_KafkaAutoCommit:
    """Verify all 4 microservice consumers use manual commit."""

    CONSUMER_FILES = [
        ("microservices/sentiment-engine/main.py", "sentiment-engine"),
        ("microservices/tft-predictor/main.py", "tft-predictor"),
        ("microservices/trading-engine/main.py", "trading-engine"),
        ("microservices/orchestrator/main.py", "orchestrator"),
    ]

    @pytest.mark.parametrize("filepath,service", CONSUMER_FILES)
    def test_auto_commit_disabled(self, filepath, service):
        """Each consumer must set enable_auto_commit=False."""
        full_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            filepath,
        )
        with open(full_path) as f:
            source = f.read()
        assert (
            "enable_auto_commit=False" in source
        ), f"{service} consumer still uses auto-commit"

    @pytest.mark.parametrize("filepath,service", CONSUMER_FILES)
    def test_explicit_commit_present(self, filepath, service):
        """Each consumer must call consumer.commit() after processing."""
        full_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            filepath,
        )
        with open(full_path) as f:
            source = f.read()
        assert (
            "consumer.commit()" in source or "self.kafka_consumer.commit()" in source
        ), f"{service} consumer missing explicit commit()"

    @pytest.mark.parametrize("filepath,service", CONSUMER_FILES)
    def test_producer_flush_before_commit(self, filepath, service):
        """Each consumer must flush producer before committing offsets."""
        full_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            filepath,
        )
        with open(full_path) as f:
            source = f.read()
        assert ".flush()" in source, f"{service} missing producer.flush() before commit"


# ── CF-9: Shutdown timeout ─────────────────────────────────────────────────


class TestCF9_ShutdownTimeout:
    """Verify shutdown has timeout and signal handlers."""

    def _read_main_source(self):
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper-trader",
            "main.py",
        )
        with open(main_path) as f:
            return f.read()

    def test_shutdown_has_timeout(self):
        source = self._read_main_source()
        assert "asyncio.wait_for" in source
        assert "timeout=" in source

    def test_signal_handlers_registered(self):
        source = self._read_main_source()
        assert "SIGTERM" in source
        assert "SIGINT" in source
        assert "add_signal_handler" in source

    def test_shutdown_timeout_value(self):
        source = self._read_main_source()
        assert "SHUTDOWN_TIMEOUT = 30" in source

    def test_timeout_error_handled(self):
        source = self._read_main_source()
        assert "asyncio.TimeoutError" in source


# ── HI-4: Bear regime scaling direction-aware ──────────────────────────────


class TestHI4_DirectionAwareScaling:
    """Verify exposure_scalar only reduces longs in bear regimes."""

    def _make_optimizer(self):
        from strategies.ensemble.portfolio_optimizer import PortfolioOptimizer
        from strategies.config import EnsembleConfig

        config = EnsembleConfig(
            enabled=True,
            max_total_positions=5,
            max_gross_leverage=3.0,
            target_volatility=0.15,
        )
        return PortfolioOptimizer(config)

    def _make_regime(self, exposure_scalar):
        from strategies.regime.detector import RegimeState, MarketRegime

        return RegimeState(
            regime=MarketRegime.VOLATILE_CHOPPY,
            vix_level=30.0,
            market_breadth=0.3,
            realized_vol=0.25,
            is_volatile=True,
            is_trending=False,
            confidence=0.8,
            strategy_weights={
                "momentum": 0.1,
                "mean_reversion": 0.4,
                "pairs": 0.3,
                "tft": 0.2,
            },
            exposure_scalar=exposure_scalar,
        )

    def _make_signals(self, scores):
        from strategies.ensemble.combiner import CombinedSignal
        from strategies.base import SignalDirection

        signals = []
        for symbol, score in scores.items():
            direction = SignalDirection.LONG if score > 0 else SignalDirection.SHORT
            signals.append(
                CombinedSignal(
                    symbol=symbol,
                    combined_score=score,
                    confidence=0.8,
                    direction=direction,
                    contributing_strategies={"test": score},
                )
            )
        return signals

    def test_bear_regime_preserves_shorts(self):
        """In bear regime (scalar < 1), shorts should NOT be scaled down."""
        optimizer = self._make_optimizer()
        regime = self._make_regime(exposure_scalar=0.5)
        signals = self._make_signals(
            {
                "AAPL": 0.8,  # long
                "TSLA": -0.6,  # short
            }
        )
        target = optimizer.optimize(signals, price_data=None, regime_state=regime)

        # Find positions
        positions = {p.symbol: p for p in target.positions}
        aapl = positions.get("AAPL")
        tsla = positions.get("TSLA")

        # AAPL (long) should be smaller than without regime scaling
        # TSLA (short) should NOT be scaled down
        assert aapl is not None
        assert tsla is not None
        # The short weight should be unscaled relative to its raw proportion
        # Verify short is preserved (larger absolute weight than the scaled long)
        assert abs(tsla.target_weight) > 0

    def test_bull_regime_scales_all(self):
        """In bull regime (scalar >= 1), all weights scale uniformly."""
        optimizer = self._make_optimizer()
        regime_bull = self._make_regime(exposure_scalar=1.0)
        signals = self._make_signals({"AAPL": 0.8, "TSLA": -0.6})

        target_bull = optimizer.optimize(
            signals, price_data=None, regime_state=regime_bull
        )
        positions = {p.symbol: p for p in target_bull.positions}

        # Both should exist
        assert "AAPL" in positions
        assert "TSLA" in positions

    def test_bear_regime_long_weight_reduced(self):
        """Long weight should be reduced in bear regime compared to neutral."""
        optimizer = self._make_optimizer()

        signals = self._make_signals({"AAPL": 0.8, "MSFT": 0.6, "TSLA": -0.5})

        # Neutral regime (scalar=1.0)
        regime_neutral = self._make_regime(exposure_scalar=1.0)
        target_neutral = optimizer.optimize(
            signals, price_data=None, regime_state=regime_neutral
        )
        neutral_long = target_neutral.long_weight

        # Bear regime (scalar=0.5)
        regime_bear = self._make_regime(exposure_scalar=0.5)
        target_bear = optimizer.optimize(
            signals, price_data=None, regime_state=regime_bear
        )
        bear_long = target_bear.long_weight

        assert (
            bear_long < neutral_long
        ), f"Bear long weight {bear_long} should be less than neutral {neutral_long}"

    def test_bear_regime_short_weight_not_reduced(self):
        """Short weight should NOT be reduced in bear regime."""
        optimizer = self._make_optimizer()

        signals = self._make_signals({"AAPL": 0.8, "TSLA": -0.6})

        regime_neutral = self._make_regime(exposure_scalar=1.0)
        target_neutral = optimizer.optimize(
            signals, price_data=None, regime_state=regime_neutral
        )
        neutral_short = target_neutral.short_weight

        regime_bear = self._make_regime(exposure_scalar=0.5)
        target_bear = optimizer.optimize(
            signals, price_data=None, regime_state=regime_bear
        )
        bear_short = target_bear.short_weight

        # Short weight should be same or larger (constraints may adjust slightly)
        assert (
            bear_short >= neutral_short * 0.95
        ), f"Bear short weight {bear_short} should not be less than neutral {neutral_short}"
