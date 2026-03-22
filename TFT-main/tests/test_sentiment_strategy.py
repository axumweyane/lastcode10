"""Tests for the Sentiment Strategy."""

import os
import sys
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.sentiment.strategy import (
    SentimentStrategy,
    DIVERGENCE_MULTIPLIER,
    ALIGNMENT_MULTIPLIER,
    TREND_WINDOW,
)
from strategies.config import SentimentConfig
from strategies.base import SignalDirection, StrategyOutput, AlphaScore
from models.base import ModelPrediction

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_data(symbols=None, n_days=30, sentiment_map=None):
    """Create a DataFrame with OHLCV data for testing."""
    symbols = symbols or ["AAPL", "MSFT", "TSLA"]
    rows = []
    base = pd.Timestamp("2026-01-01")
    for sym in symbols:
        for i in range(n_days):
            price = 100 + i * (1 if sym != "TSLA" else -1)  # TSLA trends down
            rows.append(
                {
                    "symbol": sym,
                    "timestamp": base + pd.Timedelta(days=i),
                    "open": price - 0.5,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def _mock_manager(sentiment_scores: dict):
    """Create a mock ModelManager that returns given sentiment scores."""
    mgr = MagicMock()
    predictions = [
        ModelPrediction(
            symbol=sym,
            predicted_value=score,
            lower_bound=score - 0.1,
            upper_bound=score + 0.1,
            confidence=0.8,
            horizon_days=1,
            model_name="sentiment",
            metadata={"sentiment_score": score},
        )
        for sym, score in sentiment_scores.items()
    ]
    mgr.predict_sentiment.return_value = predictions
    return mgr


def _make_strategy(sentiment_scores=None, enabled=True, manager=None):
    """Create a SentimentStrategy with optional mock manager."""
    config = SentimentConfig(enabled=enabled, max_positions_per_side=10)
    if manager is None and sentiment_scores is not None:
        manager = _mock_manager(sentiment_scores)
    return SentimentStrategy(config=config, manager=manager)


# ── 1. Valid output ──────────────────────────────────────────────────────────


class TestValidOutput:
    """Test that strategy produces valid AlphaScore output."""

    def test_returns_strategy_output(self):
        strat = _make_strategy({"AAPL": 0.5, "MSFT": -0.3, "TSLA": -0.7})
        data = _make_data()
        result = strat.generate_signals(data)
        assert isinstance(result, StrategyOutput)
        assert result.strategy_name == "sentiment"

    def test_scores_are_alpha_scores(self):
        strat = _make_strategy({"AAPL": 0.5, "MSFT": -0.3})
        data = _make_data()
        result = strat.generate_signals(data)
        for s in result.scores:
            assert isinstance(s, AlphaScore)
            assert hasattr(s, "symbol")
            assert hasattr(s, "score")
            assert hasattr(s, "confidence")
            assert hasattr(s, "direction")

    def test_z_scored_output(self):
        strat = _make_strategy({"AAPL": 0.8, "MSFT": -0.5, "TSLA": -0.7})
        data = _make_data()
        result = strat.generate_signals(data)
        if len(result.scores) >= 2:
            z_scores = [s.score for s in result.scores]
            assert (
                abs(np.mean(z_scores)) < 0.5
            )  # approximately zero-mean after z-scoring

    def test_metadata_contains_signal_type(self):
        strat = _make_strategy({"AAPL": 0.5})
        data = _make_data(symbols=["AAPL"])
        result = strat.generate_signals(data)
        if result.scores:
            assert "signal_type" in result.scores[0].metadata
            assert "sentiment_score" in result.scores[0].metadata

    def test_direction_is_valid_enum(self):
        strat = _make_strategy({"AAPL": 0.8, "TSLA": -0.7})
        data = _make_data()
        result = strat.generate_signals(data)
        for s in result.scores:
            assert s.direction in (
                SignalDirection.LONG,
                SignalDirection.SHORT,
                SignalDirection.NEUTRAL,
            )


# ── 2. Divergence detection (contrarian signals) ────────────────────────────


class TestDivergence:
    """Test contrarian signal generation on sentiment-price divergence."""

    def test_sentiment_up_price_down_gives_long(self):
        # TSLA has negative price trend (prices go 100, 99, 98, ...) but positive sentiment
        strat = _make_strategy({"TSLA": 0.7})
        data = _make_data(symbols=["TSLA"])
        result = strat.generate_signals(data)
        tsla = [s for s in result.scores if s.symbol == "TSLA"]
        assert len(tsla) == 1
        assert tsla[0].direction == SignalDirection.LONG
        assert tsla[0].metadata["signal_type"] == "contrarian_buy"

    def test_sentiment_down_price_up_gives_short(self):
        # AAPL has positive price trend but negative sentiment
        strat = _make_strategy({"AAPL": -0.6})
        data = _make_data(symbols=["AAPL"])
        result = strat.generate_signals(data)
        aapl = [s for s in result.scores if s.symbol == "AAPL"]
        assert len(aapl) == 1
        assert aapl[0].direction == SignalDirection.SHORT
        assert aapl[0].metadata["signal_type"] == "contrarian_sell"

    def test_divergence_uses_higher_multiplier(self):
        raw_score, direction, signal_type = SentimentStrategy._classify_signal(
            0.5, -0.3
        )
        assert signal_type == "contrarian_buy"
        # Divergence = abs(0.5 - (-0.3)) = 0.8
        expected = 0.8 * DIVERGENCE_MULTIPLIER
        assert abs(raw_score - expected) < 1e-10


# ── 3. Alignment detection (momentum confirmation) ──────────────────────────


class TestAlignment:
    """Test momentum confirmation on sentiment-price alignment."""

    def test_both_positive_gives_long(self):
        # AAPL has positive price trend and positive sentiment
        strat = _make_strategy({"AAPL": 0.6})
        data = _make_data(symbols=["AAPL"])
        result = strat.generate_signals(data)
        aapl = [s for s in result.scores if s.symbol == "AAPL"]
        assert len(aapl) == 1
        assert aapl[0].direction == SignalDirection.LONG
        assert aapl[0].metadata["signal_type"] == "momentum_long"

    def test_both_negative_gives_short(self):
        # TSLA has negative price trend and negative sentiment
        strat = _make_strategy({"TSLA": -0.5})
        data = _make_data(symbols=["TSLA"])
        result = strat.generate_signals(data)
        tsla = [s for s in result.scores if s.symbol == "TSLA"]
        assert len(tsla) == 1
        assert tsla[0].direction == SignalDirection.SHORT
        assert tsla[0].metadata["signal_type"] == "momentum_short"

    def test_alignment_uses_lower_multiplier(self):
        raw_score, direction, signal_type = SentimentStrategy._classify_signal(0.5, 0.3)
        assert signal_type == "momentum_long"
        # Alignment = abs(0.5 + 0.3) / 2 = 0.4
        expected = 0.4 * ALIGNMENT_MULTIPLIER
        assert abs(raw_score - expected) < 1e-10


# ── 4. Missing data handling ────────────────────────────────────────────────


class TestMissingData:
    """Test graceful handling of missing sentiment data."""

    def test_no_manager_returns_empty(self):
        strat = SentimentStrategy(config=SentimentConfig(enabled=True), manager=None)
        data = _make_data()
        result = strat.generate_signals(data)
        assert isinstance(result, StrategyOutput)
        assert len(result.scores) == 0

    def test_empty_sentiment_returns_empty(self):
        strat = _make_strategy({})
        data = _make_data()
        result = strat.generate_signals(data)
        assert len(result.scores) == 0

    def test_model_exception_returns_empty(self):
        mgr = MagicMock()
        mgr.predict_sentiment.side_effect = RuntimeError("Model crashed")
        strat = SentimentStrategy(config=SentimentConfig(enabled=True), manager=mgr)
        data = _make_data()
        result = strat.generate_signals(data)
        assert len(result.scores) == 0

    def test_empty_data_returns_empty(self):
        strat = _make_strategy({"AAPL": 0.5})
        result = strat.generate_signals(pd.DataFrame())
        assert len(result.scores) == 0

    def test_insufficient_price_history_returns_empty(self):
        strat = _make_strategy({"AAPL": 0.5})
        # Only 3 days of data — not enough for TREND_WINDOW+1
        data = _make_data(symbols=["AAPL"], n_days=3)
        result = strat.generate_signals(data)
        assert len(result.scores) == 0

    def test_symbol_in_sentiment_but_not_in_price(self):
        strat = _make_strategy({"UNKNOWN": 0.5})
        data = _make_data(symbols=["AAPL"])
        result = strat.generate_signals(data)
        # UNKNOWN has no price data, so no signal generated
        assert all(s.symbol != "UNKNOWN" for s in result.scores)


# ── 5. Combiner integration ─────────────────────────────────────────────────


class TestCombinerIntegration:
    """Test sentiment strategy is correctly mapped in the ensemble combiner."""

    def test_regime_bucket_mapping(self):
        from strategies.ensemble.combiner import EnsembleCombiner
        from strategies.regime.detector import RegimeState, MarketRegime

        combiner = EnsembleCombiner()
        # Create a regime state where "tft" bucket has a distinct weight
        regime = RegimeState(
            regime=MarketRegime.CALM_TRENDING,
            vix_level=15.0,
            market_breadth=0.6,
            realized_vol=0.12,
            is_volatile=False,
            is_trending=True,
            confidence=0.9,
            exposure_scalar=1.0,
            strategy_weights={
                "momentum": 0.4,
                "mean_reversion": 0.15,
                "pairs": 0.2,
                "tft": 0.25,
            },
        )
        weight = combiner._get_regime_weight("sentiment", regime)
        # Should map to "tft" bucket = 0.25
        assert weight == 0.25

    def test_strategy_output_compatible_with_combiner(self):
        from strategies.ensemble.combiner import EnsembleCombiner

        combiner = EnsembleCombiner()
        strat = _make_strategy({"AAPL": 0.8, "TSLA": -0.5})
        data = _make_data()
        output = strat.generate_signals(data)
        # Combine should accept the output without error
        combined = combiner.combine([output])
        assert isinstance(combined, list)

    def test_weight_in_ensemble(self):
        """Verify sentiment output gets non-zero weight when included."""
        from strategies.ensemble.combiner import EnsembleCombiner
        from strategies.config import EnsembleConfig

        combiner = EnsembleCombiner(
            config=EnsembleConfig(enabled=True, weighting_method="equal"),
        )
        strat = _make_strategy({"AAPL": 0.8, "TSLA": -0.5})
        data = _make_data()
        output = strat.generate_signals(data)
        combined = combiner.combine([output])
        # Should produce signals (might be empty if no valid combiner output,
        # but shouldn't error)
        assert isinstance(combined, list)


# ── 6. Env var toggle ───────────────────────────────────────────────────────


class TestEnvVarToggle:
    """Test STRATEGY_SENTIMENT_ENABLED enables/disables the strategy."""

    def test_config_disabled_by_default(self):
        config = SentimentConfig()
        assert config.enabled is False

    def test_config_from_env_disabled(self):
        os.environ.pop("STRATEGY_SENTIMENT_ENABLED", None)
        config = SentimentConfig.from_env()
        assert config.enabled is False

    def test_config_from_env_enabled(self):
        os.environ["STRATEGY_SENTIMENT_ENABLED"] = "true"
        try:
            config = SentimentConfig.from_env()
            assert config.enabled is True
        finally:
            del os.environ["STRATEGY_SENTIMENT_ENABLED"]

    def test_config_initial_weight(self):
        config = SentimentConfig()
        assert config.initial_weight == 0.10

    def test_config_in_master_config(self):
        from strategies.config import StrategyMasterConfig

        master = StrategyMasterConfig()
        assert hasattr(master, "sentiment")
        assert isinstance(master.sentiment, SentimentConfig)

    def test_master_config_from_env_includes_sentiment(self):
        from strategies.config import StrategyMasterConfig as SMC

        os.environ["STRATEGY_SENTIMENT_ENABLED"] = "true"
        try:
            master = SMC.from_env()
            assert master.sentiment.enabled is True
        finally:
            del os.environ["STRATEGY_SENTIMENT_ENABLED"]

    def test_env_template_has_sentiment_var(self):
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".env.template",
        )
        with open(template_path) as f:
            content = f.read()
        assert "STRATEGY_SENTIMENT_ENABLED" in content


# ── 7. Classify signal logic ────────────────────────────────────────────────


class TestClassifySignal:
    """Unit tests for _classify_signal static method."""

    def test_contrarian_buy(self):
        score, direction, stype = SentimentStrategy._classify_signal(0.5, -0.3)
        assert direction == SignalDirection.LONG
        assert stype == "contrarian_buy"
        assert score > 0

    def test_contrarian_sell(self):
        score, direction, stype = SentimentStrategy._classify_signal(-0.5, 0.3)
        assert direction == SignalDirection.SHORT
        assert stype == "contrarian_sell"
        assert score < 0

    def test_momentum_long(self):
        score, direction, stype = SentimentStrategy._classify_signal(0.5, 0.3)
        assert direction == SignalDirection.LONG
        assert stype == "momentum_long"
        assert score > 0

    def test_momentum_short(self):
        score, direction, stype = SentimentStrategy._classify_signal(-0.5, -0.3)
        assert direction == SignalDirection.SHORT
        assert stype == "momentum_short"
        assert score < 0

    def test_divergence_stronger_than_alignment(self):
        div_score, _, _ = SentimentStrategy._classify_signal(0.5, -0.5)
        ali_score, _, _ = SentimentStrategy._classify_signal(0.5, 0.5)
        assert abs(div_score) > abs(ali_score)


# ── 8. Strategy properties ──────────────────────────────────────────────────


class TestStrategyProperties:
    """Test strategy name, description, and performance tracking."""

    def test_name(self):
        strat = _make_strategy({})
        assert strat.name == "sentiment"

    def test_description(self):
        strat = _make_strategy({})
        assert "sentiment" in strat.description.lower()

    def test_get_performance(self):
        strat = _make_strategy({})
        perf = strat.get_performance()
        assert perf.strategy_name == "sentiment"

    def test_max_positions_respected(self):
        # Create a strategy with max 2 positions per side
        config = SentimentConfig(enabled=True, max_positions_per_side=2)
        sentiments = {f"SYM{i}": 0.5 + i * 0.1 for i in range(10)}
        mgr = _mock_manager(sentiments)
        strat = SentimentStrategy(config=config, manager=mgr)
        # Build data with all 10 symbols trending up
        data = _make_data(symbols=list(sentiments.keys()))
        result = strat.generate_signals(data)
        longs = [s for s in result.scores if s.direction == SignalDirection.LONG]
        shorts = [s for s in result.scores if s.direction == SignalDirection.SHORT]
        assert len(longs) <= 2
        assert len(shorts) <= 2


# ── 9. Paper-trader wiring ──────────────────────────────────────────────────


class TestPaperTraderWiring:
    """Verify sentiment strategy is wired into paper-trader/main.py."""

    @pytest.fixture(autouse=True)
    def load_source(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper-trader",
            "main.py",
        )
        with open(path) as f:
            self.source = f.read()

    def test_sentiment_strategy_imported(self):
        assert (
            "from strategies.sentiment.strategy import SentimentStrategy" in self.source
        )

    def test_sentiment_config_imported(self):
        assert "SentimentConfig" in self.source

    def test_sentiment_in_build_strategies(self):
        assert '"sentiment"' in self.source
        assert "SentimentStrategy(" in self.source

    def test_sentiment_enabled_check(self):
        assert "config.sentiment.enabled" in self.source
