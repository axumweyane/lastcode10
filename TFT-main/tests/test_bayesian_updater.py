"""Tests for the Bayesian ensemble weight updater (Beta-Binomial model)."""

import json
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.ensemble.bayesian_updater import BayesianWeightUpdater, StrategyBeta
from strategies.ensemble.combiner import EnsembleCombiner, CombinedSignal
from strategies.config import EnsembleConfig
from strategies.base import StrategyOutput, AlphaScore, SignalDirection
from datetime import datetime, timezone


def _make_strategy_output(name: str, symbols_scores: dict, sharpe: float = 0.5):
    """Helper to create a StrategyOutput."""
    scores = []
    for sym, score in symbols_scores.items():
        direction = SignalDirection.LONG if score > 0 else SignalDirection.SHORT
        scores.append(
            AlphaScore(
                symbol=sym,
                score=score,
                raw_score=score,
                confidence=0.8,
                direction=direction,
            )
        )
    return StrategyOutput(
        strategy_name=name,
        timestamp=datetime.now(timezone.utc),
        scores=scores,
        strategy_sharpe_63d=sharpe,
        strategy_sharpe_21d=sharpe,
    )


# ── 1. Weights converge toward better-performing strategies ──────────────────


class TestConvergence:
    """Test that weights converge toward better-performing strategies over 100 updates."""

    def test_good_strategy_gets_higher_weight(self):
        updater = BayesianWeightUpdater(decay_factor=0.995)
        for _ in range(100):
            updater.update(
                {
                    "good": True,  # always profitable
                    "bad": False,  # always unprofitable
                }
            )

        weights = updater.get_weights()
        assert weights["good"] > weights["bad"]
        # Good should dominate
        assert weights["good"] > 0.6

    def test_mixed_strategy_intermediate_weight(self):
        updater = BayesianWeightUpdater(decay_factor=0.995)
        rng = np.random.RandomState(42)
        for _ in range(100):
            updater.update(
                {
                    "great": True,
                    "coin_flip": bool(rng.random() > 0.5),
                    "terrible": False,
                }
            )

        weights = updater.get_weights()
        assert weights["great"] > weights["coin_flip"]
        assert weights["coin_flip"] > weights["terrible"]

    def test_convergence_speed(self):
        """After 100 updates, a 70% win-rate strategy should have weight > 0.5."""
        updater = BayesianWeightUpdater()
        rng = np.random.RandomState(42)
        for _ in range(100):
            updater.update(
                {
                    "seventy_pct": bool(rng.random() < 0.7),
                    "thirty_pct": bool(rng.random() < 0.3),
                }
            )

        weights = updater.get_weights()
        assert weights["seventy_pct"] > 0.55

    def test_equal_performance_equal_weights(self):
        updater = BayesianWeightUpdater()
        rng = np.random.RandomState(42)
        for _ in range(100):
            outcome = bool(rng.random() > 0.5)
            updater.update(
                {
                    "strat_a": outcome,
                    "strat_b": outcome,  # same outcomes
                }
            )

        weights = updater.get_weights()
        # Should be roughly equal
        assert abs(weights["strat_a"] - weights["strat_b"]) < 0.05


# ── 2. Exponential decay reduces influence of old observations ───────────────


class TestExponentialDecay:
    """Test exponential forgetting factor."""

    def test_decay_reduces_old_observations(self):
        updater = BayesianWeightUpdater(decay_factor=0.99)
        # 50 good days
        for _ in range(50):
            updater.update({"strat": True})

        state_after_good = updater.get_state()["strat"]
        alpha_after_good = state_after_good.alpha

        # Now 50 bad days
        for _ in range(50):
            updater.update({"strat": False})

        state_after_bad = updater.get_state()["strat"]
        # Weight should be below 0.5 since recent data is all bad
        assert state_after_bad.weight < 0.5

    def test_no_decay_preserves_all_history(self):
        updater = BayesianWeightUpdater(decay_factor=1.0)  # no decay
        for _ in range(50):
            updater.update({"strat": True})
        for _ in range(50):
            updater.update({"strat": False})

        state = updater.get_state()["strat"]
        # alpha should be ~51 (prior 1 + 50 wins), beta ~51 (prior 1 + 50 losses)
        assert abs(state.alpha - 51.0) < 0.1
        assert abs(state.beta - 51.0) < 0.1
        assert abs(state.weight - 0.5) < 0.01

    def test_strong_decay_forgets_fast(self):
        updater = BayesianWeightUpdater(decay_factor=0.9)  # strong decay
        # 50 good days
        for _ in range(50):
            updater.update({"strat": True})
        # 10 bad days
        for _ in range(10):
            updater.update({"strat": False})

        state = updater.get_state()["strat"]
        # With strong decay, the 50 good days are heavily decayed
        # Recent 10 bad days should pull weight well below what no-decay would give
        # Without decay: alpha=51, beta=11, weight=0.82
        # With 0.9 decay: old goods decayed a lot
        assert state.weight < 0.75

    def test_decay_factor_stored(self):
        updater = BayesianWeightUpdater(decay_factor=0.99)
        assert updater.decay_factor == 0.99


# ── 3. Persistence: save and reload produces same weights ────────────────────


class TestPersistence:
    """Test save/load roundtrip."""

    def test_json_roundtrip(self):
        updater = BayesianWeightUpdater(decay_factor=0.99)
        for _ in range(20):
            updater.update({"strat_a": True, "strat_b": False})

        json_str = updater.to_json()
        restored = BayesianWeightUpdater.from_json(json_str)

        assert updater.get_weights() == restored.get_weights()
        orig_state = updater.get_state()
        rest_state = restored.get_state()
        for name in orig_state:
            assert abs(orig_state[name].alpha - rest_state[name].alpha) < 1e-10
            assert abs(orig_state[name].beta - rest_state[name].beta) < 1e-10

    def test_db_rows_roundtrip(self):
        updater = BayesianWeightUpdater()
        for _ in range(30):
            updater.update({"momentum": True, "pairs": False, "tft": True})

        rows = updater.save_to_rows()
        assert len(rows) == 3

        restored = BayesianWeightUpdater()
        restored.load_from_rows(rows)

        orig_weights = updater.get_weights()
        rest_weights = restored.get_weights()
        for name in orig_weights:
            assert abs(orig_weights[name] - rest_weights[name]) < 1e-6

    def test_empty_roundtrip(self):
        updater = BayesianWeightUpdater()
        json_str = updater.to_json()
        restored = BayesianWeightUpdater.from_json(json_str)
        assert restored.get_weights() == {}

    def test_state_dicts_serializable(self):
        updater = BayesianWeightUpdater()
        updater.update({"strat_a": True})
        dicts = updater.get_state_dicts()
        # Must be JSON-serializable
        json.dumps(dicts)
        assert len(dicts) == 1
        assert dicts[0]["strategy_name"] == "strat_a"
        assert "alpha" in dicts[0]
        assert "weight" in dicts[0]


# ── 4. Integration with combiner produces different weights ──────────────────


class TestCombinerIntegration:
    """Test that Bayesian updater changes combiner weights when enabled."""

    def _make_outputs(self):
        return [
            _make_strategy_output("strat_a", {"AAPL": 1.0, "MSFT": 0.5}, sharpe=0.5),
            _make_strategy_output("strat_b", {"AAPL": -0.5, "TSLA": 0.8}, sharpe=0.5),
        ]

    def test_bayesian_updater_changes_weights(self):
        # Train updater to strongly prefer strat_a
        updater = BayesianWeightUpdater()
        for _ in range(50):
            updater.update({"strat_a": True, "strat_b": False})

        # Combiner WITH updater
        config_on = EnsembleConfig(
            enabled=True,
            use_bayesian_updater=True,
            max_total_positions=10,
        )
        combiner_on = EnsembleCombiner(config=config_on, bayesian_updater=updater)
        combined_on = combiner_on.combine(self._make_outputs())

        # Combiner WITHOUT updater (same Sharpe → equal performance weights)
        config_off = EnsembleConfig(
            enabled=True,
            use_bayesian_updater=False,
            max_total_positions=10,
        )
        combiner_off = EnsembleCombiner(config=config_off)
        combined_off = combiner_off.combine(self._make_outputs())

        # Weights should differ
        weights_on = combiner_on.get_weight_history(1)[-1]
        weights_off = combiner_off.get_weight_history(1)[-1]

        # strat_a should get more weight with updater since it was always profitable
        assert weights_on["strat_a"].raw_weight > weights_off["strat_a"].raw_weight

    def test_updater_weights_are_normalized(self):
        updater = BayesianWeightUpdater()
        updater.update({"strat_a": True, "strat_b": True})

        weights = updater.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_single_strategy_gets_full_weight(self):
        updater = BayesianWeightUpdater()
        updater.update({"only_one": True})
        weights = updater.get_weights()
        assert abs(weights["only_one"] - 1.0) < 1e-10


# ── 5. Backward compatibility ────────────────────────────────────────────────


class TestBackwardCompatibility:
    """Test that disabling Bayesian updater leaves fixed weights unchanged."""

    def test_disabled_by_default(self):
        config = EnsembleConfig(enabled=True)
        assert config.use_bayesian_updater is False

    def test_disabled_combiner_ignores_updater(self):
        updater = BayesianWeightUpdater()
        for _ in range(50):
            updater.update({"strat_a": True, "strat_b": False})

        # Even though updater is provided, use_bayesian_updater=False
        config = EnsembleConfig(
            enabled=True,
            use_bayesian_updater=False,
            max_total_positions=10,
        )
        combiner = EnsembleCombiner(config=config, bayesian_updater=updater)
        outputs = [
            _make_strategy_output("strat_a", {"AAPL": 1.0}, sharpe=0.5),
            _make_strategy_output("strat_b", {"AAPL": -0.5}, sharpe=0.5),
        ]
        combiner.combine(outputs)
        weights = combiner.get_weight_history(1)[-1]

        # Both strategies have same Sharpe → raw weights should be roughly equal
        assert abs(weights["strat_a"].raw_weight - weights["strat_b"].raw_weight) < 0.01

    def test_no_updater_passed(self):
        """Combiner works fine when no updater is provided at all."""
        config = EnsembleConfig(
            enabled=True,
            use_bayesian_updater=True,
            max_total_positions=10,
        )
        combiner = EnsembleCombiner(config=config, bayesian_updater=None)
        outputs = [
            _make_strategy_output("strat_a", {"AAPL": 1.0}, sharpe=0.5),
        ]
        combined = combiner.combine(outputs)
        assert len(combined) > 0

    def test_fixed_sharpe_weights_unchanged(self):
        """Without Bayesian updater, Sharpe-based weights work as before."""
        config = EnsembleConfig(
            enabled=True,
            weighting_method="sharpe",
            max_total_positions=10,
        )
        combiner = EnsembleCombiner(config=config)
        outputs = [
            _make_strategy_output("high_sharpe", {"AAPL": 1.0}, sharpe=2.0),
            _make_strategy_output("low_sharpe", {"AAPL": 0.5}, sharpe=0.1),
        ]
        combiner.combine(outputs)
        weights = combiner.get_weight_history(1)[-1]
        assert weights["high_sharpe"].raw_weight > weights["low_sharpe"].raw_weight


# ── 6. Paper-trader structural tests ────────────────────────────────────────


class TestPaperTraderWiring:
    """Verify Bayesian updater is wired into paper-trader."""

    def _read_source(self):
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper-trader",
            "main.py",
        )
        with open(main_path) as f:
            return f.read()

    def test_bayesian_updater_import(self):
        source = self._read_source()
        assert (
            "from strategies.ensemble.bayesian_updater import BayesianWeightUpdater"
            in source
        )

    def test_bayesian_updater_global(self):
        source = self._read_source()
        assert "bayesian_updater: Optional[BayesianWeightUpdater]" in source

    def test_bayesian_updater_initialized(self):
        source = self._read_source()
        assert "BayesianWeightUpdater(" in source

    def test_save_bayesian_state_called(self):
        source = self._read_source()
        assert "save_bayesian_state" in source

    def test_load_bayesian_state_called(self):
        source = self._read_source()
        assert "load_bayesian_state" in source

    def test_weights_bayesian_endpoint(self):
        source = self._read_source()
        assert '"/weights/bayesian"' in source

    def test_bayesian_table_in_schema(self):
        source = self._read_source()
        assert "bayesian_weight_state" in source

    def test_bayesian_table_in_postgres_schema(self):
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "postgres_schema.py",
        )
        with open(schema_path) as f:
            source = f.read()
        assert "bayesian_weight_state" in source

    def test_env_var_controls_activation(self):
        source = self._read_source()
        assert "ENSEMBLE_USE_BAYESIAN_WEIGHTS" in source


# ── 7. StrategyBeta unit tests ───────────────────────────────────────────────


class TestStrategyBeta:
    """Unit tests for the StrategyBeta dataclass."""

    def test_uninformative_prior(self):
        s = StrategyBeta(strategy_name="test")
        assert s.alpha == 1.0
        assert s.beta == 1.0
        assert s.weight == 0.5

    def test_weight_calculation(self):
        s = StrategyBeta(strategy_name="test", alpha=3.0, beta=1.0)
        assert s.weight == 0.75

    def test_variance_decreases_with_data(self):
        prior = StrategyBeta(strategy_name="test")
        updated = StrategyBeta(strategy_name="test", alpha=51.0, beta=51.0)
        assert updated.variance < prior.variance

    def test_to_dict(self):
        s = StrategyBeta(strategy_name="test", alpha=2.0, beta=3.0, n_updates=5)
        d = s.to_dict()
        assert d["strategy_name"] == "test"
        assert d["alpha"] == 2.0
        assert d["beta"] == 3.0
        assert d["weight"] == 0.4
        assert d["n_updates"] == 5
