"""Tests for strategies/validation/walk_forward.py — walk-forward cross-validation."""

import json
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.config import WalkForwardConfig
from strategies.validation.walk_forward import (
    WalkForwardValidator,
    NormalizationStats,
    FoldResult,
    WalkForwardReport,
    compute_sharpe,
    compute_max_drawdown,
    compute_win_rate,
    compute_profit_factor,
    ANNUALIZATION_FACTORS,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_data(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create synthetic time-series data."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    return pd.DataFrame({
        "timestamp": dates,
        "close": 100 + np.cumsum(rng.randn(n_rows) * 0.5),
        "volume": rng.randint(1000, 10000, n_rows).astype(float),
        "feature_a": rng.randn(n_rows),
        "feature_b": rng.randn(n_rows),
    })


def _dummy_strategy(train: pd.DataFrame, test: pd.DataFrame, fold: int) -> np.ndarray:
    """Simple strategy: return random OOS returns seeded by fold index."""
    rng = np.random.RandomState(fold)
    return rng.randn(len(test)) * 0.01


def _good_then_bad_strategy(train, test, fold):
    """Earlier folds return positive, later folds return negative."""
    rng = np.random.RandomState(fold)
    if fold == 0:
        return np.abs(rng.randn(len(test))) * 0.02  # positive returns
    else:
        return -np.abs(rng.randn(len(test))) * 0.02  # negative returns


# ── 1. Embargo Gap ─────────────────────────────────────────────────────────

class TestEmbargoGap:
    def test_embargo_skips_bars(self):
        """OOS start must be IS end + embargo_bars, not IS end."""
        config = WalkForwardConfig(is_window=100, oos_window=50, embargo_bars=10)
        validator = WalkForwardValidator(config)
        folds = validator.generate_folds(300)

        for (is_start, is_end), (oos_start, oos_end) in folds:
            gap = oos_start - is_end
            assert gap == 10, f"Expected embargo gap of 10, got {gap}"

    def test_embargo_zero_allowed(self):
        """Embargo of 0 means OOS starts immediately after IS."""
        config = WalkForwardConfig(is_window=100, oos_window=50, embargo_bars=0)
        validator = WalkForwardValidator(config)
        folds = validator.generate_folds(300)

        for (is_start, is_end), (oos_start, oos_end) in folds:
            assert oos_start == is_end

    def test_embargo_large_reduces_folds(self):
        """Large embargo should reduce available folds."""
        config_small = WalkForwardConfig(is_window=100, oos_window=50, embargo_bars=2)
        config_large = WalkForwardConfig(is_window=100, oos_window=50, embargo_bars=30)
        v_small = WalkForwardValidator(config_small)
        v_large = WalkForwardValidator(config_large)

        folds_small = v_small.generate_folds(400)
        folds_large = v_large.generate_folds(400)
        assert len(folds_large) <= len(folds_small)

    def test_embargo_no_overlap(self):
        """IS and OOS data must never overlap (embargo enforces gap)."""
        config = WalkForwardConfig(is_window=100, oos_window=50, embargo_bars=5)
        validator = WalkForwardValidator(config)
        folds = validator.generate_folds(500)

        for (is_start, is_end), (oos_start, oos_end) in folds:
            assert oos_start > is_end, "OOS must start after IS end"
            assert oos_start - is_end >= 5, "Embargo gap too small"


# ── 2. Most Recent Fold Deployment ─────────────────────────────────────────

class TestMostRecentFoldDeployment:
    def test_deploys_most_recent_fold(self):
        """Deployed fold must be the last fold, NOT the best Sharpe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WalkForwardConfig(
                is_window=100, oos_window=50, embargo_bars=5,
                norm_stats_dir=tmpdir,
            )
            validator = WalkForwardValidator(config)
            data = _make_data(400)
            report = validator.run(data, _good_then_bad_strategy)

            # The best Sharpe should be fold 0 (positive returns)
            # But deployed must be the LAST fold
            assert report.deployed_fold_index == report.total_folds - 1
            assert report.best_sharpe_fold_index == 0
            assert report.deployed_fold_index != report.best_sharpe_fold_index

    def test_deployed_fold_is_last_even_when_worst(self):
        """Even if the most recent fold has terrible Sharpe, still deploy it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WalkForwardConfig(
                is_window=100, oos_window=50, embargo_bars=5,
                norm_stats_dir=tmpdir,
            )
            validator = WalkForwardValidator(config)
            data = _make_data(400)

            def worst_last(train, test, fold):
                if fold == report_folds_count - 1:
                    return np.full(len(test), -0.05)
                return np.full(len(test), 0.01)

            folds = validator.generate_folds(len(data))
            report_folds_count = len(folds)

            report = validator.run(data, worst_last)
            assert report.deployed_fold_index == report.total_folds - 1


# ── 3. Normalization Stats ─────────────────────────────────────────────────

class TestNormalizationStats:
    def test_stats_saved_per_fold(self):
        """Each fold must produce a normalization stats JSON sidecar."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WalkForwardConfig(
                is_window=100, oos_window=50, embargo_bars=5,
                norm_stats_dir=tmpdir,
            )
            validator = WalkForwardValidator(config)
            data = _make_data(400)
            report = validator.run(data, _dummy_strategy)

            for fold in report.folds:
                assert fold.norm_stats_path is not None
                assert os.path.exists(fold.norm_stats_path), (
                    f"Norm stats file missing: {fold.norm_stats_path}"
                )

    def test_stats_content_correct(self):
        """Normalization stats must contain correct mean/std for IS data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WalkForwardConfig(
                is_window=100, oos_window=50, embargo_bars=5,
                norm_stats_dir=tmpdir,
            )
            validator = WalkForwardValidator(config)
            data = _make_data(300)
            report = validator.run(data, _dummy_strategy)

            # Load first fold stats
            stats = NormalizationStats.load(report.folds[0].norm_stats_path)
            assert stats.fold_index == 0
            assert "close" in stats.columns
            assert "volume" in stats.columns
            assert stats.n_samples == 100  # IS window size

            # Verify the mean is close to the actual IS data mean
            is_data = data.iloc[0:100]
            expected_close_mean = float(is_data["close"].mean())
            assert abs(stats.mean["close"] - expected_close_mean) < 1e-6

    def test_stats_load_roundtrip(self):
        """Save and load must produce identical stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = NormalizationStats(
                fold_index=3,
                mean={"a": 1.5, "b": -0.3},
                std={"a": 2.0, "b": 0.8},
                columns=["a", "b"],
                n_samples=200,
                frequency="daily",
            )
            path = os.path.join(tmpdir, "test_stats.json")
            original.save(path)
            loaded = NormalizationStats.load(path)

            assert loaded.fold_index == 3
            assert loaded.mean == {"a": 1.5, "b": -0.3}
            assert loaded.std == {"a": 2.0, "b": 0.8}
            assert loaded.columns == ["a", "b"]
            assert loaded.n_samples == 200
            assert loaded.frequency == "daily"

    def test_missing_stats_raises_error(self):
        """Loading deployed norm stats when file is missing must raise."""
        fold = FoldResult(
            fold_index=0, is_start=0, is_end=100, oos_start=105, oos_end=155,
            embargo_bars=5, sharpe=1.0, max_drawdown=0.05, win_rate=0.6,
            profit_factor=2.0, n_trades=50, total_return=0.10,
            annualization_factor=np.sqrt(252), frequency="daily",
            norm_stats_path="/nonexistent/path/stats.json",
        )
        report = WalkForwardReport(
            folds=[fold], deployed_fold_index=0, best_sharpe_fold_index=0,
            sharpe_mean=1.0, sharpe_std=0.0, sharpe_stability=1.0,
            total_folds=1, config={},
        )
        with pytest.raises(FileNotFoundError, match="Normalization stats not found"):
            WalkForwardValidator.load_deployed_norm_stats(report)


# ── 4. Sharpe Annualization ────────────────────────────────────────────────

class TestSharpeAnnualization:
    def test_daily_annualization(self):
        """Daily data must use sqrt(252)."""
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.008] * 20)
        sharpe_daily = compute_sharpe(returns, "daily")
        # Manually compute
        expected = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252))
        assert abs(sharpe_daily - expected) < 1e-10

    def test_minute_annualization(self):
        """Minute data must use sqrt(252 * 390)."""
        returns = np.array([0.001, 0.002, -0.001, 0.0005] * 50)
        sharpe_minute = compute_sharpe(returns, "minute")
        expected = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252 * 390))
        assert abs(sharpe_minute - expected) < 1e-10

    def test_daily_vs_minute_different(self):
        """Same returns should produce different Sharpe for daily vs minute."""
        returns = np.array([0.01, -0.005, 0.008, 0.003, -0.002] * 10)
        sharpe_d = compute_sharpe(returns, "daily")
        sharpe_m = compute_sharpe(returns, "minute")
        assert sharpe_m > sharpe_d  # minute has larger annualization factor

    def test_frequency_stored_in_metadata(self):
        """Fold results must store the frequency used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WalkForwardConfig(
                is_window=100, oos_window=50, embargo_bars=5,
                frequency="minute", norm_stats_dir=tmpdir,
            )
            validator = WalkForwardValidator(config)
            data = _make_data(300)
            report = validator.run(data, _dummy_strategy)
            for fold in report.folds:
                assert fold.frequency == "minute"
                assert fold.annualization_factor == np.sqrt(252 * 390)


# ── 5. Sharpe Warning ─────────────────────────────────────────────────────

class TestSharpeWarning:
    def test_warning_when_latest_fold_much_worse(self):
        """Warning must be emitted when deployed fold Sharpe is far below best."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WalkForwardConfig(
                is_window=100, oos_window=50, embargo_bars=5,
                norm_stats_dir=tmpdir,
                sharpe_warning_threshold=0.5,
            )
            validator = WalkForwardValidator(config)
            data = _make_data(400)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                report = validator.run(data, _good_then_bad_strategy)
                # Should have a warning about degrading model
                user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
                assert len(user_warnings) >= 1
                assert "degrading" in str(user_warnings[0].message).lower()

            assert len(report.warnings) >= 1

    def test_no_warning_when_folds_similar(self):
        """No warning when all folds have similar Sharpe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WalkForwardConfig(
                is_window=100, oos_window=50, embargo_bars=5,
                norm_stats_dir=tmpdir,
                sharpe_warning_threshold=0.5,
            )
            validator = WalkForwardValidator(config)
            data = _make_data(400)

            def uniform_strategy(train, test, fold):
                rng = np.random.RandomState(42)  # same seed every fold
                return rng.randn(len(test)) * 0.01

            report = validator.run(data, uniform_strategy)
            # All folds have identical returns -> no degradation warning
            degradation_warnings = [w for w in report.warnings if "degrading" in w.lower()]
            assert len(degradation_warnings) == 0


# ── 6. Fold Metrics ────────────────────────────────────────────────────────

class TestFoldMetrics:
    def test_metrics_computed(self):
        """All fold metrics must be populated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WalkForwardConfig(
                is_window=100, oos_window=50, embargo_bars=5,
                norm_stats_dir=tmpdir,
            )
            validator = WalkForwardValidator(config)
            data = _make_data(300)
            report = validator.run(data, _dummy_strategy)

            for fold in report.folds:
                assert isinstance(fold.sharpe, float)
                assert isinstance(fold.max_drawdown, float)
                assert isinstance(fold.win_rate, float)
                assert isinstance(fold.profit_factor, float)
                assert isinstance(fold.n_trades, int)
                assert 0.0 <= fold.win_rate <= 1.0

    def test_report_json_roundtrip(self):
        """Report must serialize to valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WalkForwardConfig(
                is_window=100, oos_window=50, embargo_bars=5,
                norm_stats_dir=tmpdir,
            )
            validator = WalkForwardValidator(config)
            data = _make_data(300)
            report = validator.run(data, _dummy_strategy)

            json_str = report.to_json()
            parsed = json.loads(json_str)
            assert "folds" in parsed
            assert "deployed_fold_index" in parsed
            assert "sharpe_stability" in parsed

    def test_report_markdown(self):
        """Report must generate valid markdown with table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WalkForwardConfig(
                is_window=100, oos_window=50, embargo_bars=5,
                norm_stats_dir=tmpdir,
            )
            validator = WalkForwardValidator(config)
            data = _make_data(300)
            report = validator.run(data, _dummy_strategy)

            md = report.to_markdown()
            assert "Walk-Forward Validation Report" in md
            assert "| Fold |" in md
            assert "deployed fold" in md.lower()

    def test_sharpe_stability(self):
        """Stability should be high when all folds have similar Sharpe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = WalkForwardConfig(
                is_window=100, oos_window=50, embargo_bars=5,
                norm_stats_dir=tmpdir,
            )
            validator = WalkForwardValidator(config)
            data = _make_data(400)

            def stable_strategy(train, test, fold):
                # Same seed every fold -> identical Sharpe across folds
                rng = np.random.RandomState(99)
                return rng.randn(len(test)) * 0.01 + 0.005

            report = validator.run(data, stable_strategy)
            # All folds use same seed -> near-identical Sharpe -> stability near 1.0
            # Partial last fold has different length so stability won't be exactly 1.0
            assert report.sharpe_stability > 0.9


# ── 7. Edge Cases ──────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_insufficient_data_raises(self):
        """Must raise if data is too short for even one fold."""
        config = WalkForwardConfig(is_window=100, oos_window=50, embargo_bars=5)
        validator = WalkForwardValidator(config)
        data = _make_data(100)  # too short: need 100 + 5 + 1 = 106 minimum
        with pytest.raises(ValueError, match="Not enough data"):
            validator.run(data, _dummy_strategy)

    def test_single_fold(self):
        """Exactly enough data for one fold should work."""
        config = WalkForwardConfig(is_window=50, oos_window=20, embargo_bars=3)
        validator = WalkForwardValidator(config)
        # Need 50 + 3 + 20 = 73 rows
        data = _make_data(73)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.norm_stats_dir = tmpdir
            validator = WalkForwardValidator(config)
            report = validator.run(data, _dummy_strategy)
            assert report.total_folds == 1
            assert report.deployed_fold_index == 0

    def test_env_override(self, monkeypatch):
        """WalkForwardConfig must respect environment variables."""
        monkeypatch.setenv("WF_IS_WINDOW", "500")
        monkeypatch.setenv("WF_OOS_WINDOW", "125")
        monkeypatch.setenv("WF_EMBARGO_BARS", "10")
        monkeypatch.setenv("WF_MIN_SHARPE", "0.3")
        config = WalkForwardConfig.from_env()
        assert config.is_window == 500
        assert config.oos_window == 125
        assert config.embargo_bars == 10
        assert config.min_sharpe == 0.3

    def test_compute_max_drawdown(self):
        """Max drawdown on known series."""
        # Cumulative: 0.1, 0.15, 0.05, 0.07 -> peak at 0.15, dd at 0.05 = 0.10
        returns = np.array([0.10, 0.05, -0.10, 0.02])
        dd = compute_max_drawdown(returns)
        assert abs(dd - 0.10) < 1e-10

    def test_compute_profit_factor(self):
        """Profit factor = gross gains / gross losses."""
        returns = np.array([0.10, -0.05, 0.06, -0.03])
        pf = compute_profit_factor(returns)
        assert abs(pf - (0.16 / 0.08)) < 1e-10

    def test_compute_profit_factor_no_losses(self):
        returns = np.array([0.10, 0.05, 0.08])
        pf = compute_profit_factor(returns)
        assert pf == float("inf")
