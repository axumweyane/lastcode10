"""Tests for Prometheus metrics and Grafana dashboard."""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prometheus_client import CollectorRegistry

from monitoring.metrics import (
    PrometheusMetrics,
    CONFIDENCE_BUCKETS,
    SLIPPAGE_BUCKETS,
    DURATION_BUCKETS,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_metrics() -> PrometheusMetrics:
    """Create a PrometheusMetrics with a fresh registry (avoids cross-test collisions)."""
    return PrometheusMetrics(registry=CollectorRegistry())


def _sample_signals():
    return [
        {
            "symbol": "AAPL",
            "combined_score": 1.5,
            "confidence": 0.85,
            "direction": "long",
        },
        {
            "symbol": "MSFT",
            "combined_score": 0.8,
            "confidence": 0.72,
            "direction": "long",
        },
        {
            "symbol": "TSLA",
            "combined_score": -1.2,
            "confidence": 0.65,
            "direction": "short",
        },
    ]


# ── 1. Metric registration ───────────────────────────────────────────────────


class TestMetricRegistration:
    """Verify all 9 metrics are registered with correct names and labels."""

    def test_signal_score_registered(self):
        m = _make_metrics()
        assert m.signal_score._name == "apex_signal_score"

    def test_strategy_weight_registered(self):
        m = _make_metrics()
        assert m.strategy_weight._name == "apex_strategy_weight"

    def test_regime_state_registered(self):
        m = _make_metrics()
        assert m.regime_state._name == "apex_regime_state"

    def test_ensemble_confidence_registered(self):
        m = _make_metrics()
        assert m.ensemble_confidence._name == "apex_ensemble_confidence"

    def test_execution_slippage_registered(self):
        m = _make_metrics()
        assert m.execution_slippage._name == "apex_execution_slippage_bps"

    def test_pipeline_duration_registered(self):
        m = _make_metrics()
        assert m.pipeline_duration._name == "apex_pipeline_duration_seconds"

    def test_risk_drawdown_registered(self):
        m = _make_metrics()
        assert m.risk_drawdown._name == "apex_risk_drawdown"

    def test_risk_var_99_registered(self):
        m = _make_metrics()
        assert m.risk_var_99._name == "apex_risk_var_99"

    def test_risk_cvar_95_registered(self):
        m = _make_metrics()
        assert m.risk_cvar_95._name == "apex_risk_cvar_95"

    def test_total_metric_count(self):
        m = _make_metrics()
        # 9 metrics total
        metric_names = {
            "apex_signal_score",
            "apex_strategy_weight",
            "apex_regime_state",
            "apex_ensemble_confidence",
            "apex_execution_slippage_bps",
            "apex_pipeline_duration_seconds",
            "apex_risk_drawdown",
            "apex_risk_var_99",
            "apex_risk_cvar_95",
        }
        output = m.generate().decode()
        for name in metric_names:
            assert name in output, f"Metric {name} not found in /metrics output"


# ── 2. Label correctness ─────────────────────────────────────────────────────


class TestLabelCorrectness:
    """Verify label names on labeled metrics."""

    def test_signal_score_labels(self):
        m = _make_metrics()
        assert m.signal_score._labelnames == ("symbol", "direction")

    def test_strategy_weight_labels(self):
        m = _make_metrics()
        assert m.strategy_weight._labelnames == ("strategy_name", "weight_type")

    def test_execution_slippage_labels(self):
        m = _make_metrics()
        assert m.execution_slippage._labelnames == ("symbol",)

    def test_regime_state_no_custom_labels(self):
        # Info metric has no user-defined labelnames
        m = _make_metrics()
        assert m.regime_state._name == "apex_regime_state"


# ── 3. Signal updates ────────────────────────────────────────────────────────


class TestSignalUpdates:
    """Test update_signals() sets gauges and observes histograms."""

    def test_signal_scores_set(self):
        m = _make_metrics()
        m.update_signals(_sample_signals())
        val = m.signal_score.labels(symbol="AAPL", direction="long")._value.get()
        assert val == 1.5

    def test_confidence_histogram_observed(self):
        m = _make_metrics()
        m.update_signals(_sample_signals())
        # Sum of observations should be 0.85 + 0.72 + 0.65 = 2.22
        output = m.generate().decode()
        assert "apex_ensemble_confidence_count 3.0" in output

    def test_clears_stale_signals(self):
        m = _make_metrics()
        m.update_signals(
            [
                {
                    "symbol": "OLD",
                    "combined_score": 1,
                    "confidence": 0.5,
                    "direction": "long",
                }
            ]
        )
        m.update_signals(_sample_signals())
        # OLD should be gone
        output = m.generate().decode()
        assert 'symbol="OLD"' not in output

    def test_handles_missing_fields(self):
        m = _make_metrics()
        m.update_signals([{"symbol": "X"}])
        val = m.signal_score.labels(symbol="X", direction="neutral")._value.get()
        assert val == 0.0

    def test_empty_signals(self):
        m = _make_metrics()
        m.update_signals([])
        output = m.generate().decode()
        assert "apex_signal_score" in output  # metric exists but no labeled values


# ── 4. Weight updates ────────────────────────────────────────────────────────


class TestWeightUpdates:
    """Test update_weights() for fixed and Bayesian weights."""

    def test_fixed_weights_set(self):
        m = _make_metrics()
        m.update_weights({"momentum": 0.35, "tft": 0.30})
        val = m.strategy_weight.labels(
            strategy_name="momentum", weight_type="fixed"
        )._value.get()
        assert val == 0.35

    def test_bayesian_weights_set(self):
        m = _make_metrics()
        m.update_weights({"momentum": 0.35}, bayesian_weights={"momentum": 0.42})
        val = m.strategy_weight.labels(
            strategy_name="momentum", weight_type="bayesian"
        )._value.get()
        assert val == 0.42

    def test_no_bayesian_when_none(self):
        m = _make_metrics()
        m.update_weights({"tft": 0.5}, bayesian_weights=None)
        output = m.generate().decode()
        assert 'weight_type="bayesian"' not in output

    def test_clears_stale_weights(self):
        m = _make_metrics()
        m.update_weights({"old_strat": 0.5})
        m.update_weights({"new_strat": 0.7})
        output = m.generate().decode()
        assert "old_strat" not in output


# ── 5. Regime updates ────────────────────────────────────────────────────────


class TestRegimeUpdates:
    """Test update_regime() sets Info metric."""

    def test_regime_info_set(self):
        m = _make_metrics()
        m.update_regime("calm_trending", is_volatile=False, is_trending=True)
        output = m.generate().decode()
        assert 'regime="calm_trending"' in output
        assert 'volatility="calm"' in output
        assert 'trend="trending"' in output

    def test_volatile_regime(self):
        m = _make_metrics()
        m.update_regime("volatile_choppy", is_volatile=True, is_trending=False)
        output = m.generate().decode()
        assert 'volatility="volatile"' in output
        assert 'trend="choppy"' in output


# ── 6. Risk updates ──────────────────────────────────────────────────────────


class TestRiskUpdates:
    """Test update_risk() sets gauges."""

    def test_risk_values_set(self):
        m = _make_metrics()
        m.update_risk(drawdown=0.08, var_99=0.025, cvar_95=0.04)
        assert m.risk_drawdown._value.get() == 0.08
        assert m.risk_var_99._value.get() == 0.025
        assert m.risk_cvar_95._value.get() == 0.04

    def test_defaults_to_zero(self):
        m = _make_metrics()
        m.update_risk()
        assert m.risk_drawdown._value.get() == 0.0


# ── 7. Slippage observations ─────────────────────────────────────────────────


class TestSlippageObservations:
    """Test observe_slippage() records per-symbol histogram."""

    def test_slippage_observed(self):
        m = _make_metrics()
        m.observe_slippage("AAPL", 5.2)
        m.observe_slippage("AAPL", -3.1)
        output = m.generate().decode()
        assert 'apex_execution_slippage_bps_count{symbol="AAPL"} 2.0' in output

    def test_multiple_symbols(self):
        m = _make_metrics()
        m.observe_slippage("AAPL", 1.0)
        m.observe_slippage("MSFT", 2.0)
        output = m.generate().decode()
        assert 'symbol="AAPL"' in output
        assert 'symbol="MSFT"' in output


# ── 8. Pipeline duration ─────────────────────────────────────────────────────


class TestPipelineDuration:
    """Test observe_pipeline_duration() records histogram."""

    def test_duration_observed(self):
        m = _make_metrics()
        m.observe_pipeline_duration(42.5)
        output = m.generate().decode()
        assert "apex_pipeline_duration_seconds_count 1.0" in output

    def test_multiple_observations(self):
        m = _make_metrics()
        m.observe_pipeline_duration(10.0)
        m.observe_pipeline_duration(30.0)
        m.observe_pipeline_duration(90.0)
        output = m.generate().decode()
        assert "apex_pipeline_duration_seconds_count 3.0" in output


# ── 9. /metrics endpoint format ──────────────────────────────────────────────


class TestMetricsEndpoint:
    """Test generate() returns valid Prometheus exposition format."""

    def test_returns_bytes(self):
        m = _make_metrics()
        result = m.generate()
        assert isinstance(result, bytes)

    def test_contains_help_lines(self):
        m = _make_metrics()
        output = m.generate().decode()
        assert "# HELP apex_signal_score" in output
        assert "# HELP apex_risk_drawdown" in output

    def test_contains_type_lines(self):
        m = _make_metrics()
        output = m.generate().decode()
        assert "# TYPE apex_signal_score gauge" in output
        assert "# TYPE apex_ensemble_confidence histogram" in output
        assert "# TYPE apex_pipeline_duration_seconds histogram" in output

    def test_asgi_app_callable(self):
        m = _make_metrics()
        app = m.get_asgi_app()
        assert callable(app)

    def test_full_simulated_pipeline(self):
        """Simulate a full pipeline update and verify all metrics present."""
        m = _make_metrics()
        m.update_signals(_sample_signals())
        m.update_weights(
            {"momentum": 0.35, "tft": 0.30, "pairs": 0.20},
            bayesian_weights={"momentum": 0.38, "tft": 0.32, "pairs": 0.18},
        )
        m.update_regime("calm_trending", is_volatile=False, is_trending=True)
        m.update_risk(drawdown=0.05, var_99=0.02, cvar_95=0.03)
        m.observe_slippage("AAPL", 4.5)
        m.observe_pipeline_duration(35.2)

        output = m.generate().decode()
        # All metric families present
        for name in [
            "apex_signal_score",
            "apex_strategy_weight",
            "apex_regime_state",
            "apex_ensemble_confidence",
            "apex_execution_slippage_bps",
            "apex_pipeline_duration_seconds",
            "apex_risk_drawdown",
            "apex_risk_var_99",
            "apex_risk_cvar_95",
        ]:
            assert name in output, f"Missing metric: {name}"


# ── 10. Histogram bucket definitions ─────────────────────────────────────────


class TestBucketDefinitions:
    """Verify histogram bucket configurations."""

    def test_confidence_buckets(self):
        assert CONFIDENCE_BUCKETS == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    def test_slippage_buckets(self):
        assert SLIPPAGE_BUCKETS == (
            -50,
            -20,
            -10,
            -5,
            -2,
            0,
            2,
            5,
            10,
            20,
            50,
            100,
            200,
        )

    def test_duration_buckets(self):
        assert DURATION_BUCKETS == (1, 5, 10, 30, 60, 120, 300, 600)


# ── 11. Grafana dashboard JSON ───────────────────────────────────────────────


class TestGrafanaDashboard:
    """Validate the Grafana dashboard JSON structure."""

    @pytest.fixture(autouse=True)
    def load_dashboard(self):
        dashboard_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "monitoring",
            "grafana",
            "dashboards",
            "apex_ensemble.json",
        )
        with open(dashboard_path) as f:
            self.dashboard = json.load(f)

    def test_valid_json(self):
        assert isinstance(self.dashboard, dict)

    def test_has_8_panels(self):
        assert len(self.dashboard["panels"]) == 8

    def test_panel_ids_unique(self):
        ids = [p["id"] for p in self.dashboard["panels"]]
        assert len(ids) == len(set(ids))

    def test_auto_refresh_60s(self):
        assert self.dashboard["refresh"] == "1m"

    def test_has_symbol_variable(self):
        var_names = [v["name"] for v in self.dashboard["templating"]["list"]]
        assert "symbol" in var_names

    def test_has_strategy_variable(self):
        var_names = [v["name"] for v in self.dashboard["templating"]["list"]]
        assert "strategy" in var_names

    def test_has_datasource_variable(self):
        var_names = [v["name"] for v in self.dashboard["templating"]["list"]]
        assert "DS_PROMETHEUS" in var_names

    def test_all_panels_have_targets(self):
        for panel in self.dashboard["panels"]:
            assert "targets" in panel, f"Panel '{panel['title']}' missing targets"
            assert len(panel["targets"]) > 0

    def test_panel_types(self):
        types = {p["type"] for p in self.dashboard["panels"]}
        expected = {"timeseries", "histogram", "stat", "barchart", "heatmap"}
        assert expected.issubset(types)

    def test_dashboard_uid(self):
        assert self.dashboard["uid"] == "apex-ensemble-dashboard"

    def test_dashboard_tags(self):
        assert "apex" in self.dashboard["tags"]
        assert "trading" in self.dashboard["tags"]

    def test_signal_contribution_panel(self):
        panel = self.dashboard["panels"][0]
        assert panel["title"] == "Per-Strategy Signal Contribution"
        assert panel["type"] == "timeseries"
        assert "apex_signal_score" in panel["targets"][0]["expr"]

    def test_weights_panel(self):
        panel = self.dashboard["panels"][1]
        assert "Weight" in panel["title"]
        assert any("fixed" in t["expr"] for t in panel["targets"])
        assert any("bayesian" in t["expr"] for t in panel["targets"])

    def test_drawdown_panel_has_thresholds(self):
        panel = self.dashboard["panels"][5]
        assert "Drawdown" in panel["title"]
        thresholds = panel["fieldConfig"]["defaults"]["thresholds"]["steps"]
        assert len(thresholds) >= 2

    def test_slippage_panel(self):
        panel = self.dashboard["panels"][6]
        assert "Slippage" in panel["title"]
        assert "apex_execution_slippage_bps" in panel["targets"][0]["expr"]


# ── 12. Config files ─────────────────────────────────────────────────────────


class TestConfigFiles:
    """Validate Prometheus and Grafana config files."""

    def test_datasource_yml_exists(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "monitoring",
            "grafana",
            "datasources",
            "datasource.yml",
        )
        assert os.path.exists(path)

    def test_datasource_has_prometheus(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "monitoring",
            "grafana",
            "datasources",
            "datasource.yml",
        )
        with open(path) as f:
            content = f.read()
        assert "prometheus" in content.lower()
        assert "http://prometheus:9090" in content

    def test_prometheus_targets_exists(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "monitoring",
            "prometheus",
            "apex_targets.yml",
        )
        assert os.path.exists(path)

    def test_prometheus_targets_has_paper_trader(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "monitoring",
            "prometheus",
            "apex_targets.yml",
        )
        with open(path) as f:
            content = f.read()
        assert "paper-trader:8010" in content
        assert "/metrics" in content

    def test_env_template_has_metrics_var(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".env.template",
        )
        with open(path) as f:
            content = f.read()
        assert "METRICS_ENABLED" in content


# ── 13. Paper-trader wiring ──────────────────────────────────────────────────


class TestPaperTraderWiring:
    """Verify metrics are wired into paper-trader/main.py."""

    @pytest.fixture(autouse=True)
    def load_source(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper-trader",
            "main.py",
        )
        with open(path) as f:
            self.source = f.read()

    def test_metrics_import(self):
        assert "from monitoring.metrics import PrometheusMetrics" in self.source

    def test_metrics_global(self):
        assert "prom_metrics" in self.source

    def test_metrics_enabled_env(self):
        assert "METRICS_ENABLED" in self.source

    def test_update_signals_called(self):
        assert "prom_metrics.update_signals" in self.source

    def test_update_weights_called(self):
        assert "prom_metrics.update_weights" in self.source

    def test_update_regime_called(self):
        assert "prom_metrics.update_regime" in self.source

    def test_update_risk_called(self):
        assert "prom_metrics.update_risk" in self.source

    def test_observe_pipeline_duration_called(self):
        assert "prom_metrics.observe_pipeline_duration" in self.source

    def test_observe_slippage_called(self):
        assert "prom_metrics.observe_slippage" in self.source

    def test_metrics_endpoint_mounted(self):
        assert 'app.mount("/metrics"' in self.source
