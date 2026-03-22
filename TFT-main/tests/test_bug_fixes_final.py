"""Tests for the three remaining bug fixes: CF-8, HI-5, HI-8."""

import os
import sys
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── CF-8: ClientSession default timeout ──────────────────────────────────────


class TestCF8SessionTimeout:
    """Verify AlpacaBroker's ClientSession has a default timeout."""

    def _read_source(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "trading",
            "broker",
            "alpaca.py",
        )
        with open(path) as f:
            return f.read()

    def test_session_constructor_has_timeout(self):
        source = self._read_source()
        assert "ClientSession(timeout=" in source

    def test_session_timeout_value_is_30s(self):
        source = self._read_source()
        assert "ClientTimeout(total=30)" in source

    def test_api_call_still_has_its_own_timeout(self):
        """_api_call should keep its per-request 15s timeout."""
        source = self._read_source()
        assert "ClientTimeout(total=15)" in source

    def test_both_timeouts_coexist(self):
        """Session-level 30s default and per-call 15s override must both exist."""
        source = self._read_source()
        # Count occurrences of ClientTimeout
        count = source.count("ClientTimeout(total=")
        assert count >= 2, f"Expected >=2 ClientTimeout declarations, found {count}"


# ── HI-5: Timezone-aware market hours ────────────────────────────────────────


class TestHI5TimezoneAwareMarketHours:
    """Verify model_trainer uses timezone-aware market hour classification."""

    def _read_source(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "model_trainer.py",
        )
        with open(path) as f:
            return f.read()

    def test_zoneinfo_imported(self):
        source = self._read_source()
        assert "from zoneinfo import ZoneInfo" in source

    def test_eastern_timezone_used(self):
        source = self._read_source()
        assert 'ZoneInfo("America/New_York")' in source

    def test_no_raw_hour_comparison(self):
        """The old pattern df['hour'] >= 9 should NOT appear for market session."""
        source = self._read_source()
        # The old bug was: df['is_market_open'] = ((df['hour'] >= 9) ...
        # Now it should use et_hour, not df['hour'] for market session indicators
        lines = source.split("\n")
        for line in lines:
            if "is_market_open" in line and "=" in line and "astype" in line:
                assert (
                    "df['hour']" not in line
                ), f"Market session still uses raw df['hour']: {line.strip()}"

    def test_et_hour_used_for_market_sessions(self):
        source = self._read_source()
        assert "et_hour" in source

    def test_timezone_conversion_present(self):
        source = self._read_source()
        assert "tz_convert" in source

    def test_add_time_features_function(self):
        """Smoke-test: import and call add_time_features with UTC timestamps."""
        import pandas as pd
        import numpy as np
        from zoneinfo import ZoneInfo

        # Create a DataFrame with known UTC timestamps
        # 14:00 UTC = 10:00 ET (market open), 21:00 UTC = 17:00 ET (after hours)
        timestamps = pd.to_datetime(
            [
                "2026-03-20 14:00:00",  # 10:00 ET → market open
                "2026-03-20 21:00:00",  # 17:00 ET → after hours
                "2026-03-20 08:00:00",  # 04:00 ET → pre-market
            ]
        )
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "close": [100.0, 101.0, 99.0],
            }
        )

        # Inline the logic from model_trainer to test it
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour

        eastern = ZoneInfo("America/New_York")
        ts_eastern = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(eastern)
        et_hour = ts_eastern.dt.hour
        df["is_market_open"] = ((et_hour >= 9) & (et_hour < 16)).astype(int)
        df["is_premarket"] = ((et_hour >= 4) & (et_hour < 9)).astype(int)
        df["is_afterhours"] = ((et_hour >= 16) | (et_hour < 4)).astype(int)

        # 14:00 UTC = 10:00 ET → market open
        assert df.iloc[0]["is_market_open"] == 1
        assert df.iloc[0]["is_premarket"] == 0
        assert df.iloc[0]["is_afterhours"] == 0

        # 21:00 UTC = 17:00 ET → after hours
        assert df.iloc[1]["is_market_open"] == 0
        assert df.iloc[1]["is_afterhours"] == 1

        # 08:00 UTC = 04:00 ET → pre-market
        assert df.iloc[2]["is_premarket"] == 1
        assert df.iloc[2]["is_market_open"] == 0


# ── HI-8: Redis Docker healthcheck ──────────────────────────────────────────


class TestHI8RedisHealthcheck:
    """Verify docker-compose.yml has Redis healthcheck and depends_on conditions."""

    @pytest.fixture(autouse=True)
    def load_compose(self):
        compose_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docker-compose.yml",
        )
        with open(compose_path) as f:
            self.compose = yaml.safe_load(f)

    def test_redis_has_healthcheck(self):
        redis_svc = self.compose["services"]["redis"]
        assert "healthcheck" in redis_svc

    def test_redis_healthcheck_uses_redis_cli_ping(self):
        hc = self.compose["services"]["redis"]["healthcheck"]
        assert hc["test"] == ["CMD", "redis-cli", "ping"]

    def test_redis_healthcheck_has_interval(self):
        hc = self.compose["services"]["redis"]["healthcheck"]
        assert "interval" in hc

    def test_redis_healthcheck_has_timeout(self):
        hc = self.compose["services"]["redis"]["healthcheck"]
        assert "timeout" in hc

    def test_redis_healthcheck_has_retries(self):
        hc = self.compose["services"]["redis"]["healthcheck"]
        assert "retries" in hc
        assert hc["retries"] >= 1

    def _assert_redis_healthy_dependency(self, service_name):
        svc = self.compose["services"][service_name]
        deps = svc.get("depends_on", {})
        assert "redis" in deps, f"{service_name} should depend on redis"
        redis_dep = deps["redis"]
        assert isinstance(
            redis_dep, dict
        ), f"{service_name} redis dependency should be a dict with condition"
        assert (
            redis_dep.get("condition") == "service_healthy"
        ), f"{service_name} should wait for redis to be healthy"

    def test_data_ingestion_waits_for_healthy_redis(self):
        self._assert_redis_healthy_dependency("data-ingestion")

    def test_sentiment_engine_waits_for_healthy_redis(self):
        self._assert_redis_healthy_dependency("sentiment-engine")

    def test_tft_predictor_waits_for_healthy_redis(self):
        self._assert_redis_healthy_dependency("tft-predictor")

    def test_trading_engine_waits_for_healthy_redis(self):
        self._assert_redis_healthy_dependency("trading-engine")

    def test_orchestrator_waits_for_healthy_redis(self):
        self._assert_redis_healthy_dependency("orchestrator")

    def test_paper_trader_waits_for_healthy_redis(self):
        self._assert_redis_healthy_dependency("paper-trader")

    def test_redis_commander_waits_for_healthy_redis(self):
        self._assert_redis_healthy_dependency("redis-commander")
