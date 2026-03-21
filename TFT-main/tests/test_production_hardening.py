"""Tests for production hardening: Kafka retention, TimescaleDB policies, schema registry cache."""

import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import yaml
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── 1. Kafka retention config in docker-compose.yml ─────────────────────────

class TestKafkaRetention:
    """Verify Kafka broker service has correct retention config."""

    @pytest.fixture(autouse=True)
    def load_compose(self):
        with open(os.path.join(BASE_DIR, "docker-compose.yml")) as f:
            self.compose = yaml.safe_load(f)

    def test_kafka_broker_service_exists(self):
        assert "kafka-broker" in self.compose["services"]

    def test_retention_hours(self):
        env = self.compose["services"]["kafka-broker"]["environment"]
        assert env["KAFKA_LOG_RETENTION_HOURS"] == 168

    def test_retention_bytes(self):
        env = self.compose["services"]["kafka-broker"]["environment"]
        assert env["KAFKA_LOG_RETENTION_BYTES"] == 5368709120  # 5 GB

    def test_segment_bytes(self):
        env = self.compose["services"]["kafka-broker"]["environment"]
        assert env["KAFKA_LOG_SEGMENT_BYTES"] == 1073741824  # 1 GB

    def test_kafka_broker_has_healthcheck(self):
        assert "healthcheck" in self.compose["services"]["kafka-broker"]

    def test_kafka_data_volume_defined(self):
        assert "kafka_data" in self.compose["volumes"]

    def test_schema_registry_service_exists(self):
        assert "schema-registry" in self.compose["services"]

    def test_schema_registry_depends_on_kafka(self):
        deps = self.compose["services"]["schema-registry"]["depends_on"]
        assert "kafka-broker" in deps

    def test_microservices_depend_on_kafka(self):
        for svc in ["data-ingestion", "sentiment-engine", "tft-predictor",
                     "trading-engine", "orchestrator"]:
            deps = self.compose["services"][svc]["depends_on"]
            assert "kafka-broker" in deps, f"{svc} should depend on kafka-broker"

    def test_microservices_have_schema_registry_url(self):
        for svc in ["data-ingestion", "sentiment-engine", "tft-predictor",
                     "trading-engine", "orchestrator"]:
            env = self.compose["services"][svc]["environment"]
            sr_vars = [e for e in env if "SCHEMA_REGISTRY_URL" in str(e)]
            assert len(sr_vars) == 1, f"{svc} should have SCHEMA_REGISTRY_URL"

    def test_kafka_ui_points_to_internal_broker(self):
        env = self.compose["services"]["kafka-ui"]["environment"]
        assert env["KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS"] == "kafka-broker:9092"

    def test_kafka_ui_has_schema_registry(self):
        env = self.compose["services"]["kafka-ui"]["environment"]
        assert "KAFKA_CLUSTERS_0_SCHEMAREGISTRY" in env


# ── 2. TimescaleDB retention policies in postgres_schema.py ──────────────────

class TestTimescaleDBSchema:
    """Verify TimescaleDB hypertables, retention, and continuous aggregates in schema SQL."""

    @pytest.fixture(autouse=True)
    def load_schema(self):
        from postgres_schema import CREATE_SCHEMA_SQL
        self.sql = CREATE_SCHEMA_SQL

    def test_timescaledb_extension(self):
        assert "CREATE EXTENSION IF NOT EXISTS timescaledb" in self.sql

    def test_ohlcv_hypertable(self):
        assert "create_hypertable('ohlcv'" in self.sql

    def test_paper_risk_reports_hypertable(self):
        assert "create_hypertable('paper_risk_reports'" in self.sql

    def test_paper_execution_stats_hypertable(self):
        assert "create_hypertable('paper_execution_stats'" in self.sql

    def test_paper_signal_analyses_hypertable(self):
        assert "create_hypertable('paper_signal_analyses'" in self.sql

    def test_hypertable_idempotent(self):
        assert self.sql.count("IF NOT EXISTS") > 4  # tables + hypertable checks

    def test_hypertable_migrate_data(self):
        assert self.sql.count("migrate_data => true") == 4

    def test_ohlcv_retention_365d(self):
        assert "add_retention_policy('ohlcv', INTERVAL '365 days'" in self.sql

    def test_risk_reports_retention_90d(self):
        assert "add_retention_policy('paper_risk_reports', INTERVAL '90 days'" in self.sql

    def test_execution_stats_retention_90d(self):
        assert "add_retention_policy('paper_execution_stats', INTERVAL '90 days'" in self.sql

    def test_signal_analyses_retention_90d(self):
        assert "add_retention_policy('paper_signal_analyses', INTERVAL '90 days'" in self.sql

    def test_retention_idempotent(self):
        assert self.sql.count("if_not_exists => true") >= 4

    def test_continuous_aggregate_15m(self):
        assert "ohlcv_15m" in self.sql
        assert "time_bucket('15 minutes'" in self.sql

    def test_continuous_aggregate_1h(self):
        assert "ohlcv_1h" in self.sql
        assert "time_bucket('1 hour'" in self.sql

    def test_continuous_aggregate_1d(self):
        assert "ohlcv_1d" in self.sql
        assert "time_bucket('1 day'" in self.sql

    def test_continuous_aggregate_policies(self):
        assert "add_continuous_aggregate_policy('ohlcv_15m'" in self.sql
        assert "add_continuous_aggregate_policy('ohlcv_1h'" in self.sql
        assert "add_continuous_aggregate_policy('ohlcv_1d'" in self.sql

    def test_aggregates_have_ohlcv_columns(self):
        for view in ["ohlcv_15m", "ohlcv_1h", "ohlcv_1d"]:
            # Find the section for this view
            idx = self.sql.index(view)
            section = self.sql[idx:idx + 500]
            for col in ["open", "high", "low", "close", "volume"]:
                assert col in section, f"{view} should aggregate {col}"

    def test_aggregates_with_no_data(self):
        assert self.sql.count("WITH NO DATA") == 3


# ── 3. Schema registry connection cache ──────────────────────────────────────

class TestSchemaRegistryCache:
    """Verify singleton pattern and exponential backoff retry."""

    @pytest.fixture(autouse=True)
    def reset(self):
        sys.path.insert(0, os.path.join(BASE_DIR, "microservices"))
        import microservices.schema_registry as sr
        sr.reset_client()
        yield
        sr.reset_client()

    def test_singleton_returns_same_instance(self):
        import microservices.schema_registry as sr
        with patch.object(sr, "SchemaRegistryClient") as MockClient:
            mock = MagicMock()
            MockClient.return_value = mock
            c1 = sr.get_schema_registry_client("http://fake:8081")
            c2 = sr.get_schema_registry_client("http://fake:8081")
            assert c1 is c2
            assert MockClient.call_count == 1

    def test_different_url_creates_new_instance(self):
        import microservices.schema_registry as sr
        mock1 = MagicMock()
        mock2 = MagicMock()
        with patch.object(sr, "SchemaRegistryClient", side_effect=[mock1, mock2]):
            c1 = sr.get_schema_registry_client("http://fake1:8081")
            c2 = sr.get_schema_registry_client("http://fake2:8081")
            assert c1 is mock1
            assert c2 is mock2
            assert c1 is not c2

    def test_retry_with_exponential_backoff(self):
        import microservices.schema_registry as sr
        call_times = []

        def failing_init(url):
            call_times.append(time.monotonic())
            raise ConnectionError("connection refused")

        with patch.object(sr, "SchemaRegistryClient", side_effect=failing_init):
            with pytest.raises(ConnectionError, match="after 3 attempts"):
                sr.get_schema_registry_client(
                    "http://fail:8081", max_retries=3, base_backoff_s=0.05
                )

        assert len(call_times) == 3
        # Second gap should be roughly 2x the first (exponential backoff)
        gap1 = call_times[1] - call_times[0]
        gap2 = call_times[2] - call_times[1]
        assert gap2 > gap1 * 1.5  # Allow some tolerance

    def test_retry_succeeds_on_second_attempt(self):
        import microservices.schema_registry as sr
        attempt = {"count": 0}

        def flaky_init(url):
            attempt["count"] += 1
            if attempt["count"] < 2:
                raise ConnectionError("temporary failure")
            return MagicMock()

        with patch.object(sr, "SchemaRegistryClient", side_effect=flaky_init):
            client = sr.get_schema_registry_client(
                "http://flaky:8081", max_retries=3, base_backoff_s=0.01
            )
            assert client is not None

    def test_reset_client_clears_cache(self):
        import microservices.schema_registry as sr
        with patch.object(sr, "SchemaRegistryClient") as MockClient:
            mock = MagicMock()
            MockClient.return_value = mock
            sr.get_schema_registry_client("http://fake:8081")
            sr.reset_client()
            assert sr._instance is None

    def test_thread_safety(self):
        import microservices.schema_registry as sr
        results = []

        def get_client():
            with patch.object(sr, "SchemaRegistryClient") as MockClient:
                MockClient.return_value = MagicMock()
                c = sr.get_schema_registry_client("http://thread:8081")
                results.append(id(c))

        # First call to set the singleton
        with patch.object(sr, "SchemaRegistryClient") as MockClient:
            MockClient.return_value = MagicMock()
            sr.get_schema_registry_client("http://thread:8081")

        threads = [threading.Thread(target=get_client) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same cached instance
        assert len(set(results)) == 1


class TestSchemaRegistryClientAPI:
    """Test the SchemaRegistryClient methods."""

    def test_get_schema_calls_correct_url(self):
        with patch("requests.Session") as MockSession:
            mock_session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"schema": "{}"}
            mock_session.get.return_value = mock_resp
            MockSession.return_value = mock_session

            sys.path.insert(0, os.path.join(BASE_DIR, "microservices"))
            from microservices.schema_registry import SchemaRegistryClient
            client = SchemaRegistryClient("http://test:8081")
            client.get_schema("my-subject")

            calls = [str(c) for c in mock_session.get.call_args_list]
            assert any("my-subject" in c for c in calls)

    def test_get_subjects(self):
        with patch("requests.Session") as MockSession:
            mock_session = MagicMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = ["subject1", "subject2"]
            mock_session.get.return_value = mock_resp
            MockSession.return_value = mock_session

            from microservices.schema_registry import SchemaRegistryClient
            client = SchemaRegistryClient("http://test:8081")
            subjects = client.get_subjects()
            assert subjects == ["subject1", "subject2"]
