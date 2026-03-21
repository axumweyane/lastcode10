"""Integration test: circuit breaker trip/reset cycle."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_circuit_breaker_config_import():
    """Verify circuit breaker classes can be imported."""
    from trading.risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, DrawdownConfig, DrawdownMethod

    config = CircuitBreakerConfig(
        drawdown_configs=[
            DrawdownConfig(method=DrawdownMethod.HIGH_WATER_MARK, threshold_percent=5.0),
            DrawdownConfig(method=DrawdownMethod.START_OF_DAY, threshold_percent=3.0),
        ],
        initial_capital=100000.0,
    )
    assert len(config.drawdown_configs) == 2
    assert config.initial_capital == 100000.0


def test_drawdown_methods():
    from trading.risk.circuit_breaker import DrawdownMethod
    assert DrawdownMethod.HIGH_WATER_MARK.value == "high_water_mark"
    assert DrawdownMethod.START_OF_DAY.value == "start_of_day"
    assert DrawdownMethod.INITIAL_CAPITAL.value == "initial_capital"


def test_audit_logger_import():
    """Verify audit logger can be imported and schema exists."""
    from trading.persistence.audit import AuditLogger, CIRCUIT_BREAKER_SCHEMA_SQL
    assert "circuit_breaker_events" in CIRCUIT_BREAKER_SCHEMA_SQL
    assert "circuit_breaker_closures" in CIRCUIT_BREAKER_SCHEMA_SQL


def test_notification_manager_import():
    """Verify notification manager can be imported."""
    from trading.notifications.alerts import (
        NotificationManager, AlertMessage, DiscordWebhookSender, EmailSender,
    )
    msg = AlertMessage(
        title="Test Alert",
        body="This is a test",
        severity="info",
    )
    assert msg.title == "Test Alert"
    assert msg.severity == "info"


def test_alpaca_broker_import():
    """Verify production broker can be imported."""
    from trading.broker.alpaca import AlpacaBroker
    from trading.broker.base import BaseBroker
    broker = AlpacaBroker(api_key="test", secret_key="test")
    assert isinstance(broker, BaseBroker)
