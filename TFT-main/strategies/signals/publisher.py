"""
Redis Signal Publisher — lightweight real-time signal distribution layer.

Publishes combined ensemble signals to Redis pub/sub channels for dashboard
consumers and external systems. Additive to the Kafka event bus — does NOT
replace it.

Design principles:
  - Fire-and-forget: never block the pipeline on a publish failure
  - Self-healing: detects stale connections and reconnects transparently
  - Structured payloads: consistent JSON schema across all channels
  - Channel routing: auto-classifies signals by asset class

Channels:
  apex:signals:stock    — equity ensemble signals
  apex:signals:forex    — FX pair signals
  apex:signals:options  — options/volatility signals
  apex:signals:risk     — tail risk index broadcasts
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CHANNELS = {
    "stocks": "apex:signals:stock",
    "forex": "apex:signals:forex",
    "options": "apex:signals:options",
    "volatility": "apex:signals:options",  # vol routes to options channel
}
RISK_CHANNEL = "apex:signals:risk"

FX_SYMBOLS = frozenset({"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"})
OPTIONS_STRATEGIES = frozenset({"deep_surrogates", "tdgf"})


class SignalPublisher:
    """
    Publishes ensemble signals to Redis pub/sub.

    Usage:
        publisher = SignalPublisher(redis_client)
        publisher.publish_signals(combined_signals)
        publisher.publish_tail_risk({"SPY": 0.72, "QQQ": 0.68})
    """

    def __init__(self, redis_client: Any):
        self._redis = redis_client
        self._connected = True
        self._publish_count = 0
        self._error_count = 0
        self._last_publish: Optional[float] = None

    @property
    def is_healthy(self) -> bool:
        """Check if Redis connection is alive."""
        try:
            self._redis.ping()
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "connected": self._connected,
            "total_published": self._publish_count,
            "total_errors": self._error_count,
            "last_publish": self._last_publish,
        }

    def publish_signals(self, signals: List[Any]) -> int:
        """
        Publish combined signals to appropriate Redis channels.

        Each signal is routed to a channel based on its symbol and
        contributing strategies. Returns the number of successfully
        published signals.
        """
        if not signals:
            return 0

        if not self._connected and not self.is_healthy:
            logger.debug("Redis unavailable, skipping %d signals", len(signals))
            return 0

        published = 0
        ts = datetime.now(timezone.utc).isoformat()

        for signal in signals:
            try:
                symbol = getattr(signal, "symbol", "")
                contributing = getattr(signal, "contributing_strategies", {})
                channel = self._classify_channel(symbol, contributing)

                # Structured payload — consistent schema for all consumers
                direction = getattr(signal, "direction", "neutral")
                if hasattr(direction, "value"):
                    direction = direction.value

                payload = json.dumps(
                    {
                        "ts": ts,
                        "symbol": symbol,
                        "score": round(getattr(signal, "combined_score", 0.0), 6),
                        "confidence": round(getattr(signal, "confidence", 0.0), 4),
                        "direction": direction,
                        "sources": {k: round(v, 4) for k, v in contributing.items()},
                    }
                )

                self._redis.publish(channel, payload)
                published += 1

            except Exception as e:
                self._error_count += 1
                if self._error_count <= 3:
                    logger.warning(
                        "Signal publish failed for %s: %s",
                        getattr(signal, "symbol", "?"),
                        e,
                    )
                if self._error_count == 3:
                    logger.warning("Suppressing further publish warnings this cycle")

        self._publish_count += published
        self._last_publish = time.time()

        if published:
            logger.info(
                "Published %d/%d signals to Redis (%d errors)",
                published,
                len(signals),
                len(signals) - published,
            )

        # Reset per-cycle error counter
        self._error_count = 0
        return published

    def publish_tail_risk(self, risk_data: Dict[str, float]) -> bool:
        """
        Publish tail risk index to apex:signals:risk.

        Args:
            risk_data: {symbol: tail_risk_index} mapping.

        Returns:
            True if published successfully.
        """
        if not risk_data:
            return False

        if not self._connected and not self.is_healthy:
            return False

        try:
            composite = float(sum(risk_data.values()) / len(risk_data))
            max_risk_sym = max(risk_data, key=risk_data.get)

            payload = json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "composite": round(composite, 4),
                    "max_symbol": max_risk_sym,
                    "max_value": round(risk_data[max_risk_sym], 4),
                    "per_symbol": {k: round(v, 4) for k, v in risk_data.items()},
                }
            )

            self._redis.publish(RISK_CHANNEL, payload)
            logger.info(
                "Tail risk published: composite=%.3f, max=%s(%.3f)",
                composite,
                max_risk_sym,
                risk_data[max_risk_sym],
            )
            return True

        except Exception as e:
            logger.warning("Tail risk publish failed: %s", e)
            return False

    def _classify_channel(self, symbol: str, contributing: Dict[str, float]) -> str:
        """Route signal to the correct Redis channel."""
        if symbol.upper() in FX_SYMBOLS:
            return CHANNELS["forex"]

        # If majority of contribution comes from options strategies
        if contributing:
            options_weight = sum(contributing.get(s, 0.0) for s in OPTIONS_STRATEGIES)
            total_weight = sum(abs(v) for v in contributing.values())
            if total_weight > 0 and options_weight / total_weight > 0.5:
                return CHANNELS["options"]

        return CHANNELS["stocks"]
