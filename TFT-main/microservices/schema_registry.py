"""
Schema Registry connection cache with singleton pattern and exponential backoff retry.

Usage:
    from schema_registry import get_schema_registry_client
    client = get_schema_registry_client("http://schema-registry:8081")
"""

import logging
import os
import threading
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

SCHEMA_REGISTRY_URL = os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081")
MAX_RETRIES = 3
BASE_BACKOFF_S = 1.0


class SchemaRegistryClient:
    """Lightweight schema registry client with health check."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self._session = requests.Session()
        self._verify_connection()

    def _verify_connection(self):
        resp = self._session.get(f"{self.url}/subjects", timeout=5)
        resp.raise_for_status()

    def get_schema(self, subject: str, version: str = "latest") -> dict:
        resp = self._session.get(
            f"{self.url}/subjects/{subject}/versions/{version}", timeout=5
        )
        resp.raise_for_status()
        return resp.json()

    def register_schema(self, subject: str, schema: dict) -> int:
        resp = self._session.post(
            f"{self.url}/subjects/{subject}/versions",
            json={"schema": schema if isinstance(schema, str) else str(schema)},
            headers={"Content-Type": "application/vnd.schemaregistry.v1+json"},
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def get_subjects(self) -> list:
        resp = self._session.get(f"{self.url}/subjects", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._session.close()


_lock = threading.Lock()
_instance: Optional[SchemaRegistryClient] = None
_instance_url: Optional[str] = None


def get_schema_registry_client(
    url: Optional[str] = None,
    max_retries: int = MAX_RETRIES,
    base_backoff_s: float = BASE_BACKOFF_S,
) -> SchemaRegistryClient:
    """Return a cached singleton SchemaRegistryClient, creating one with retry if needed."""
    global _instance, _instance_url
    target_url = url or SCHEMA_REGISTRY_URL

    if _instance is not None and _instance_url == target_url:
        return _instance

    with _lock:
        # Double-check after acquiring lock
        if _instance is not None and _instance_url == target_url:
            return _instance

        # Close stale instance if URL changed
        if _instance is not None:
            try:
                _instance.close()
            except Exception:
                pass

        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                client = SchemaRegistryClient(target_url)
                _instance = client
                _instance_url = target_url
                logger.info(
                    "Schema registry connected at %s (attempt %d)", target_url, attempt
                )
                return client
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    backoff = base_backoff_s * (2 ** (attempt - 1))
                    logger.warning(
                        "Schema registry connection failed (attempt %d/%d), "
                        "retrying in %.1fs: %s",
                        attempt,
                        max_retries,
                        backoff,
                        exc,
                    )
                    time.sleep(backoff)

        raise ConnectionError(
            f"Failed to connect to schema registry at {target_url} "
            f"after {max_retries} attempts: {last_exc}"
        )


def reset_client():
    """Reset the cached client (useful for testing or reconnection)."""
    global _instance, _instance_url
    with _lock:
        if _instance is not None:
            try:
                _instance.close()
            except Exception:
                pass
        _instance = None
        _instance_url = None
