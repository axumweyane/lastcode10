"""Tests for the signal provider REST API."""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from api.signal_provider import create_signal_api, SignalCache, RateLimiter

# ── Fixtures ──────────────────────────────────────────────────────────────────

TEST_API_KEY = "test-key-12345"


def _make_cache() -> SignalCache:
    cache = SignalCache()
    cache.refresh(
        signals=[
            {
                "symbol": "AAPL",
                "combined_score": 1.2,
                "confidence": 0.85,
                "direction": "long",
                "contributing_strategies": {"momentum": 0.5, "tft": 0.7},
            },
            {
                "symbol": "MSFT",
                "combined_score": 0.8,
                "confidence": 0.72,
                "direction": "long",
                "contributing_strategies": {"momentum": 0.3, "pairs": 0.5},
            },
            {
                "symbol": "TSLA",
                "combined_score": -0.9,
                "confidence": 0.65,
                "direction": "short",
                "contributing_strategies": {"mean_reversion": -0.9},
            },
        ],
        weights={"momentum": 0.35, "tft": 0.3, "pairs": 0.2, "mean_reversion": 0.15},
        regime="calm_trending",
        regime_detail={
            "regime": "calm_trending",
            "vix_level": 15.2,
            "is_volatile": False,
            "is_trending": True,
            "confidence": 0.82,
            "exposure_scalar": 1.0,
        },
        bayesian_weights={
            "momentum": 0.38,
            "tft": 0.32,
            "pairs": 0.18,
            "mean_reversion": 0.12,
        },
        bayesian_state=[
            {"strategy_name": "momentum", "alpha": 30.0, "beta": 20.0, "weight": 0.6},
        ],
    )
    return cache


def _fake_db_query(query: str, params: tuple) -> list:
    """Fake DB query returning canned signal history."""
    return [
        (
            "2026-03-20",
            "momentum",
            "AAPL",
            1.1,
            0.8,
            "long",
            {},
            "2026-03-20T10:00:00Z",
        ),
        ("2026-03-19", "tft", "AAPL", 0.9, 0.75, "long", {}, "2026-03-19T10:00:00Z"),
        ("2026-03-18", "pairs", "AAPL", -0.3, 0.6, "short", {}, "2026-03-18T10:00:00Z"),
    ]


def _make_client(cache=None, db_fn=None, rate_limit=100) -> TestClient:
    cache = cache or _make_cache()
    api = create_signal_api(
        api_key=TEST_API_KEY,
        cache=cache,
        db_query_fn=db_fn or _fake_db_query,
        rate_limit=rate_limit,
    )
    return TestClient(api)


def _headers(key=TEST_API_KEY):
    return {"X-API-Key": key}


# ── 1. GET /signals ──────────────────────────────────────────────────────────


class TestGetSignals:
    """Test the /signals endpoint."""

    def test_returns_all_signals(self):
        client = _make_client()
        resp = client.get("/signals", headers=_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["signal_count"] == 3
        assert len(data["signals"]) == 3

    def test_signal_schema(self):
        client = _make_client()
        resp = client.get("/signals", headers=_headers())
        data = resp.json()
        sig = data["signals"][0]
        # All required fields present
        for field in (
            "timestamp",
            "symbol",
            "direction",
            "score",
            "confidence",
            "regime",
            "strategies",
            "metadata",
        ):
            assert field in sig, f"Missing field: {field}"

    def test_includes_timestamp_and_regime(self):
        client = _make_client()
        resp = client.get("/signals", headers=_headers())
        data = resp.json()
        assert data["regime"] == "calm_trending"
        assert data["timestamp"] is not None

    def test_includes_etag_header(self):
        client = _make_client()
        resp = client.get("/signals", headers=_headers())
        assert "etag" in resp.headers

    def test_strategies_breakdown_included(self):
        client = _make_client()
        resp = client.get("/signals", headers=_headers())
        data = resp.json()
        aapl_signal = [s for s in data["signals"] if s["symbol"] == "AAPL"][0]
        assert "momentum" in aapl_signal["strategies"]
        assert "tft" in aapl_signal["strategies"]

    def test_empty_cache(self):
        cache = SignalCache()
        client = _make_client(cache=cache)
        resp = client.get("/signals", headers=_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["signal_count"] == 0


# ── 2. GET /signals/{symbol} ─────────────────────────────────────────────────


class TestGetSignalBySymbol:
    """Test the /signals/{symbol} endpoint."""

    def test_returns_specific_symbol(self):
        client = _make_client()
        resp = client.get("/signals/AAPL", headers=_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "AAPL"
        assert data["direction"] == "long"

    def test_case_insensitive(self):
        client = _make_client()
        resp = client.get("/signals/aapl", headers=_headers())
        assert resp.status_code == 200
        assert resp.json()["symbol"] == "AAPL"

    def test_unknown_symbol_404(self):
        client = _make_client()
        resp = client.get("/signals/UNKNOWN", headers=_headers())
        assert resp.status_code == 404

    def test_includes_strategy_breakdown(self):
        client = _make_client()
        resp = client.get("/signals/TSLA", headers=_headers())
        data = resp.json()
        assert "mean_reversion" in data["strategies"]

    def test_includes_score_and_confidence(self):
        client = _make_client()
        resp = client.get("/signals/MSFT", headers=_headers())
        data = resp.json()
        assert data["score"] == 0.8
        assert data["confidence"] == 0.72


# ── 3. GET /signals/history/{symbol} ─────────────────────────────────────────


class TestSignalHistory:
    """Test the history endpoint."""

    def test_returns_history(self):
        client = _make_client()
        resp = client.get("/signals/history/AAPL?days=7", headers=_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "AAPL"
        assert data["days"] == 7
        assert data["count"] == 3
        assert len(data["signals"]) == 3

    def test_history_signal_schema(self):
        client = _make_client()
        resp = client.get("/signals/history/AAPL", headers=_headers())
        data = resp.json()
        sig = data["signals"][0]
        for field in ("date", "strategy", "symbol", "score", "confidence", "direction"):
            assert field in sig

    def test_days_parameter(self):
        client = _make_client()
        resp = client.get("/signals/history/AAPL?days=30", headers=_headers())
        assert resp.status_code == 200
        assert resp.json()["days"] == 30

    def test_days_max_90(self):
        client = _make_client()
        resp = client.get("/signals/history/AAPL?days=91", headers=_headers())
        assert resp.status_code == 422  # validation error

    def test_no_db_returns_503(self):
        api = create_signal_api(
            api_key=TEST_API_KEY,
            cache=_make_cache(),
            db_query_fn=None,
        )
        client = TestClient(api)
        resp = client.get("/signals/history/AAPL", headers=_headers())
        assert resp.status_code == 503

    def test_db_error_returns_500(self):
        def broken_db(q, p):
            raise RuntimeError("DB down")

        client = _make_client(db_fn=broken_db)
        resp = client.get("/signals/history/AAPL", headers=_headers())
        assert resp.status_code == 500


# ── 4. GET /signals/regime ───────────────────────────────────────────────────


class TestRegime:
    """Test the regime endpoint."""

    def test_returns_regime(self):
        client = _make_client()
        resp = client.get("/signals/regime", headers=_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_regime"] == "calm_trending"

    def test_regime_detail_included(self):
        client = _make_client()
        resp = client.get("/signals/regime", headers=_headers())
        data = resp.json()
        detail = data["detail"]
        assert detail["vix_level"] == 15.2
        assert detail["is_trending"] is True

    def test_regime_with_etag(self):
        client = _make_client()
        resp = client.get("/signals/regime", headers=_headers())
        assert "etag" in resp.headers


# ── 5. GET /signals/weights ──────────────────────────────────────────────────


class TestWeights:
    """Test the weights endpoint."""

    def test_returns_weights(self):
        client = _make_client()
        resp = client.get("/signals/weights", headers=_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "momentum" in data["weights"]
        assert data["weights"]["momentum"] == 0.35

    def test_bayesian_weights_included(self):
        client = _make_client()
        resp = client.get("/signals/weights", headers=_headers())
        data = resp.json()
        assert data["bayesian_enabled"] is True
        assert "bayesian_weights" in data
        assert data["bayesian_weights"]["momentum"] == 0.38

    def test_bayesian_state_included(self):
        client = _make_client()
        resp = client.get("/signals/weights", headers=_headers())
        data = resp.json()
        assert "bayesian_state" in data
        assert len(data["bayesian_state"]) == 1

    def test_no_bayesian_when_disabled(self):
        cache = _make_cache()
        cache.bayesian_weights = None
        cache.bayesian_state = None
        client = _make_client(cache=cache)
        resp = client.get("/signals/weights", headers=_headers())
        data = resp.json()
        assert data["bayesian_enabled"] is False
        assert "bayesian_weights" not in data


# ── 6. Authentication ────────────────────────────────────────────────────────


class TestAuthentication:
    """Test API key authentication."""

    def test_valid_key_allowed(self):
        client = _make_client()
        resp = client.get("/signals", headers={"X-API-Key": TEST_API_KEY})
        assert resp.status_code == 200

    def test_missing_key_401(self):
        client = _make_client()
        resp = client.get("/signals")
        assert resp.status_code == 401
        assert "Missing" in resp.json()["error"]

    def test_invalid_key_403(self):
        client = _make_client()
        resp = client.get("/signals", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 403
        assert "Invalid" in resp.json()["error"]

    def test_empty_key_401(self):
        client = _make_client()
        resp = client.get("/signals", headers={"X-API-Key": ""})
        assert resp.status_code == 401

    def test_auth_required_on_all_endpoints(self):
        client = _make_client()
        endpoints = [
            "/signals",
            "/signals/AAPL",
            "/signals/history/AAPL",
            "/signals/regime",
            "/signals/weights",
        ]
        for ep in endpoints:
            resp = client.get(ep)
            assert resp.status_code == 401, f"{ep} should require auth"


# ── 7. Rate limiting ────────────────────────────────────────────────────────


class TestRateLimiting:
    """Test rate limiting."""

    def test_rate_limit_enforced(self):
        client = _make_client(rate_limit=5)
        for i in range(5):
            resp = client.get("/signals", headers=_headers())
            assert resp.status_code == 200, f"Request {i+1} should succeed"

        # 6th request should be blocked
        resp = client.get("/signals", headers=_headers())
        assert resp.status_code == 429
        assert "Rate limit" in resp.json()["error"]

    def test_rate_limit_headers_present(self):
        client = _make_client(rate_limit=10)
        resp = client.get("/signals", headers=_headers())
        assert "X-RateLimit-Limit" in resp.headers
        assert "X-RateLimit-Remaining" in resp.headers
        assert resp.headers["X-RateLimit-Limit"] == "10"

    def test_rate_limiter_resets_after_window(self):
        limiter = RateLimiter(max_requests=2, window_s=0)
        assert limiter.check("k") is True
        assert limiter.check("k") is True
        # Window is 0s, so it resets immediately
        import time

        time.sleep(0.01)
        assert limiter.check("k") is True

    def test_rate_limiter_remaining(self):
        limiter = RateLimiter(max_requests=5, window_s=60)
        assert limiter.remaining("k") == 5
        limiter.check("k")
        assert limiter.remaining("k") == 4

    def test_different_keys_independent(self):
        limiter = RateLimiter(max_requests=2, window_s=60)
        assert limiter.check("a") is True
        assert limiter.check("a") is True
        assert limiter.check("a") is False
        # Different key still works
        assert limiter.check("b") is True

    def test_429_includes_retry_after(self):
        client = _make_client(rate_limit=1)
        client.get("/signals", headers=_headers())
        resp = client.get("/signals", headers=_headers())
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers


# ── 8. SignalCache ───────────────────────────────────────────────────────────


class TestSignalCache:
    """Test the SignalCache data structure."""

    def test_refresh_updates_etag(self):
        cache = SignalCache()
        assert cache.etag == ""
        cache.refresh(signals=[], weights={}, regime="calm")
        assert cache.etag != ""

    def test_get_signal_found(self):
        cache = _make_cache()
        sig = cache.get_signal("AAPL")
        assert sig is not None
        assert sig["symbol"] == "AAPL"

    def test_get_signal_case_insensitive(self):
        cache = _make_cache()
        assert cache.get_signal("aapl") is not None

    def test_get_signal_not_found(self):
        cache = _make_cache()
        assert cache.get_signal("NOPE") is None

    def test_etag_changes_on_refresh(self):
        cache = SignalCache()
        cache.refresh(signals=[], weights={}, regime="calm")
        etag1 = cache.etag
        import time

        time.sleep(0.01)
        cache.refresh(signals=[{"symbol": "X"}], weights={}, regime="volatile")
        etag2 = cache.etag
        assert etag1 != etag2


# ── 9. Paper-trader structural tests ─────────────────────────────────────────


class TestPaperTraderWiring:
    """Verify signal provider is wired into paper-trader."""

    def _read_source(self):
        main_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "paper-trader",
            "main.py",
        )
        with open(main_path) as f:
            return f.read()

    def test_signal_provider_import(self):
        source = self._read_source()
        assert "from api.signal_provider import create_signal_api" in source

    def test_signal_cache_import(self):
        source = self._read_source()
        assert "SignalCache" in source

    def test_signal_api_enabled_env(self):
        source = self._read_source()
        assert "SIGNAL_API_ENABLED" in source

    def test_signal_api_key_env(self):
        source = self._read_source()
        assert "SIGNAL_API_KEY" in source

    def test_mounted_at_api_v1(self):
        source = self._read_source()
        assert '"/api/v1"' in source

    def test_signal_cache_refreshed_in_pipeline(self):
        source = self._read_source()
        assert "signal_cache.refresh(" in source

    def test_env_template_has_signal_api_vars(self):
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".env.template",
        )
        with open(template_path) as f:
            source = f.read()
        assert "SIGNAL_API_ENABLED" in source
        assert "SIGNAL_API_KEY" in source
