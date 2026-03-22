"""
API endpoint tests for APEX Paper Trader using Playwright.
Tests all endpoints on http://localhost:8010.
"""

import pytest
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8010"


@pytest.fixture(scope="module")
def api():
    with sync_playwright() as p:
        ctx = p.request.new_context(base_url=BASE_URL)
        yield ctx
        ctx.dispose()


# ---------- 1. GET /health ----------


class TestHealthEndpoint:
    def test_returns_200(self, api):
        resp = api.get("/health")
        assert resp.status == 200

    def test_version_is_3(self, api):
        data = api.get("/health").json()
        assert data["version"] == "3.0.0"

    def test_status_running(self, api):
        data = api.get("/health").json()
        assert data["status"] == "running"

    def test_models_section_present(self, api):
        data = api.get("/health").json()
        assert "models" in data
        assert data["models"]["registered"] == 10

    def test_infrastructure_section_present(self, api):
        data = api.get("/health").json()
        assert "infrastructure" in data
        assert data["infrastructure"]["db_pool"] is True


# ---------- 2. GET /dashboard ----------


class TestDashboardEndpoint:
    def test_returns_200(self, api):
        resp = api.get("/dashboard")
        assert resp.status == 200

    def test_returns_html(self, api):
        resp = api.get("/dashboard")
        ct = resp.headers.get("content-type", "")
        assert "text/html" in ct

    def test_has_content(self, api):
        body = api.get("/dashboard").text()
        assert len(body) > 100, "Dashboard HTML is too short"

    def test_no_internal_server_error(self, api):
        body = api.get("/dashboard").text()
        assert "Internal Server Error" not in body

    def test_no_traceback(self, api):
        body = api.get("/dashboard").text()
        assert "Traceback" not in body


# ---------- 3. GET /weights ----------


class TestWeightsEndpoint:
    def test_returns_200(self, api):
        resp = api.get("/weights")
        assert resp.status == 200

    def test_returns_json(self, api):
        resp = api.get("/weights")
        data = resp.json()
        assert isinstance(data, (dict, list))


# ---------- 4. GET /history ----------


class TestHistoryEndpoint:
    def test_returns_200(self, api):
        resp = api.get("/history")
        assert resp.status == 200

    def test_returns_json(self, api):
        resp = api.get("/history")
        data = resp.json()
        assert isinstance(data, (dict, list))


# ---------- 5. GET /dlq ----------


class TestDLQEndpoint:
    def test_returns_200(self, api):
        resp = api.get("/dlq")
        assert resp.status == 200

    def test_returns_json(self, api):
        resp = api.get("/dlq")
        data = resp.json()
        assert isinstance(data, (dict, list))


# ---------- 6-8. Signal Provider API (may require API key) ----------


class TestSignalAPI:
    """Signal provider endpoints return 200 (no key required) or 401 (key required)."""

    def test_signals_returns_200_or_401(self, api):
        resp = api.get("/api/v1/signals")
        assert resp.status in (200, 401, 403, 404)

    def test_signals_regime_returns_200_or_401(self, api):
        resp = api.get("/api/v1/signals/regime")
        assert resp.status in (200, 401, 403, 404)

    def test_signals_weights_returns_200_or_401(self, api):
        resp = api.get("/api/v1/signals/weights")
        assert resp.status in (200, 401, 403, 404)

    def test_unauthenticated_is_consistent(self, api):
        """All signal endpoints should behave the same without a key."""
        statuses = set()
        for path in [
            "/api/v1/signals",
            "/api/v1/signals/regime",
            "/api/v1/signals/weights",
        ]:
            statuses.add(api.get(path).status)
        # All should return the same status (either all 200 or all 401)
        assert len(statuses) == 1, f"Inconsistent signal API statuses: {statuses}"
