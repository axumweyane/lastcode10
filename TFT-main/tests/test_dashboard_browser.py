"""
Dashboard browser tests using Playwright with real Chromium.
Loads http://localhost:8010/dashboard and validates the rendered page.
"""

import os
import pytest
from playwright.sync_api import sync_playwright

DASHBOARD_URL = "http://localhost:8010/dashboard"
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
SCREENSHOT_PATH = os.path.join(SCREENSHOT_DIR, "dashboard.png")


@pytest.fixture(scope="module")
def page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        pg = browser.new_page()
        pg.goto(DASHBOARD_URL, wait_until="networkidle", timeout=15000)
        yield pg
        browser.close()


class TestDashboardLoads:
    def test_page_responds_200(self):
        """Dashboard URL returns HTTP 200."""
        with sync_playwright() as p:
            ctx = p.request.new_context()
            resp = ctx.get(DASHBOARD_URL)
            assert resp.status == 200
            ctx.dispose()

    def test_no_navigation_error(self, page):
        """Page loaded without Playwright navigation errors."""
        # If goto failed, the fixture would have raised. This confirms the URL resolved.
        assert page.url.rstrip("/").endswith("/dashboard")


class TestDashboardContent:
    def test_page_has_meaningful_content(self, page):
        """Page body is not empty or trivially short."""
        body_text = page.text_content("body") or ""
        assert (
            len(body_text.strip()) > 50
        ), f"Dashboard body too short ({len(body_text)} chars)"

    def test_page_has_html_elements(self, page):
        """Page contains real HTML structure, not a blank page."""
        tags = page.query_selector_all("div, table, h1, h2, h3, p, span")
        assert len(tags) > 3, "Dashboard has too few HTML elements"


class TestDashboardNoErrors:
    def test_no_internal_server_error(self, page):
        body = page.text_content("body") or ""
        assert "Internal Server Error" not in body

    def test_no_python_traceback(self, page):
        body = page.text_content("body") or ""
        assert "Traceback (most recent call last)" not in body

    def test_no_500_error_text(self, page):
        body = page.text_content("body") or ""
        assert "500 Internal" not in body

    def test_no_exception_text(self, page):
        body = page.text_content("body") or ""
        assert "Exception:" not in body


class TestDashboardStructure:
    def test_has_title_or_heading(self, page):
        """Page has a <title> or visible heading."""
        title = page.title() or ""
        h1 = page.query_selector("h1")
        h2 = page.query_selector("h2")
        has_title = len(title.strip()) > 0
        has_heading = h1 is not None or h2 is not None
        assert has_title or has_heading, "No page title or heading found"


class TestDashboardScreenshot:
    def test_screenshot_saved(self, page):
        """Take a full-page screenshot and save to tests/screenshots/dashboard.png."""
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        page.screenshot(path=SCREENSHOT_PATH, full_page=True)
        assert os.path.exists(SCREENSHOT_PATH), "Screenshot file not created"
        size = os.path.getsize(SCREENSHOT_PATH)
        assert size > 1000, f"Screenshot too small ({size} bytes), page may be blank"
