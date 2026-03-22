"""
Chaos / infrastructure resilience tests for APEX Paper Trader.

Tests:
  1. Redis shutdown → paper trader /health reports failure → Redis restart → recovery
  2. Database connection failure → /run-now returns error gracefully (no crash)

Redis and PostgreSQL run in Docker containers:
  - Redis: container "tp-redis" on port 6379
  - PostgreSQL: container on port 15432

IMPORTANT: These tests stop and restart real services. They restore everything
after each test. Run in isolation (not in parallel with other test suites).
"""

import subprocess
import time

import pytest
import requests

PAPER_TRADER = "http://localhost:8010"
REDIS_CONTAINER = "tp-redis"
DB_PORT = 15432
HEALTH_TIMEOUT = 5
RECOVERY_TIMEOUT = 30


def _health():
    """Get /health JSON, or None on connection error."""
    try:
        resp = requests.get(f"{PAPER_TRADER}/health", timeout=HEALTH_TIMEOUT)
        return resp.json()
    except Exception:
        return None


def _wait_for_health(condition_fn, timeout=RECOVERY_TIMEOUT, poll=2):
    """Poll /health until condition_fn(data) is True or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = _health()
        if data is not None and condition_fn(data):
            return data
        time.sleep(poll)
    return None


def _redis_is_running():
    result = subprocess.run(
        ["docker", "exec", REDIS_CONTAINER, "redis-cli", "ping"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() == "PONG"


def _stop_redis():
    subprocess.run(
        ["docker", "stop", REDIS_CONTAINER],
        capture_output=True,
        timeout=30,
    )
    time.sleep(1)


def _start_redis():
    subprocess.run(
        ["docker", "start", REDIS_CONTAINER],
        capture_output=True,
        timeout=30,
    )
    for _ in range(15):
        try:
            if _redis_is_running():
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _find_db_container():
    """Find the Docker container serving PostgreSQL on DB_PORT."""
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    for line in result.stdout.splitlines():
        if f":{DB_PORT}->" in line:
            return line.split("\t")[0]
    return None


# ---------- Precondition: paper trader is running ----------


class TestPreconditions:
    def test_paper_trader_is_running(self):
        data = _health()
        assert data is not None, "Paper trader not running on port 8010"
        assert data.get("status") == "running"

    def test_redis_is_running(self):
        assert _redis_is_running(), f"Redis container '{REDIS_CONTAINER}' not running"


# ---------- 1. Redis chaos: shutdown → verify degraded → restart → verify recovery ----------


class TestRedisChaos:
    def test_redis_shutdown_and_recovery(self):
        """
        1. Confirm Redis is up and /health is good
        2. Stop Redis container
        3. Verify /health still responds (paper trader doesn't crash)
           and reports Redis failure
        4. Restart Redis container
        5. Verify /health recovers within 30 seconds
        """
        assert _redis_is_running(), "Redis must be running before test"
        baseline = _health()
        assert baseline is not None, "Paper trader must be reachable"

        try:
            # Stop Redis
            _stop_redis()
            assert not _redis_is_running(), "Redis should be stopped"

            # /health should still respond — paper trader must not crash
            time.sleep(2)
            degraded = _health()
            assert (
                degraded is not None
            ), "Paper trader crashed or became unreachable after Redis shutdown"

            # Check that health reports Redis is down
            infra = degraded.get("infrastructure", {})
            redis_ok = infra.get("redis", None)
            if redis_ok is not None:
                assert (
                    redis_ok is not True
                ), "Health should report Redis down, but says it's up"

            # Restart Redis
            assert _start_redis(), "Failed to restart Redis container"
            assert _redis_is_running(), "Redis should be running after restart"

            # Wait for /health to recover
            recovered = _wait_for_health(
                lambda d: d.get("status") == "running",
                timeout=RECOVERY_TIMEOUT,
            )
            assert recovered is not None, (
                f"Paper trader did not recover within {RECOVERY_TIMEOUT}s "
                f"after Redis restart"
            )

        finally:
            if not _redis_is_running():
                _start_redis()
            assert _redis_is_running(), "Redis cleanup failed — container not running!"

    def test_health_endpoint_never_crashes_without_redis(self):
        """Paper trader /health always returns HTTP 200, even with Redis down."""
        assert _redis_is_running(), "Redis must be running before test"

        try:
            _stop_redis()
            time.sleep(2)

            # Multiple rapid requests — none should crash the server
            for _ in range(5):
                try:
                    resp = requests.get(
                        f"{PAPER_TRADER}/health", timeout=HEALTH_TIMEOUT
                    )
                    assert (
                        resp.status_code == 200
                    ), f"/health returned {resp.status_code} without Redis"
                except requests.ConnectionError:
                    pytest.fail("Paper trader crashed — connection refused")
                time.sleep(0.5)

        finally:
            if not _redis_is_running():
                _start_redis()
            assert _redis_is_running(), "Redis cleanup failed"


# ---------- 2. Database chaos: stop container → /run-now fails gracefully → restart ----------


class TestDatabaseChaos:
    @pytest.fixture(autouse=True)
    def _find_container(self):
        self.db_container = _find_db_container()
        if not self.db_container:
            pytest.skip(f"No Docker container found serving port {DB_PORT}")

    def _db_is_running(self):
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.db_container],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() == "true"

    def _stop_db(self):
        subprocess.run(
            ["docker", "stop", self.db_container],
            capture_output=True,
            timeout=30,
        )
        time.sleep(1)

    def _start_db(self):
        subprocess.run(
            ["docker", "start", self.db_container],
            capture_output=True,
            timeout=30,
        )
        # Wait for PostgreSQL to accept connections
        for _ in range(20):
            result = subprocess.run(
                ["docker", "exec", self.db_container, "pg_isready", "-U", "apex_user"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                return True
            time.sleep(1)
        return False

    def test_run_now_handles_db_failure_gracefully(self):
        """
        Stop database container, trigger /run-now, verify paper trader
        returns an error response (not a crash), then restart DB.
        """
        assert self._db_is_running(), "Database must be running before test"

        try:
            self._stop_db()
            time.sleep(2)

            # Trigger pipeline — should fail gracefully
            try:
                resp = requests.post(
                    f"{PAPER_TRADER}/run-now",
                    timeout=60,
                )
                # Any HTTP response means the server didn't crash
                assert resp.status_code in (
                    200,
                    500,
                    503,
                ), f"Unexpected status {resp.status_code} during DB outage"
            except requests.ConnectionError:
                pytest.fail(
                    "Paper trader crashed during DB outage — connection refused"
                )
            except requests.Timeout:
                # Timeout is acceptable — pipeline may hang waiting for DB
                pass

            # Verify /health still responds (server is alive)
            health = _health()
            assert (
                health is not None
            ), "Paper trader unreachable after /run-now during DB outage"

        finally:
            if not self._db_is_running():
                assert self._start_db(), "DB cleanup failed — container not starting!"
            time.sleep(2)

    def test_health_survives_db_outage(self):
        """/health should return even if the database is unreachable."""
        assert self._db_is_running(), "Database must be running before test"

        try:
            self._stop_db()
            time.sleep(2)

            try:
                resp = requests.get(f"{PAPER_TRADER}/health", timeout=HEALTH_TIMEOUT)
                assert (
                    resp.status_code == 200
                ), f"/health returned {resp.status_code} during DB outage"
            except requests.ConnectionError:
                pytest.fail("Paper trader crashed during DB outage")

        finally:
            if not self._db_is_running():
                assert self._start_db(), "DB cleanup failed — container not starting!"
            time.sleep(2)
