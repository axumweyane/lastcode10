"""
Load test for APEX Paper Trader.

Run:
  locust -f tests/load/locustfile.py --host http://localhost:8010 \
         --users 20 --spawn-rate 5 --run-time 30s --headless
"""

from locust import HttpUser, task, between


class HealthCheckUser(HttpUser):
    """Simulates dashboard/monitoring traffic."""

    weight = 3
    wait_time = between(1, 3)

    @task(10)
    def health(self):
        self.client.get("/health")

    @task(5)
    def dashboard(self):
        self.client.get("/dashboard")

    @task(3)
    def weights(self):
        self.client.get("/weights")

    @task(3)
    def history(self):
        self.client.get("/history")

    @task(2)
    def dlq(self):
        self.client.get("/dlq")


class SignalAPIUser(HttpUser):
    """Simulates external signal API consumers."""

    weight = 1
    wait_time = between(2, 5)

    @task(5)
    def signals(self):
        self.client.get("/api/v1/signals")

    @task(3)
    def regime(self):
        self.client.get("/api/v1/signals/regime")

    @task(2)
    def signal_weights(self):
        self.client.get("/api/v1/signals/weights")
