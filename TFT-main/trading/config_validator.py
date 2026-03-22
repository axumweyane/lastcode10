"""
Pre-flight validation for live trading configuration.
Run: python -m trading.config_validator
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from typing import List

import psycopg2
import redis

from trading.broker.alpaca import AlpacaBroker

logger = logging.getLogger(__name__)

PLACEHOLDER_STRINGS = {
    "your_",
    "YOUR_",
    "sk-your",
    "your-",
    "placeholder",
    "changeme",
    "CHANGEME",
    "xxx",
    "XXX",
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    severity: str  # "critical", "warning", "info"


class ConfigValidator:
    def __init__(self):
        self.results: List[CheckResult] = []

    def _add(
        self, name: str, passed: bool, message: str, severity: str = "critical"
    ) -> None:
        self.results.append(CheckResult(name, passed, message, severity))

    def _is_placeholder(self, value: str) -> bool:
        return any(p in value for p in PLACEHOLDER_STRINGS)

    # ---- Checks ----

    def check_trading_mode(self) -> None:
        mode = os.getenv("TRADING_MODE", "paper").lower()
        base_url = os.getenv("ALPACA_BASE_URL", "")
        if mode == "live" and "paper" in base_url.lower():
            self._add(
                "Trading Mode",
                False,
                f"TRADING_MODE=live but ALPACA_BASE_URL contains 'paper': {base_url}",
            )
        elif (
            mode == "paper"
            and base_url
            and "paper" not in base_url.lower()
            and "localhost" not in base_url.lower()
        ):
            self._add(
                "Trading Mode",
                False,
                f"TRADING_MODE=paper but ALPACA_BASE_URL looks like live: {base_url}",
            )
        else:
            self._add(
                "Trading Mode", True, f"TRADING_MODE={mode}, URL consistent", "info"
            )

    def check_api_key_presence(self) -> None:
        key = os.getenv("ALPACA_API_KEY", "")
        secret = os.getenv("ALPACA_SECRET_KEY", "")
        if not key or not secret:
            self._add(
                "API Key Presence",
                False,
                "ALPACA_API_KEY or ALPACA_SECRET_KEY is empty",
            )
        elif self._is_placeholder(key) or self._is_placeholder(secret):
            self._add("API Key Presence", False, "API keys contain placeholder text")
        else:
            self._add("API Key Presence", True, "API keys are set", "info")

    async def check_api_key_validity(self) -> None:
        broker = AlpacaBroker()
        try:
            await broker.connect()
            account = await broker.get_account()
            self._add(
                "API Key Validity",
                True,
                f"Authenticated OK (account={account.account_id}, status={account.status})",
                "info",
            )
        except Exception as e:
            self._add(
                "API Key Validity", False, f"Cannot authenticate with Alpaca: {e}"
            )
        finally:
            await broker.disconnect()

    def check_circuit_breaker_enabled(self) -> None:
        enabled = os.getenv("CIRCUIT_BREAKER_ENABLED", "").lower()
        if enabled != "true":
            self._add(
                "Circuit Breaker",
                False,
                f"CIRCUIT_BREAKER_ENABLED={enabled!r} (must be 'true' for live trading)",
            )
        else:
            self._add("Circuit Breaker", True, "Circuit breaker is enabled", "info")

    def check_position_limits(self) -> None:
        drawdown = os.getenv("MAX_PORTFOLIO_DRAWDOWN_PERCENT", "")
        if not drawdown:
            self._add(
                "Position Limits",
                False,
                "MAX_PORTFOLIO_DRAWDOWN_PERCENT is not set",
            )
        else:
            self._add("Position Limits", True, f"Max drawdown = {drawdown}%", "info")

    async def check_database_connectivity(self) -> None:
        try:
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                database=os.getenv("DB_NAME", "stock_trading_analysis"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", ""),
            )
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.close()
            self._add("Database", True, "PostgreSQL connection OK", "info")
        except Exception as e:
            self._add("Database", False, f"PostgreSQL connection failed: {e}")

    async def check_redis_connectivity(self) -> None:
        try:
            url = os.getenv("REDIS_URL", "redis://localhost:6379")
            r = redis.from_url(url)
            r.ping()
            r.close()
            self._add("Redis", True, "Redis PING OK", "info")
        except Exception as e:
            self._add("Redis", False, f"Redis connection failed: {e}")

    def check_notification_config(self) -> None:
        discord = os.getenv("DISCORD_WEBHOOK_URL", "")
        email_user = os.getenv("EMAIL_USER", "")
        email_pass = os.getenv("EMAIL_PASSWORD", "")

        channels = []
        if discord and not self._is_placeholder(discord):
            channels.append("Discord")
        if email_user and email_pass and not self._is_placeholder(email_user):
            channels.append("Email")

        if not channels:
            self._add(
                "Notifications",
                False,
                "No notification channels configured (Discord/Email have placeholders)",
                "warning",
            )
        else:
            self._add(
                "Notifications",
                True,
                f"Channels configured: {', '.join(channels)}",
                "info",
            )

    def check_paper_live_key_mismatch(self) -> None:
        mode = os.getenv("TRADING_MODE", "paper").lower()
        live_key = os.getenv("ALPACA_API_KEY", "")
        paper_key = os.getenv("ALPACA_PAPER_API_KEY", "")

        if mode == "live" and live_key and paper_key and live_key == paper_key:
            self._add(
                "Paper/Live Key Mismatch",
                False,
                "TRADING_MODE=live but ALPACA_API_KEY matches ALPACA_PAPER_API_KEY",
            )
        else:
            self._add(
                "Paper/Live Key Mismatch",
                True,
                "Keys are distinct or mode is paper",
                "info",
            )

    def check_drawdown_thresholds(self) -> None:
        raw = os.getenv("CB_DRAWDOWN_METHODS", "")
        if not raw:
            self._add("Drawdown Thresholds", True, "Using defaults", "info")
            return

        issues = []
        for part in raw.split(","):
            part = part.strip()
            if ":" not in part:
                continue
            _, thresh_str = part.split(":", 1)
            try:
                thresh = float(thresh_str.strip())
                if thresh < 1.0:
                    issues.append(f"{part} threshold <1% (very tight)")
                elif thresh > 20.0:
                    issues.append(f"{part} threshold >20% (very loose)")
            except ValueError:
                issues.append(f"{part} has invalid threshold")

        if issues:
            self._add(
                "Drawdown Thresholds",
                False,
                "; ".join(issues),
                "warning",
            )
        else:
            self._add(
                "Drawdown Thresholds",
                True,
                f"Thresholds look reasonable: {raw}",
                "info",
            )

    # ---- Run all ----

    async def run_all(self) -> None:
        # Sync checks
        self.check_trading_mode()
        self.check_api_key_presence()
        self.check_circuit_breaker_enabled()
        self.check_position_limits()
        self.check_notification_config()
        self.check_paper_live_key_mismatch()
        self.check_drawdown_thresholds()

        # Async checks
        await self.check_api_key_validity()
        await self.check_database_connectivity()
        await self.check_redis_connectivity()

    def print_report(self) -> int:
        """Print color-coded report. Returns 0 if all critical pass, else 1."""
        RED = "\033[91m"
        YELLOW = "\033[93m"
        GREEN = "\033[92m"
        CYAN = "\033[96m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        print(f"\n{BOLD}{'=' * 60}")
        print("  TRADING CONFIGURATION VALIDATION REPORT")
        print(f"{'=' * 60}{RESET}\n")

        critical_failures = 0
        warnings = 0

        for r in self.results:
            if r.passed:
                icon = f"{GREEN}PASS{RESET}"
            elif r.severity == "warning":
                icon = f"{YELLOW}WARN{RESET}"
                warnings += 1
            else:
                icon = f"{RED}FAIL{RESET}"
                critical_failures += 1

            print(f"  [{icon}] {r.name}")
            print(f"         {r.message}")
            print()

        print(f"{BOLD}{'=' * 60}{RESET}")
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        print(f"  {passed}/{total} checks passed", end="")
        if critical_failures:
            print(f" | {RED}{critical_failures} CRITICAL{RESET}", end="")
        if warnings:
            print(f" | {YELLOW}{warnings} WARNING(S){RESET}", end="")
        print()

        if critical_failures:
            print(f"\n  {RED}{BOLD}DO NOT proceed to live trading.{RESET}")
            return 1
        elif warnings:
            print(f"\n  {YELLOW}Review warnings before proceeding.{RESET}")
            return 0
        else:
            print(f"\n  {GREEN}All checks passed. Ready for next step.{RESET}")
            return 0


async def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()

    validator = ConfigValidator()
    await validator.run_all()
    return validator.print_report()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
