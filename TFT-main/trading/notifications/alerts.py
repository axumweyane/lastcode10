"""
Notification system for trading alerts (Discord webhook + email).
All sends are fire-and-forget: failures are logged but never block callers.
"""

import asyncio
import json
import logging
import os
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class AlertMessage:
    title: str
    body: str
    severity: str  # "info", "warning", "critical"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertSender(ABC):
    @abstractmethod
    async def send(self, message: AlertMessage) -> bool:
        """Send an alert. Return True on success."""


class DiscordWebhookSender(AlertSender):
    SEVERITY_COLORS = {
        "info": 0x3498DB,  # blue
        "warning": 0xF39C12,  # orange
        "critical": 0xE74C3C,  # red
    }

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send(self, message: AlertMessage) -> bool:
        color = self.SEVERITY_COLORS.get(message.severity, 0x95A5A6)
        embed = {
            "title": message.title,
            "description": message.body,
            "color": color,
            "timestamp": message.timestamp.isoformat(),
            "fields": [
                {"name": k, "value": str(v), "inline": True}
                for k, v in message.metadata.items()
            ],
        }
        payload = {"embeds": [embed]}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status in (200, 204):
                        return True
                    logger.error(
                        "Discord webhook returned %s: %s",
                        resp.status,
                        await resp.text(),
                    )
                    return False
        except Exception as e:
            logger.error("Discord webhook failed: %s", e)
            return False


class EmailSender(AlertSender):
    SUBJECT_PREFIX = {
        "info": "[INFO]",
        "warning": "[WARNING]",
        "critical": "[CRITICAL]",
    }

    def __init__(
        self,
        smtp_user: str,
        smtp_password: str,
        recipient: str,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
    ):
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.recipient = recipient
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    async def send(self, message: AlertMessage) -> bool:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._send_sync, message)
            return True
        except Exception as e:
            logger.error("Email send failed: %s", e)
            return False

    def _send_sync(self, message: AlertMessage) -> None:
        prefix = self.SUBJECT_PREFIX.get(message.severity, "[ALERT]")
        subject = f"{prefix} {message.title}"

        body_parts = [message.body, ""]
        if message.metadata:
            body_parts.append("Details:")
            for k, v in message.metadata.items():
                body_parts.append(f"  {k}: {v}")
        body_parts.append(f"\nTimestamp: {message.timestamp.isoformat()}")

        msg = MIMEMultipart()
        msg["From"] = self.smtp_user
        msg["To"] = self.recipient
        msg["Subject"] = subject
        msg.attach(MIMEText("\n".join(body_parts), "plain"))

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)


class NotificationManager:
    """Auto-configures senders from env vars. Sends to all channels."""

    def __init__(self, senders: Optional[List[AlertSender]] = None):
        self.senders: List[AlertSender] = senders or []

    @classmethod
    def from_env(cls) -> "NotificationManager":
        senders: List[AlertSender] = []

        discord_url = os.getenv("DISCORD_WEBHOOK_URL", "")
        if discord_url and "YOUR_WEBHOOK" not in discord_url:
            senders.append(DiscordWebhookSender(discord_url))
            logger.info("Discord notifications enabled")

        email_user = os.getenv("EMAIL_USER", "")
        email_pass = os.getenv("EMAIL_PASSWORD", "")
        email_to = os.getenv("EMAIL_TO", "")
        if (
            email_user
            and email_pass
            and email_to
            and "your_" not in email_user
            and "your_" not in email_pass
        ):
            senders.append(EmailSender(email_user, email_pass, email_to))
            logger.info("Email notifications enabled")

        if not senders:
            logger.warning("No notification channels configured")

        return cls(senders)

    async def send(self, message: AlertMessage) -> None:
        """Fire-and-forget to all channels."""
        for sender in self.senders:
            try:
                await sender.send(message)
            except Exception as e:
                logger.error(
                    "Notification sender %s failed: %s", type(sender).__name__, e
                )

    async def notify_circuit_breaker_trip(
        self,
        reason: str,
        drawdown_percent: float,
        portfolio_value: float,
        positions_closed: int,
    ) -> None:
        await self.send(
            AlertMessage(
                title="CIRCUIT BREAKER TRIPPED",
                body=(
                    f"Trading has been halted.\n"
                    f"Reason: {reason}\n"
                    f"All {positions_closed} position(s) have been closed."
                ),
                severity="critical",
                metadata={
                    "Drawdown": f"{drawdown_percent:.2f}%",
                    "Portfolio Value": f"${portfolio_value:,.2f}",
                    "Positions Closed": str(positions_closed),
                },
            )
        )

    async def notify_circuit_breaker_reset(
        self,
        operator: str,
        reason: str,
        portfolio_value: float,
    ) -> None:
        await self.send(
            AlertMessage(
                title="Circuit Breaker Reset",
                body=(
                    f"Trading has been re-enabled.\n"
                    f"Operator: {operator}\n"
                    f"Reason: {reason}"
                ),
                severity="warning",
                metadata={
                    "Portfolio Value": f"${portfolio_value:,.2f}",
                    "Reset By": operator,
                },
            )
        )
