"""
Startup environment variable validation.

Call validate() at application startup to ensure all required
environment variables are set before the system tries to use them.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

# Environment variables that MUST be set for production.
# Missing any of these is a fatal startup error.
REQUIRED_VARS = [
    "DB_PASSWORD",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
]

# Environment variables that SHOULD be set but have sensible defaults.
# A warning is logged if missing.
OPTIONAL_VARS = {
    "DB_HOST": "localhost",
    "DB_PORT": "5432",
    "DB_NAME": "tft_trading",
    "DB_USER": "tft_user",
    "REDIS_URL": "redis://localhost:6379",
    "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
    "DISCORD_WEBHOOK_URL": "",
    "EMAIL_PASSWORD": "",
    "POLYGON_API_KEY": "",
}

# Patterns that indicate a value is still a placeholder, not a real credential.
_PLACEHOLDER_PATTERNS = [
    "your_",
    "YOUR_",
    "CHANGE_ME",
    "changeme",
    "placeholder",
    "xxx",
    "XXX",
    "example",
    "EXAMPLE",
]


def _is_placeholder(value: str) -> bool:
    return any(p in value for p in _PLACEHOLDER_PATTERNS)


def validate(strict: bool = True) -> list[str]:
    """Validate that all required env vars are set and not placeholders.

    Args:
        strict: If True (default), raise SystemExit on missing required vars.
                If False, return list of error messages without exiting.

    Returns:
        List of error/warning messages (empty if all OK).
    """
    messages: list[str] = []

    # Check required vars
    missing = []
    placeholder = []
    for var in REQUIRED_VARS:
        value = os.getenv(var)
        if value is None or value == "":
            missing.append(var)
        elif _is_placeholder(value):
            placeholder.append(var)

    for var in missing:
        msg = f"FATAL: {var} env var is required but not set"
        messages.append(msg)
        logger.error(msg)

    for var in placeholder:
        msg = f"FATAL: {var} contains a placeholder value — set a real credential"
        messages.append(msg)
        logger.error(msg)

    # Check optional vars
    for var, default in OPTIONAL_VARS.items():
        value = os.getenv(var)
        if value is None:
            msg = f"WARNING: {var} not set, using default: {default!r}"
            messages.append(msg)
            logger.warning(msg)
        elif _is_placeholder(value):
            msg = f"WARNING: {var} contains a placeholder value"
            messages.append(msg)
            logger.warning(msg)

    if strict and (missing or placeholder):
        logger.critical(
            "Startup blocked: %d required env var(s) missing or placeholder. "
            "Set them in .env or your environment before running.",
            len(missing) + len(placeholder),
        )
        sys.exit(1)

    return messages
