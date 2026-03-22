"""Tests for security hardening: no hardcoded paths, passwords, or secrets."""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories and file patterns to exclude from scanning
EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    "node_modules",
    "lightning_logs",
    ".pytest_cache",
    ".cache",
    ".eggs",
}
EXCLUDE_FILES = {
    "test_security_hardening.py",  # this file
}


def _iter_source_files(extensions=(".py", ".yml", ".yaml", ".sh")):
    """Yield (relative_path, full_path) for all source files."""
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for fname in files:
            if fname in EXCLUDE_FILES:
                continue
            if any(fname.endswith(ext) for ext in extensions):
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, BASE_DIR)
                yield rel, full


def _iter_all_text_files():
    """Yield (relative_path, full_path) for all text files including md/txt."""
    exts = (".py", ".yml", ".yaml", ".sh", ".md", ".txt", ".cfg", ".ini", ".toml")
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for fname in files:
            if fname in EXCLUDE_FILES:
                continue
            if any(fname.endswith(ext) for ext in exts):
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, BASE_DIR)
                yield rel, full


# ── 1. No hardcoded /home/kironix paths ─────────────────────────────────────


class TestNoHardcodedPaths:
    """No file should reference /home/kironix."""

    def test_no_home_kironix_in_source(self):
        violations = []
        for rel, full in _iter_all_text_files():
            try:
                with open(full, encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if "/home/kironix" in line:
                            violations.append(f"{rel}:{i}")
            except Exception:
                pass
        assert violations == [], f"Found /home/kironix in: {violations}"


# ── 2. No hardcoded password defaults ───────────────────────────────────────


class TestNoHardcodedPasswords:
    """No Python file should have password defaults in getenv calls."""

    # Patterns that indicate a hardcoded password default:
    # os.getenv("..._PASSWORD", "some_real_value")  or  password='literal'
    PASSWORD_DEFAULT_RE = re.compile(
        r"""getenv\(\s*['"][^'"]*(?:PASSWORD|SECRET)[^'"]*['"]"""
        r"""\s*,\s*['"](?!$)(?!['"])([^'"]+)['"]\)""",
        re.IGNORECASE,
    )
    HARDCODED_PW_RE = re.compile(
        r"""password\s*=\s*['"](?!$)(?!['"])([^'"]+)['"]""",
        re.IGNORECASE,
    )

    # Allowlist: empty strings and clearly placeholder values are OK
    ALLOWED_DEFAULTS = {"", "your_secure_password_here", "CHANGE_ME"}

    def test_no_password_defaults_in_getenv(self):
        violations = []
        for rel, full in _iter_source_files(extensions=(".py",)):
            try:
                with open(full, encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        m = self.PASSWORD_DEFAULT_RE.search(line)
                        if m and m.group(1) not in self.ALLOWED_DEFAULTS:
                            violations.append(f"{rel}:{i} — default={m.group(1)!r}")
            except Exception:
                pass
        assert violations == [], f"Hardcoded password defaults in getenv: {violations}"

    def test_no_hardcoded_password_literals(self):
        """No password='some_literal' (except empty string or env var lookups)."""
        violations = []
        for rel, full in _iter_source_files(extensions=(".py",)):
            try:
                with open(full, encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        m = self.HARDCODED_PW_RE.search(line)
                        if m:
                            val = m.group(1)
                            if val not in self.ALLOWED_DEFAULTS:
                                violations.append(f"{rel}:{i} — value={val!r}")
            except Exception:
                pass
        assert violations == [], f"Hardcoded password literals: {violations}"

    def test_no_tft_password_in_compose(self):
        compose_path = os.path.join(BASE_DIR, "docker-compose.yml")
        with open(compose_path) as f:
            content = f.read()
        assert (
            "tft_password" not in content
        ), "docker-compose.yml still contains hardcoded tft_password"

    def test_no_admin_password_in_compose(self):
        compose_path = os.path.join(BASE_DIR, "docker-compose.yml")
        with open(compose_path) as f:
            content = f.read()
        # Should not have GF_SECURITY_ADMIN_PASSWORD=admin (literal)
        assert "ADMIN_PASSWORD=admin" not in content


# ── 3. No real credentials in tracked files ──────────────────────────────────


class TestNoLeakedSecrets:
    """Scan for patterns that look like real API keys or tokens."""

    # Regex patterns for common secret formats
    SECRET_PATTERNS = [
        (r"beriha@123KB!", "hardcoded password"),
        (r"t9p6k7C5Wfo2fAlk7xn6CjyQtaAJPVOI", "Polygon API key"),
        (r"PKEFQ3SDGH2O2RH1PKD7", "Alpaca API key"),
        (r"T5pFq7vxZoaDj5bK3yAa9kLq8Gt3JiPl2W9rBr8i", "Alpaca secret"),
        (r"kVcqPqCT1Zf46EbQ2-77Tw", "Reddit client ID"),
        (r"mH8_klCUFx9Q5mYYJUk7D0ggJB7Vdw", "Reddit client secret"),
    ]

    def test_no_known_secrets_in_codebase(self):
        violations = []
        for rel, full in _iter_all_text_files():
            try:
                with open(full, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    for pattern, label in self.SECRET_PATTERNS:
                        if pattern in content:
                            violations.append(f"{rel}: contains {label}")
            except Exception:
                pass
        assert violations == [], f"Real secrets found in codebase: {violations}"


# ── 4. .env is gitignored ───────────────────────────────────────────────────


class TestGitignore:
    """Verify .env and secrets are properly gitignored."""

    @pytest.fixture(autouse=True)
    def load_gitignore(self):
        with open(os.path.join(BASE_DIR, ".gitignore")) as f:
            self.content = f.read()

    def test_env_in_gitignore(self):
        assert ".env" in self.content

    def test_env_local_in_gitignore(self):
        assert ".env.local" in self.content

    def test_key_files_in_gitignore(self):
        assert "*.key" in self.content
        assert "*.pem" in self.content

    def test_credentials_dir_in_gitignore(self):
        assert "credentials/" in self.content


# ── 5. .env.example has only placeholders ────────────────────────────────────


class TestEnvExample:
    """Verify .env.example has only placeholder values, never real creds."""

    @pytest.fixture(autouse=True)
    def load_example(self):
        with open(os.path.join(BASE_DIR, ".env.example")) as f:
            self.lines = f.readlines()

    def test_exists_and_not_empty(self):
        assert len(self.lines) > 5

    def test_no_real_passwords(self):
        for i, line in enumerate(self.lines, 1):
            line = line.strip()
            if "=" not in line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip().upper()
            if "PASSWORD" in key or "SECRET" in key:
                assert value in (
                    "",
                    "your_secure_password_here",
                    "your_alpaca_secret_key_here",
                    "your_reddit_client_secret_here",
                    "your_grafana_password_here",
                ), f"Line {i}: {key} has non-placeholder value: {value!r}"

    def test_has_required_vars(self):
        content = "\n".join(self.lines)
        assert "ALPACA_API_KEY" in content
        assert "ALPACA_SECRET_KEY" in content
        assert "DB_PASSWORD" in content
        assert "POSTGRES_PASSWORD" in content


# ── 6. env_validator module ──────────────────────────────────────────────────


class TestEnvValidator:
    """Test the startup environment validator."""

    def test_validate_catches_missing_required(self):
        from utils.env_validator import validate, REQUIRED_VARS

        # Clear all required vars
        saved = {}
        for var in REQUIRED_VARS:
            saved[var] = os.environ.pop(var, None)

        try:
            messages = validate(strict=False)
            fatal = [m for m in messages if "FATAL" in m]
            assert len(fatal) >= len(REQUIRED_VARS)
        finally:
            for var, val in saved.items():
                if val is not None:
                    os.environ[var] = val

    def test_validate_catches_placeholder(self):
        from utils.env_validator import validate

        os.environ["DB_PASSWORD"] = "your_password_here"
        os.environ["ALPACA_API_KEY"] = "real_key"
        os.environ["ALPACA_SECRET_KEY"] = "real_secret"
        try:
            messages = validate(strict=False)
            placeholder_msgs = [m for m in messages if "placeholder" in m]
            assert len(placeholder_msgs) >= 1
        finally:
            os.environ.pop("DB_PASSWORD", None)
            os.environ.pop("ALPACA_API_KEY", None)
            os.environ.pop("ALPACA_SECRET_KEY", None)

    def test_validate_passes_with_real_values(self):
        from utils.env_validator import validate, REQUIRED_VARS

        saved = {}
        for var in REQUIRED_VARS:
            saved[var] = os.environ.get(var)
            os.environ[var] = "real_production_value_123"

        try:
            messages = validate(strict=False)
            fatal = [m for m in messages if "FATAL" in m]
            assert len(fatal) == 0
        finally:
            for var, val in saved.items():
                if val is not None:
                    os.environ[var] = val
                else:
                    os.environ.pop(var, None)

    def test_validate_warns_on_missing_optional(self):
        from utils.env_validator import validate, REQUIRED_VARS, OPTIONAL_VARS

        saved = {}
        for var in REQUIRED_VARS:
            saved[var] = os.environ.get(var)
            os.environ[var] = "real_value_123"
        for var in OPTIONAL_VARS:
            saved[var] = os.environ.pop(var, None)

        try:
            messages = validate(strict=False)
            warnings = [m for m in messages if "WARNING" in m]
            assert len(warnings) > 0
        finally:
            for var, val in saved.items():
                if val is not None:
                    os.environ[var] = val
                else:
                    os.environ.pop(var, None)

    def test_strict_mode_exits_on_missing(self):
        from utils.env_validator import validate, REQUIRED_VARS

        saved = {}
        for var in REQUIRED_VARS:
            saved[var] = os.environ.pop(var, None)

        try:
            with pytest.raises(SystemExit):
                validate(strict=True)
        finally:
            for var, val in saved.items():
                if val is not None:
                    os.environ[var] = val

    def test_is_placeholder_detects_common_patterns(self):
        from utils.env_validator import _is_placeholder

        assert _is_placeholder("your_api_key")
        assert _is_placeholder("YOUR_KEY")
        assert _is_placeholder("CHANGE_ME")
        assert _is_placeholder("changeme_please")
        assert _is_placeholder("placeholder_value")
        assert _is_placeholder("xxxxx")
        assert not _is_placeholder("pk_live_abc123")
        assert not _is_placeholder("")


# ── 7. Docker compose uses env var interpolation ─────────────────────────────


class TestDockerComposeSecure:
    """Verify docker-compose.yml uses env var references for secrets."""

    @pytest.fixture(autouse=True)
    def load_compose(self):
        import yaml

        with open(os.path.join(BASE_DIR, "docker-compose.yml")) as f:
            self.raw = f.read()
            f.seek(0)
            # Note: can't yaml.safe_load because of ${} interpolation
            # Just check raw text patterns

    def test_postgres_password_uses_env_var(self):
        assert "${POSTGRES_PASSWORD" in self.raw

    def test_grafana_password_uses_env_var(self):
        assert "${GRAFANA_ADMIN_PASSWORD" in self.raw

    def test_database_url_uses_interpolation(self):
        # All DATABASE_URL values should use ${} interpolation
        for line in self.raw.split("\n"):
            if "DATABASE_URL=" in line and "postgresql://" in line:
                assert (
                    "${POSTGRES_PASSWORD" in line
                ), f"DATABASE_URL has hardcoded password: {line.strip()}"
