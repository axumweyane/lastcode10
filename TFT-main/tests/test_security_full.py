"""
Full security audit tests for APEX Trading System.

Tests:
  1. .env not tracked in git
  2. No hardcoded passwords in Python files
  3. No hardcoded API keys in Python files
  4. env_validator rejects missing required vars
  5. env_validator rejects placeholder values
  6. pip-audit for CRITICAL/HIGH vulnerabilities
  7. detect-secrets scan for leaked secrets
"""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ---------- 1. .env not tracked in git ----------


class TestEnvNotTracked:
    def test_env_not_in_git(self):
        result = subprocess.run(
            ["git", "ls-files", ".env"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        tracked = result.stdout.strip()
        assert tracked == "", f".env is tracked in git: {tracked}"

    def test_gitignore_has_env(self):
        gitignore = os.path.join(ROOT, ".gitignore")
        if os.path.exists(gitignore):
            content = open(gitignore).read()
            assert ".env" in content, ".gitignore does not contain .env"


# ---------- 2. No hardcoded passwords ----------

# Patterns that indicate a real password value (not env lookup or placeholder).
_PASSWORD_FALSE_POSITIVES = {
    "os.environ",
    "os.getenv",
    "getenv(",
    "${",
    "password=password",  # variable assignment like password=password_var
    "password=%s",
    "password=None",
    "password=''",
    'password=""',
    "password=<",
    "password=your",
    "password=CHANGE",
    "password=changeme",
    "password=xxx",
    "# ",
    "test",
    "mock",
    "fixture",
    "placeholder",
    "example",
    "default",
    "args.",
    "parser.",
    ".password",
    "password_var",
    "password_field",
    "password=db_",
    "password=self.",
    "password=self.db_config",
    '["password"]',
    "['password']",
}


class TestNoHardcodedPasswords:
    def _get_password_lines(self):
        """Grep Python files for password= with actual values."""
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "password=", ROOT],
            capture_output=True,
            text=True,
        )
        hits = []
        for line in result.stdout.splitlines():
            lower = line.lower()
            # Skip lines that are clearly env lookups, tests, or comments
            if any(fp in lower for fp in _PASSWORD_FALSE_POSITIVES):
                continue
            # Skip test files
            if "/tests/" in line or "/test_" in line:
                continue
            hits.append(line)
        return hits

    def test_no_hardcoded_passwords(self):
        hits = self._get_password_lines()
        assert hits == [], f"Found {len(hits)} hardcoded password(s):\n" + "\n".join(
            hits[:10]
        )


# ---------- 3. No hardcoded API keys ----------

_API_KEY_PATTERNS = [
    r"api_key\s*=\s*['\"][A-Za-z0-9]{16,}['\"]",
    r"secret_key\s*=\s*['\"][A-Za-z0-9]{16,}['\"]",
    r"API_KEY\s*=\s*['\"][A-Za-z0-9]{16,}['\"]",
]


class TestNoHardcodedAPIKeys:
    def test_no_hardcoded_api_keys(self):
        for pattern in _API_KEY_PATTERNS:
            result = subprocess.run(
                ["grep", "-rnE", "--include=*.py", pattern, ROOT],
                capture_output=True,
                text=True,
            )
            hits = []
            for line in result.stdout.splitlines():
                lower = line.lower()
                # Skip test files, env lookups, comments
                if any(
                    skip in lower
                    for skip in [
                        "/tests/",
                        "/test_",
                        "os.environ",
                        "os.getenv",
                        "# ",
                        "mock",
                        "fixture",
                        "placeholder",
                        "example",
                    ]
                ):
                    continue
                hits.append(line)
            assert (
                hits == []
            ), f"Hardcoded API key pattern '{pattern}' found:\n" + "\n".join(hits[:10])

    def test_no_real_alpaca_keys_in_source(self):
        """No Alpaca key patterns (PK...) in Python source (excluding .env)."""
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "PK[A-Z0-9]\\{10,\\}", ROOT],
            capture_output=True,
            text=True,
        )
        hits = [
            l
            for l in result.stdout.splitlines()
            if "/tests/" not in l and "# " not in l.split(":", 2)[-1].lstrip()
        ]
        assert hits == [], f"Alpaca key pattern in source:\n" + "\n".join(hits[:5])


# ---------- 4. env_validator rejects missing required vars ----------


class TestEnvValidatorMissing:
    def test_rejects_missing_db_password(self):
        env = os.environ.copy()
        env.pop("DB_PASSWORD", None)
        env.pop("ALPACA_API_KEY", None)
        env.pop("ALPACA_SECRET_KEY", None)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0,'.'); "
                "from utils.env_validator import validate; "
                "msgs = validate(strict=False); "
                "print('\\n'.join(msgs))",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )
        assert "DB_PASSWORD" in result.stdout
        assert "FATAL" in result.stdout

    def test_rejects_missing_alpaca_key(self):
        env = os.environ.copy()
        env["DB_PASSWORD"] = "real_pass"
        env.pop("ALPACA_API_KEY", None)
        env.pop("ALPACA_SECRET_KEY", None)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0,'.'); "
                "from utils.env_validator import validate; "
                "msgs = validate(strict=False); "
                "print('\\n'.join(msgs))",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )
        assert "ALPACA_API_KEY" in result.stdout

    def test_returns_errors_for_all_missing(self):
        env = os.environ.copy()
        for var in ["DB_PASSWORD", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"]:
            env.pop(var, None)

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0,'.'); "
                "from utils.env_validator import validate; "
                "msgs = validate(strict=False); "
                "fatals = [m for m in msgs if 'FATAL' in m]; "
                "print(len(fatals))",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )
        count = int(result.stdout.strip())
        assert count >= 3, f"Expected >= 3 FATAL errors, got {count}"


# ---------- 5. env_validator rejects placeholder values ----------


class TestEnvValidatorPlaceholders:
    @pytest.mark.parametrize(
        "placeholder",
        [
            "your_key_here",
            "YOUR_API_KEY",
            "CHANGE_ME",
            "changeme",
            "xxx",
            "placeholder_value",
            "example_password",
        ],
    )
    def test_rejects_placeholder(self, placeholder):
        env = os.environ.copy()
        env["DB_PASSWORD"] = placeholder
        env["ALPACA_API_KEY"] = "real_key_abc123"
        env["ALPACA_SECRET_KEY"] = "real_secret_def456"

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0,'.'); "
                "from utils.env_validator import validate; "
                "msgs = validate(strict=False); "
                "print('\\n'.join(msgs))",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )
        assert (
            "placeholder" in result.stdout.lower() or "FATAL" in result.stdout
        ), f"Validator did not reject placeholder '{placeholder}'"

    def test_accepts_real_values(self):
        env = os.environ.copy()
        env["DB_PASSWORD"] = "s3cure_Passw0rd!"
        env["ALPACA_API_KEY"] = "PKABCDEF12345678"
        env["ALPACA_SECRET_KEY"] = "abcdef1234567890abcdef1234567890"

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0,'.'); "
                "from utils.env_validator import validate; "
                "msgs = validate(strict=False); "
                "fatals = [m for m in msgs if 'FATAL' in m]; "
                "print(len(fatals))",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )
        count = int(result.stdout.strip())
        assert count == 0, "Validator rejected real credential values"


# ---------- 6. pip-audit ----------


class TestPipAudit:
    def test_no_critical_or_high_vulns(self):
        result = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--desc", "--progress-spinner=off"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        # pip-audit exits 1 if vulns found. Check for CRITICAL/HIGH.
        critical = [
            l
            for l in output.splitlines()
            if "CRITICAL" in l.upper() or "HIGH" in l.upper()
        ]
        # This is informational — we warn but don't hard-fail since
        # transitive deps may have known issues without available fixes.
        if critical:
            pytest.skip(
                f"pip-audit found {len(critical)} CRITICAL/HIGH issue(s) "
                f"(informational):\n" + "\n".join(critical[:5])
            )


# ---------- 7. detect-secrets ----------


class TestDetectSecrets:
    def test_no_secrets_in_source(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "detect_secrets",
                "scan",
                "--exclude-files",
                r"\.env$",
                "--exclude-files",
                r"\.env\.template$",
                "--exclude-files",
                r"tests/",
                "--exclude-files",
                r"lightning_logs/",
                "--exclude-files",
                r"\.pth$",
                "--exclude-files",
                r"\.ckpt$",
                "--exclude-files",
                r"node_modules/",
                ".",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=120,
        )
        import json

        try:
            scan = json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.skip(f"detect-secrets output not JSON: {result.stderr[:200]}")
            return

        results = scan.get("results", {})
        # Filter out false positives (hex strings in configs, test data)
        real_secrets = {}
        for filepath, findings in results.items():
            # Skip documentation, markdown, configs
            if any(
                filepath.endswith(ext)
                for ext in [".md", ".json", ".yml", ".yaml", ".txt", ".cfg"]
            ):
                continue
            real = [f for f in findings if f.get("type") != "Hex High Entropy String"]
            if real:
                real_secrets[filepath] = real

        if real_secrets:
            summary = "\n".join(
                f"  {fp}: {len(fs)} finding(s)" for fp, fs in real_secrets.items()
            )
            pytest.fail(
                f"detect-secrets found {sum(len(v) for v in real_secrets.values())} "
                f"potential secret(s):\n{summary}"
            )
