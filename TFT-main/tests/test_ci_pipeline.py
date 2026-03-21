"""Tests for the CI/CD pipeline configuration."""

import os
import sys

import yaml
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── 1. ci.yml validity ──────────────────────────────────────────────────────

class TestCIYamlValid:
    """Test that ci.yml is valid YAML with expected structure."""

    @pytest.fixture(autouse=True)
    def load_ci(self):
        ci_path = os.path.join(BASE_DIR, ".github", "workflows", "ci.yml")
        with open(ci_path) as f:
            self.ci = yaml.safe_load(f)

    def test_is_valid_yaml(self):
        assert isinstance(self.ci, dict)

    def test_has_name(self):
        assert "name" in self.ci
        assert self.ci["name"] == "APEX CI"

    def test_trigger_on_push(self):
        # PyYAML parses 'on' as True, so access via True key
        triggers = self.ci.get("on") or self.ci.get(True, {})
        assert "push" in triggers
        branches = triggers["push"]["branches"]
        assert "main" in branches
        assert "master" in branches

    def test_trigger_on_pull_request(self):
        triggers = self.ci.get("on") or self.ci.get(True, {})
        assert "pull_request" in triggers
        branches = triggers["pull_request"]["branches"]
        assert "main" in branches
        assert "master" in branches

    def test_trigger_manual_dispatch(self):
        triggers = self.ci.get("on") or self.ci.get(True, {})
        assert "workflow_dispatch" in triggers

    def test_has_4_jobs(self):
        assert len(self.ci["jobs"]) == 4

    def test_has_lint_job(self):
        assert "lint" in self.ci["jobs"]

    def test_has_test_job(self):
        assert "test" in self.ci["jobs"]

    def test_has_security_job(self):
        assert "security" in self.ci["jobs"]

    def test_has_docker_job(self):
        assert "docker" in self.ci["jobs"]

    def test_lint_runs_black(self):
        steps = self.ci["jobs"]["lint"]["steps"]
        step_names = [s.get("name", "") for s in steps]
        assert any("black" in n.lower() for n in step_names)

    def test_lint_runs_flake8(self):
        steps = self.ci["jobs"]["lint"]["steps"]
        step_names = [s.get("name", "") for s in steps]
        assert any("flake8" in n.lower() for n in step_names)

    def test_lint_runs_mypy(self):
        steps = self.ci["jobs"]["lint"]["steps"]
        step_names = [s.get("name", "") for s in steps]
        assert any("mypy" in n.lower() for n in step_names)

    def test_mypy_continues_on_error(self):
        steps = self.ci["jobs"]["lint"]["steps"]
        mypy_step = [s for s in steps if "mypy" in s.get("name", "").lower()]
        assert len(mypy_step) == 1
        assert mypy_step[0].get("continue-on-error") is True

    def test_test_job_runs_pytest(self):
        steps = self.ci["jobs"]["test"]["steps"]
        run_cmds = [s.get("run", "") for s in steps]
        assert any("pytest" in cmd for cmd in run_cmds)

    def test_test_job_uploads_artifact(self):
        steps = self.ci["jobs"]["test"]["steps"]
        assert any(s.get("uses", "").startswith("actions/upload-artifact") for s in steps)

    def test_security_runs_pip_audit(self):
        steps = self.ci["jobs"]["security"]["steps"]
        run_cmds = [s.get("run", "") for s in steps]
        assert any("pip-audit" in cmd for cmd in run_cmds)

    def test_security_runs_detect_secrets(self):
        steps = self.ci["jobs"]["security"]["steps"]
        run_cmds = [s.get("run", "") for s in steps]
        assert any("detect-secrets" in cmd for cmd in run_cmds)

    def test_security_continues_on_error(self):
        steps = self.ci["jobs"]["security"]["steps"]
        audit_step = [s for s in steps if s.get("name", "") == "Check for vulnerable dependencies"]
        assert len(audit_step) == 1
        assert audit_step[0].get("continue-on-error") is True

    def test_docker_only_on_push(self):
        docker_if = self.ci["jobs"]["docker"]["if"]
        assert "push" in docker_if

    def test_docker_needs_lint_and_test(self):
        needs = self.ci["jobs"]["docker"]["needs"]
        assert "lint" in needs
        assert "test" in needs

    def test_yamllint_step_exists(self):
        steps = self.ci["jobs"]["lint"]["steps"]
        step_names = [s.get("name", "") for s in steps]
        assert any("yaml" in n.lower() for n in step_names)

    def test_python_312(self):
        for job_name in ["lint", "test", "security"]:
            steps = self.ci["jobs"][job_name]["steps"]
            python_step = [s for s in steps if "setup-python" in s.get("uses", "")]
            assert len(python_step) == 1
            assert python_step[0]["with"]["python-version"] == "3.12"


# ── 2. Referenced paths exist ────────────────────────────────────────────────

class TestReferencedPaths:
    """Verify that files and directories referenced by CI exist."""

    def test_requirements_txt_exists(self):
        assert os.path.isfile(os.path.join(BASE_DIR, "requirements.txt"))

    def test_tests_directory_exists(self):
        assert os.path.isdir(os.path.join(BASE_DIR, "tests"))

    def test_docker_compose_exists(self):
        assert os.path.isfile(os.path.join(BASE_DIR, "docker-compose.yml"))

    def test_ci_yml_exists(self):
        assert os.path.isfile(os.path.join(BASE_DIR, ".github", "workflows", "ci.yml"))

    def test_prometheus_targets_exists(self):
        assert os.path.isfile(
            os.path.join(BASE_DIR, "monitoring", "prometheus", "apex_targets.yml")
        )

    def test_grafana_datasource_exists(self):
        assert os.path.isfile(
            os.path.join(BASE_DIR, "monitoring", "grafana", "datasources", "datasource.yml")
        )


# ── 3. requirements.txt parseable ───────────────────────────────────────────

class TestRequirementsTxt:
    """Verify requirements.txt is parseable."""

    @pytest.fixture(autouse=True)
    def load_reqs(self):
        req_path = os.path.join(BASE_DIR, "requirements.txt")
        with open(req_path) as f:
            self.lines = f.readlines()

    def test_not_empty(self):
        assert len(self.lines) > 0

    def test_no_syntax_errors(self):
        for i, line in enumerate(self.lines, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Each non-comment, non-flag line should be a package spec
            # Valid patterns: package, package==ver, package>=ver, package[extra]
            assert len(line) > 0, f"Empty requirement on line {i}"
            # Should not contain spaces (except in markers like ; python_version)
            parts = line.split(";")[0].strip()
            assert " " not in parts or "[" in parts, (
                f"Malformed requirement on line {i}: {line}"
            )

    def test_common_deps_present(self):
        content = "\n".join(self.lines).lower()
        for dep in ["fastapi", "pandas", "numpy"]:
            assert dep in content, f"Expected dependency '{dep}' not in requirements.txt"


# ── 4. README badge ─────────────────────────────────────────────────────────

class TestReadmeBadge:
    """Verify CI badge is in README.md."""

    def test_badge_present(self):
        readme_path = os.path.join(BASE_DIR, "README.md")
        with open(readme_path) as f:
            content = f.read()
        assert "actions/workflows/ci.yml/badge.svg" in content


# ── 5. CONTRIBUTING.md ──────────────────────────────────────────────────────

class TestContributing:
    """Verify CONTRIBUTING.md exists and documents branch protection."""

    @pytest.fixture(autouse=True)
    def load_contrib(self):
        path = os.path.join(BASE_DIR, "CONTRIBUTING.md")
        with open(path) as f:
            self.content = f.read()

    def test_exists(self):
        assert len(self.content) > 0

    def test_documents_ci_requirement(self):
        assert "CI" in self.content

    def test_documents_review_requirement(self):
        assert "review" in self.content.lower()

    def test_documents_no_force_push(self):
        assert "force push" in self.content.lower()
