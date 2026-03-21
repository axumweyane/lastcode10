# Contributing to APEX

## Branch Protection Rules

The `main` and `master` branches are protected. The following rules apply:

1. **CI must pass before merge** — All jobs in the APEX CI workflow (lint, tests, security) must succeed before a pull request can be merged.
2. **Require 1 approving review** — At least one team member must approve the PR.
3. **No force push** — Force pushes to `main`/`master` are disabled to preserve commit history.
4. **No direct commits** — All changes must go through a pull request.

## Development Workflow

1. Create a feature branch from `master`:
   ```bash
   git checkout -b feature/my-change master
   ```

2. Make your changes and run tests locally:
   ```bash
   cd TFT-main
   pytest tests/ -v --tb=short -x
   ```

3. Check formatting before pushing:
   ```bash
   black --check .
   flake8 . --max-line-length 120 --exclude .git,__pycache__,venv
   ```

4. Push and open a pull request against `master`.

5. Wait for CI to pass and get a review, then merge.

## Test Requirements

- All new features must include tests in `tests/`.
- Tests must pass with `pytest tests/ -v` before merging.
- Aim for coverage of: happy path, error handling, edge cases.

## Strategy Conventions

- All strategies extend `BaseStrategy` from `strategies/base.py`.
- Strategies produce `StrategyOutput` with z-scored `AlphaScore` signals.
- All strategies are disabled by default; enable via `STRATEGY_*_ENABLED=true` in `.env`.
- Map new strategies to a regime weight bucket in `strategies/ensemble/combiner.py`.
