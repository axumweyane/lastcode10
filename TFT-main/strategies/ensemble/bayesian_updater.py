"""
Beta-Binomial Bayesian weight updater for APEX ensemble strategies.

Each strategy maintains a Beta(alpha, beta) posterior:
    - Prior: Beta(1, 1) (uninformative / uniform)
    - Update: profitable day → alpha += 1, unprofitable → beta += 1
    - Weight = E[Beta] = alpha / (alpha + beta)
    - Exponential forgetting (lambda=0.995) decays old observations

The updater can persist state to PostgreSQL so learned weights survive restarts.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DECAY = 0.995
DEFAULT_MAX_WINDOW = 500


@dataclass
class StrategyBeta:
    """Beta distribution state for a single strategy."""

    strategy_name: str
    alpha: float = 1.0
    beta: float = 1.0
    n_updates: int = 0
    last_updated: str = ""

    @property
    def weight(self) -> float:
        """E[Beta] = alpha / (alpha + beta)."""
        total = self.alpha + self.beta
        if total <= 0:
            return 0.5
        return self.alpha / total

    @property
    def variance(self) -> float:
        """Var[Beta] = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))."""
        total = self.alpha + self.beta
        if total <= 0:
            return 0.0
        return (self.alpha * self.beta) / (total**2 * (total + 1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "alpha": round(self.alpha, 6),
            "beta": round(self.beta, 6),
            "weight": round(self.weight, 6),
            "variance": round(self.variance, 8),
            "n_updates": self.n_updates,
            "last_updated": self.last_updated,
        }


class BayesianWeightUpdater:
    """
    Adaptive strategy weight updater using Beta-Binomial conjugate model.

    Usage:
        updater = BayesianWeightUpdater()
        updater.update({"momentum": True, "pairs": False, "tft": True})
        weights = updater.get_weights()
        # {"momentum": 0.66, "pairs": 0.33, "tft": 0.66}
    """

    def __init__(
        self,
        decay_factor: float = DEFAULT_DECAY,
        max_window: int = DEFAULT_MAX_WINDOW,
    ):
        self.decay_factor = decay_factor
        self.max_window = max_window
        self._strategies: Dict[str, StrategyBeta] = {}

    def _ensure_strategy(self, name: str) -> StrategyBeta:
        """Get or create Beta state for a strategy."""
        if name not in self._strategies:
            self._strategies[name] = StrategyBeta(strategy_name=name)
        return self._strategies[name]

    def update(self, outcomes: Dict[str, bool]) -> Dict[str, float]:
        """
        Update all strategies given today's outcomes.

        Args:
            outcomes: {strategy_name: was_profitable} mapping.

        Returns:
            Updated weights per strategy.
        """
        now = datetime.now(timezone.utc).isoformat()

        for name, profitable in outcomes.items():
            state = self._ensure_strategy(name)

            # Exponential decay: shrink existing counts toward prior
            state.alpha = 1.0 + (state.alpha - 1.0) * self.decay_factor
            state.beta = 1.0 + (state.beta - 1.0) * self.decay_factor

            # Update with new observation
            if profitable:
                state.alpha += 1.0
            else:
                state.beta += 1.0

            state.n_updates += 1
            state.last_updated = now

        return self.get_weights()

    def get_weights(self) -> Dict[str, float]:
        """Return current weight per strategy (normalized to sum to 1)."""
        if not self._strategies:
            return {}
        raw = {name: s.weight for name, s in self._strategies.items()}
        total = sum(raw.values())
        if total <= 0:
            n = len(raw)
            return {name: 1.0 / n for name in raw}
        return {name: w / total for name, w in raw.items()}

    def get_raw_weights(self) -> Dict[str, float]:
        """Return unnormalized E[Beta] per strategy."""
        return {name: s.weight for name, s in self._strategies.items()}

    def get_state(self) -> Dict[str, StrategyBeta]:
        """Return full Beta state for all strategies."""
        return dict(self._strategies)

    def get_state_dicts(self) -> List[Dict[str, Any]]:
        """Return serializable state for all strategies."""
        return [s.to_dict() for s in self._strategies.values()]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize full state to JSON."""
        data = {
            "decay_factor": self.decay_factor,
            "max_window": self.max_window,
            "strategies": {name: asdict(s) for name, s in self._strategies.items()},
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "BayesianWeightUpdater":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        updater = cls(
            decay_factor=data.get("decay_factor", DEFAULT_DECAY),
            max_window=data.get("max_window", DEFAULT_MAX_WINDOW),
        )
        for name, s_data in data.get("strategies", {}).items():
            updater._strategies[name] = StrategyBeta(**s_data)
        return updater

    def save_to_rows(self) -> List[Tuple]:
        """Return list of (strategy_name, alpha, beta, n_updates, state_json) tuples for DB."""
        rows = []
        for name, s in self._strategies.items():
            rows.append(
                (
                    name,
                    s.alpha,
                    s.beta,
                    s.n_updates,
                    json.dumps(s.to_dict()),
                )
            )
        return rows

    def load_from_rows(self, rows: List[Tuple]) -> None:
        """Load state from DB rows: (strategy_name, alpha, beta, n_updates, state_json)."""
        for row in rows:
            name, alpha, beta, n_updates = row[0], row[1], row[2], row[3]
            state_json = row[4] if len(row) > 4 else "{}"
            extra = json.loads(state_json) if state_json else {}
            self._strategies[name] = StrategyBeta(
                strategy_name=name,
                alpha=float(alpha),
                beta=float(beta),
                n_updates=int(n_updates),
                last_updated=extra.get("last_updated", ""),
            )
        logger.info(
            "Loaded Bayesian state for %d strategies: %s",
            len(rows),
            {
                name: f"a={s.alpha:.2f} b={s.beta:.2f} w={s.weight:.3f}"
                for name, s in self._strategies.items()
            },
        )
