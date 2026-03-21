"""
Walk-forward cross-validation for the APEX multi-strategy system.

Fixes CF-1 (deploy most recent fold), CF-2 (embargo gap), CF-4 (normalization
stats per fold), and CF-3 (frequency-aware Sharpe annualization).

Rolling window walk-forward:
    |--- IS (train) ---|-- embargo --|--- OOS (test) ---|
                       |--- IS (train) ---|-- embargo --|--- OOS (test) ---|
                                          |--- IS (train) ---|-- ...

Each fold produces:
    - Sharpe, max drawdown, win rate, profit factor, trade count
    - Normalization stats saved as JSON sidecar
    - The MOST RECENT fold is always selected for deployment
"""

import json
import logging
import os
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.config import WalkForwardConfig

logger = logging.getLogger(__name__)

# Annualization factors
ANNUALIZATION_FACTORS = {
    "daily": np.sqrt(252),
    "minute": np.sqrt(252 * 390),
}


@dataclass
class NormalizationStats:
    """Per-fold normalization statistics saved as JSON sidecar."""
    fold_index: int
    mean: Dict[str, float]
    std: Dict[str, float]
    columns: List[str]
    n_samples: int
    frequency: str
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "NormalizationStats":
        data = json.loads(json_str)
        return cls(**data)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())
        logger.info("Saved normalization stats for fold %d to %s", self.fold_index, path)

    @classmethod
    def load(cls, path: str) -> "NormalizationStats":
        with open(path) as f:
            return cls.from_json(f.read())


@dataclass
class FoldResult:
    """Metrics from a single walk-forward fold."""
    fold_index: int
    is_start: int          # row index into original data
    is_end: int
    oos_start: int         # after embargo
    oos_end: int
    embargo_bars: int
    sharpe: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    total_return: float
    annualization_factor: float
    frequency: str
    norm_stats_path: Optional[str] = None

    @property
    def is_most_recent(self) -> bool:
        return False  # set externally

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WalkForwardReport:
    """Summary report across all folds."""
    folds: List[FoldResult]
    deployed_fold_index: int
    best_sharpe_fold_index: int
    sharpe_mean: float
    sharpe_std: float
    sharpe_stability: float   # 1 - (std / mean) if mean > 0, else 0
    total_folds: int
    config: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_folds": self.total_folds,
            "deployed_fold_index": self.deployed_fold_index,
            "best_sharpe_fold_index": self.best_sharpe_fold_index,
            "sharpe_mean": round(self.sharpe_mean, 4),
            "sharpe_std": round(self.sharpe_std, 4),
            "sharpe_stability": round(self.sharpe_stability, 4),
            "config": self.config,
            "warnings": self.warnings,
            "folds": [f.to_dict() for f in self.folds],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# Walk-Forward Validation Report",
            "",
            f"**Total folds:** {self.total_folds}",
            f"**Deployed fold:** {self.deployed_fold_index} (most recent)",
            f"**Best Sharpe fold:** {self.best_sharpe_fold_index}",
            f"**Sharpe mean:** {self.sharpe_mean:.4f}",
            f"**Sharpe std:** {self.sharpe_std:.4f}",
            f"**Sharpe stability:** {self.sharpe_stability:.4f}",
            "",
        ]
        if self.warnings:
            lines.append("## Warnings")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        lines.append("## Fold Details")
        lines.append("")
        lines.append("| Fold | Sharpe | MaxDD | WinRate | PF | Trades | Return |")
        lines.append("|------|--------|-------|---------|----|--------|--------|")
        for f in self.folds:
            deployed = " *" if f.fold_index == self.deployed_fold_index else ""
            lines.append(
                f"| {f.fold_index}{deployed} | {f.sharpe:.4f} | "
                f"{f.max_drawdown:.2%} | {f.win_rate:.2%} | "
                f"{f.profit_factor:.2f} | {f.n_trades} | {f.total_return:.2%} |"
            )
        lines.append("")
        lines.append("\\* = deployed fold")
        return "\n".join(lines)


def compute_sharpe(returns: np.ndarray, frequency: str = "daily") -> float:
    """Compute annualized Sharpe ratio with correct frequency factor."""
    if len(returns) < 2:
        return 0.0
    ann_factor = ANNUALIZATION_FACTORS.get(frequency, ANNUALIZATION_FACTORS["daily"])
    std = np.std(returns, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.mean(returns) / std * ann_factor)


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from a return series."""
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0


def compute_win_rate(returns: np.ndarray) -> float:
    """Fraction of positive returns."""
    if len(returns) == 0:
        return 0.0
    return float(np.sum(returns > 0) / len(returns))


def compute_profit_factor(returns: np.ndarray) -> float:
    """Gross profits / gross losses. Returns inf if no losses."""
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    total_gains = float(np.sum(gains)) if len(gains) > 0 else 0.0
    total_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    if total_losses < 1e-12:
        return float("inf") if total_gains > 0 else 0.0
    return total_gains / total_losses


class WalkForwardValidator:
    """
    Rolling-window walk-forward cross-validation engine.

    Usage:
        validator = WalkForwardValidator(config)
        report = validator.run(data, strategy_fn)

    strategy_fn signature:
        def strategy_fn(
            train_data: pd.DataFrame,
            test_data: pd.DataFrame,
            fold_index: int,
        ) -> np.ndarray:
            # Train on train_data, predict on test_data
            # Return array of OOS returns (one per bar in test_data)
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig.from_env()

    def generate_folds(
        self, n_rows: int,
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Generate (IS, OOS) index ranges for walk-forward folds.

        Returns list of ((is_start, is_end), (oos_start, oos_end)) tuples.
        oos_start includes the embargo offset.
        """
        is_window = self.config.is_window
        oos_window = self.config.oos_window
        embargo = self.config.embargo_bars

        folds = []
        start = 0

        while True:
            is_start = start
            is_end = is_start + is_window

            # OOS starts after IS end + embargo gap
            oos_start = is_end + embargo
            oos_end = oos_start + oos_window

            if oos_end > n_rows:
                # Partial final fold: only if we have at least 1 OOS bar
                if oos_start < n_rows:
                    folds.append(((is_start, is_end), (oos_start, n_rows)))
                break

            folds.append(((is_start, is_end), (oos_start, oos_end)))

            # Step forward by OOS window size
            start += oos_window

        return folds

    def compute_normalization_stats(
        self, data: pd.DataFrame, fold_index: int,
    ) -> NormalizationStats:
        """Compute and return normalization stats for the IS window."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        means = data[numeric_cols].mean().to_dict()
        stds = data[numeric_cols].std().to_dict()
        # Replace NaN with 0 for mean, 1 for std
        means = {k: (v if pd.notna(v) else 0.0) for k, v in means.items()}
        stds = {k: (v if pd.notna(v) and v > 0 else 1.0) for k, v in stds.items()}

        return NormalizationStats(
            fold_index=fold_index,
            mean=means,
            std=stds,
            columns=numeric_cols,
            n_samples=len(data),
            frequency=self.config.frequency,
        )

    def _save_norm_stats(
        self, stats: NormalizationStats, fold_index: int,
    ) -> str:
        """Save normalization stats and return the file path."""
        directory = self.config.norm_stats_dir
        path = os.path.join(directory, f"fold_{fold_index:03d}_norm_stats.json")
        stats.save(path)
        return path

    def run(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable[[pd.DataFrame, pd.DataFrame, int], np.ndarray],
    ) -> WalkForwardReport:
        """
        Execute walk-forward validation across all folds.

        Args:
            data: Full dataset (time-ordered rows).
            strategy_fn: Callable(train_df, test_df, fold_index) -> OOS returns array.

        Returns:
            WalkForwardReport with all fold results and deployment recommendation.
        """
        folds = self.generate_folds(len(data))

        if not folds:
            raise ValueError(
                f"Not enough data ({len(data)} rows) for walk-forward with "
                f"IS={self.config.is_window}, OOS={self.config.oos_window}, "
                f"embargo={self.config.embargo_bars}"
            )

        logger.info(
            "Walk-forward: %d folds, IS=%d, OOS=%d, embargo=%d, freq=%s",
            len(folds), self.config.is_window, self.config.oos_window,
            self.config.embargo_bars, self.config.frequency,
        )

        fold_results: List[FoldResult] = []
        ann_factor = ANNUALIZATION_FACTORS.get(
            self.config.frequency, ANNUALIZATION_FACTORS["daily"],
        )

        for i, ((is_start, is_end), (oos_start, oos_end)) in enumerate(folds):
            train_data = data.iloc[is_start:is_end]
            test_data = data.iloc[oos_start:oos_end]

            # Compute and save normalization stats from IS data
            norm_stats = self.compute_normalization_stats(train_data, fold_index=i)
            norm_path = self._save_norm_stats(norm_stats, fold_index=i)

            # Run strategy
            try:
                oos_returns = strategy_fn(train_data, test_data, i)
                oos_returns = np.asarray(oos_returns, dtype=float)
            except Exception as e:
                logger.error("Fold %d strategy_fn failed: %s", i, e)
                oos_returns = np.array([])

            # Compute fold metrics
            sharpe = compute_sharpe(oos_returns, self.config.frequency)
            max_dd = compute_max_drawdown(oos_returns)
            win_rate = compute_win_rate(oos_returns)
            pf = compute_profit_factor(oos_returns)
            n_trades = int(np.sum(oos_returns != 0))
            total_return = float(np.sum(oos_returns)) if len(oos_returns) > 0 else 0.0

            result = FoldResult(
                fold_index=i,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
                embargo_bars=self.config.embargo_bars,
                sharpe=sharpe,
                max_drawdown=max_dd,
                win_rate=win_rate,
                profit_factor=pf,
                n_trades=n_trades,
                total_return=total_return,
                annualization_factor=ann_factor,
                frequency=self.config.frequency,
                norm_stats_path=norm_path,
            )
            fold_results.append(result)

            logger.info(
                "Fold %d: Sharpe=%.4f, MaxDD=%.2f%%, WinRate=%.1f%%, "
                "PF=%.2f, Trades=%d, Return=%.2f%%",
                i, sharpe, max_dd * 100, win_rate * 100,
                pf, n_trades, total_return * 100,
            )

        # CF-1 fix: ALWAYS deploy the most recent fold
        deployed_index = fold_results[-1].fold_index

        # Find best Sharpe fold
        sharpes = [f.sharpe for f in fold_results]
        best_sharpe_index = int(np.argmax(sharpes))
        sharpe_mean = float(np.mean(sharpes))
        sharpe_std = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
        sharpe_stability = (1.0 - sharpe_std / abs(sharpe_mean)) if abs(sharpe_mean) > 1e-8 else 0.0

        # Warnings
        warn_list: List[str] = []
        deployed_sharpe = fold_results[-1].sharpe
        best_sharpe = sharpes[best_sharpe_index]

        if best_sharpe - deployed_sharpe > self.config.sharpe_warning_threshold:
            msg = (
                f"Most recent fold (Sharpe={deployed_sharpe:.4f}) is "
                f"{best_sharpe - deployed_sharpe:.4f} below the best fold "
                f"(fold {best_sharpe_index}, Sharpe={best_sharpe:.4f}). "
                f"Model may be degrading."
            )
            warn_list.append(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
            logger.warning("WALK-FORWARD WARNING: %s", msg)

        if deployed_sharpe < self.config.min_sharpe:
            msg = (
                f"Deployed fold Sharpe ({deployed_sharpe:.4f}) is below "
                f"minimum threshold ({self.config.min_sharpe})"
            )
            warn_list.append(msg)
            logger.warning("WALK-FORWARD WARNING: %s", msg)

        report = WalkForwardReport(
            folds=fold_results,
            deployed_fold_index=deployed_index,
            best_sharpe_fold_index=best_sharpe_index,
            sharpe_mean=sharpe_mean,
            sharpe_std=sharpe_std,
            sharpe_stability=sharpe_stability,
            total_folds=len(fold_results),
            config=asdict(self.config),
            warnings=warn_list,
        )

        logger.info(
            "Walk-forward complete: %d folds, deployed=fold_%d (Sharpe=%.4f), "
            "best=fold_%d (Sharpe=%.4f), stability=%.4f",
            len(fold_results), deployed_index, deployed_sharpe,
            best_sharpe_index, best_sharpe, sharpe_stability,
        )

        return report

    @staticmethod
    def load_deployed_norm_stats(report: WalkForwardReport) -> NormalizationStats:
        """
        Load normalization stats for the deployed fold.

        Raises FileNotFoundError if the sidecar file is missing.
        """
        deployed = report.folds[report.deployed_fold_index]
        path = deployed.norm_stats_path
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(
                f"Normalization stats not found for deployed fold "
                f"{report.deployed_fold_index}. Expected at: {path}. "
                f"Cannot run inference without matching normalization stats."
            )
        return NormalizationStats.load(path)
