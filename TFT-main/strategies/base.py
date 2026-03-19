"""
Abstract strategy interface for APEX multi-strategy system.

Every strategy produces a DataFrame of per-symbol alpha scores (z-scored).
The ensemble combiner consumes these scores to build the final portfolio.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class StrategyState(str, Enum):
    DISABLED = "disabled"
    ACTIVE = "active"
    PAPER_ONLY = "paper_only"
    KILLED = "killed"  # per-strategy kill switch triggered


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class AlphaScore:
    """Per-symbol alpha output from a strategy."""
    symbol: str
    score: float          # z-scored alpha (positive = long, negative = short)
    raw_score: float      # pre-normalization score for diagnostics
    confidence: float     # 0-1, how much the strategy trusts this signal
    direction: SignalDirection
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyOutput:
    """Full output from a strategy's generate_signals() call."""
    strategy_name: str
    timestamp: datetime
    scores: List[AlphaScore]
    strategy_sharpe_63d: float = 0.0   # rolling 63-day Sharpe for weighting
    strategy_sharpe_21d: float = 0.0   # rolling 21-day Sharpe for fast regime adapt
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert scores to DataFrame for easy combination."""
        if not self.scores:
            return pd.DataFrame(columns=["symbol", "score", "raw_score",
                                         "confidence", "direction"])
        rows = []
        for s in self.scores:
            rows.append({
                "symbol": s.symbol,
                "score": s.score,
                "raw_score": s.raw_score,
                "confidence": s.confidence,
                "direction": s.direction.value,
            })
        return pd.DataFrame(rows)


@dataclass
class StrategyPerformance:
    """Rolling performance tracking for a strategy."""
    strategy_name: str
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    cumulative_pnl: float = 0.0
    peak_pnl: float = 0.0
    current_drawdown: float = 0.0
    sharpe_21d: float = 0.0
    sharpe_63d: float = 0.0
    win_rate_21d: float = 0.0
    trade_count: int = 0
    is_killed: bool = False
    kill_reason: Optional[str] = None

    def update(self, daily_return: float) -> None:
        """Append a daily return and recalculate metrics."""
        self.daily_returns = pd.concat([
            self.daily_returns,
            pd.Series([daily_return])
        ]).reset_index(drop=True)

        self.cumulative_pnl += daily_return
        self.peak_pnl = max(self.peak_pnl, self.cumulative_pnl)

        if self.peak_pnl > 0:
            self.current_drawdown = (self.peak_pnl - self.cumulative_pnl) / self.peak_pnl
        else:
            self.current_drawdown = 0.0

        recent_63 = self.daily_returns.tail(63)
        recent_21 = self.daily_returns.tail(21)

        if len(recent_63) >= 10 and recent_63.std() > 0:
            self.sharpe_63d = (recent_63.mean() / recent_63.std()) * np.sqrt(252)
        if len(recent_21) >= 5 and recent_21.std() > 0:
            self.sharpe_21d = (recent_21.mean() / recent_21.std()) * np.sqrt(252)
        if len(recent_21) > 0:
            self.win_rate_21d = (recent_21 > 0).sum() / len(recent_21)


class BaseStrategy(ABC):
    """
    Abstract base class for all APEX trading strategies.

    Lifecycle:
        1. __init__(config) — load parameters
        2. initialize(historical_data) — warm up indicators, fit models
        3. generate_signals(current_data) — produce alpha scores
        4. get_performance() — return rolling performance for ensemble weighting
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""

    @abstractmethod
    def initialize(self, historical_data: pd.DataFrame) -> None:
        """
        Warm up the strategy with historical data.

        Args:
            historical_data: DataFrame with columns including at minimum:
                symbol, timestamp, open, high, low, close, volume
                Plus any features the strategy needs.
        """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        """
        Generate alpha scores for the current period.

        Args:
            data: DataFrame with the latest market data (same schema as initialize).

        Returns:
            StrategyOutput with per-symbol AlphaScores.
        """

    @abstractmethod
    def get_performance(self) -> StrategyPerformance:
        """Return rolling performance metrics for ensemble weighting."""

    def should_be_killed(self, max_drawdown: float = 0.20,
                         min_sharpe: float = -1.0) -> Optional[str]:
        """
        Check per-strategy kill switch conditions.
        Returns kill reason string if triggered, None otherwise.
        """
        perf = self.get_performance()
        if perf.current_drawdown >= max_drawdown:
            return (f"Strategy drawdown {perf.current_drawdown:.1%} "
                    f"exceeds limit {max_drawdown:.1%}")
        if len(perf.daily_returns) >= 21 and perf.sharpe_21d < min_sharpe:
            return (f"21-day Sharpe {perf.sharpe_21d:.2f} "
                    f"below kill threshold {min_sharpe:.2f}")
        return None
