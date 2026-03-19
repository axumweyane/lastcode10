"""
Pairs Trading Strategy — Statistical Arbitrage via cointegrated spread trading.

This is a market-neutral strategy: for each pair, the long and short legs
offset, so the P&L comes from the spread converging, not from market direction.
Near-zero beta to SPY is the key property that makes this strategy valuable
in the ensemble — it provides returns that are uncorrelated with directional
strategies (TFT, momentum).

Entry/exit logic:
  - ENTER when spread z-score exceeds entry threshold (|z| > 2.0)
      If z > +2.0: spread is too wide → short A, long B (expect convergence)
      If z < -2.0: spread is too narrow → long A, short B
  - EXIT when spread z-score reverts (|z| < 0.5)
  - STOP-LOSS when spread diverges further (|z| > 4.0) — the pair may have
    broken (structural change, merger, etc.)

Position sizing:
  - Each pair is dollar-neutral: $X long one leg, $X short the other
  - Hedge ratio from OLS determines the share ratio
  - Max position per pair capped at config.max_position_per_pair of portfolio
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base import (
    AlphaScore,
    BaseStrategy,
    SignalDirection,
    StrategyOutput,
    StrategyPerformance,
)
from strategies.config import StatArbConfig
from strategies.statarb.scanner import PairScanner, TradingPair

logger = logging.getLogger(__name__)


class PairState:
    """Tracks the live state of a single pair position."""
    FLAT = "flat"
    LONG_SPREAD = "long_spread"    # long A, short B (entered on z < -entry)
    SHORT_SPREAD = "short_spread"  # short A, long B (entered on z > +entry)


@dataclass
class ActivePair:
    """Runtime state for a pair that may or may not have an open position."""
    pair: TradingPair
    state: str = PairState.FLAT
    entry_zscore: float = 0.0
    entry_spread: float = 0.0
    entry_date: Optional[datetime] = None
    cumulative_pnl: float = 0.0
    last_zscore: float = 0.0
    last_spread: float = 0.0


class PairsTrading(BaseStrategy):
    """
    Statistical arbitrage via cointegrated pairs.

    Lifecycle:
        1. initialize() — run PairScanner to find cointegrated pairs
        2. generate_signals() — compute spread z-scores, emit entry/exit signals
        3. Periodically re-scan (every rescan_interval_days) to refresh pairs
    """

    def __init__(self, config: Optional[StatArbConfig] = None):
        self.config = config or StatArbConfig.from_env()
        self._scanner = PairScanner(self.config)
        self._performance = StrategyPerformance(strategy_name=self.name)
        self._active_pairs: Dict[str, ActivePair] = {}
        self._last_scan_date: Optional[datetime] = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "pairs_trading"

    @property
    def description(self) -> str:
        return (
            "Market-neutral statistical arbitrage via cointegrated pairs. "
            "Profits from mean-reverting spreads between correlated stocks."
        )

    def initialize(
        self,
        historical_data: pd.DataFrame,
        sector_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Run initial pair scan on historical data.

        Args:
            historical_data: DataFrame with [symbol, timestamp, close].
            sector_mapping: Optional {symbol: sector} for same-sector filtering.
        """
        logger.info("Initializing %s", self.name)

        pairs = self._scanner.scan(historical_data, sector_mapping)
        self._active_pairs = {
            p.pair_id: ActivePair(pair=p) for p in pairs
        }
        self._last_scan_date = datetime.now(timezone.utc)
        self._initialized = True

        logger.info(
            "%s initialized with %d pairs", self.name, len(self._active_pairs)
        )

    def generate_signals(self, data: pd.DataFrame) -> StrategyOutput:
        """
        Generate entry/exit signals for all active pairs.

        For the ensemble combiner, we translate pair signals into per-symbol
        AlphaScores:
          - If we want to go LONG symbol X in a pair, it gets a positive score
          - If we want to go SHORT symbol X, it gets a negative score
          - The score magnitude reflects the z-score extremity (higher = stronger)

        Args:
            data: DataFrame with [symbol, timestamp, close] for the latest period.

        Returns:
            StrategyOutput with per-symbol AlphaScores.
        """
        if not self._initialized:
            self.initialize(data)

        # Check if we need to re-scan
        if self._should_rescan():
            logger.info("Re-scanning pairs (interval reached)")
            self._rescan(data)

        # Get latest prices per symbol
        latest_prices = self._get_latest_prices(data)

        # Update spread z-scores and generate signals for each pair
        symbol_scores: Dict[str, _SymbolAccumulator] = {}

        for pair_id, active in self._active_pairs.items():
            pair = active.pair

            price_a = latest_prices.get(pair.symbol_a)
            price_b = latest_prices.get(pair.symbol_b)

            if price_a is None or price_b is None:
                continue

            # Recompute spread statistics from recent data
            self._update_spread_stats(active, data)

            # Current spread and z-score
            current_spread = pair.spread(price_a, price_b)
            zscore = pair.zscore(current_spread)
            active.last_zscore = zscore
            active.last_spread = current_spread

            # Determine action
            action = self._evaluate_pair(active, zscore)

            if action == "no_action":
                continue

            # Translate pair action to per-symbol scores
            score_magnitude = min(abs(zscore) / 3.0, 1.0)  # normalize to 0-1ish

            if action == "enter_short_spread":
                # z > +entry: spread too wide → short A, long B
                self._accumulate(symbol_scores, pair.symbol_a,
                                 -score_magnitude, zscore, pair_id, "short_leg")
                self._accumulate(symbol_scores, pair.symbol_b,
                                 +score_magnitude, zscore, pair_id, "long_leg")
                active.state = PairState.SHORT_SPREAD
                active.entry_zscore = zscore
                active.entry_spread = current_spread
                active.entry_date = datetime.now(timezone.utc)

            elif action == "enter_long_spread":
                # z < -entry: spread too narrow → long A, short B
                self._accumulate(symbol_scores, pair.symbol_a,
                                 +score_magnitude, zscore, pair_id, "long_leg")
                self._accumulate(symbol_scores, pair.symbol_b,
                                 -score_magnitude, zscore, pair_id, "short_leg")
                active.state = PairState.LONG_SPREAD
                active.entry_zscore = zscore
                active.entry_spread = current_spread
                active.entry_date = datetime.now(timezone.utc)

            elif action == "exit":
                # Close: produce scores that offset the position
                if active.state == PairState.SHORT_SPREAD:
                    # Was short A, long B → now close (buy A, sell B)
                    self._accumulate(symbol_scores, pair.symbol_a,
                                     +0.1, zscore, pair_id, "exit_cover")
                    self._accumulate(symbol_scores, pair.symbol_b,
                                     -0.1, zscore, pair_id, "exit_sell")
                elif active.state == PairState.LONG_SPREAD:
                    self._accumulate(symbol_scores, pair.symbol_a,
                                     -0.1, zscore, pair_id, "exit_sell")
                    self._accumulate(symbol_scores, pair.symbol_b,
                                     +0.1, zscore, pair_id, "exit_cover")

                # Track P&L
                spread_pnl = self._estimate_pair_pnl(active, current_spread)
                active.cumulative_pnl += spread_pnl
                active.state = PairState.FLAT
                active.entry_zscore = 0.0
                active.entry_spread = 0.0
                active.entry_date = None

            elif action == "stop_loss":
                # Emergency exit — same as exit but logged differently
                logger.warning(
                    "STOP LOSS on pair %s: z=%.2f exceeded %.1f",
                    pair_id, zscore, self.config.stop_loss_zscore,
                )
                if active.state == PairState.SHORT_SPREAD:
                    self._accumulate(symbol_scores, pair.symbol_a,
                                     +0.1, zscore, pair_id, "stop_cover")
                    self._accumulate(symbol_scores, pair.symbol_b,
                                     -0.1, zscore, pair_id, "stop_sell")
                elif active.state == PairState.LONG_SPREAD:
                    self._accumulate(symbol_scores, pair.symbol_a,
                                     -0.1, zscore, pair_id, "stop_sell")
                    self._accumulate(symbol_scores, pair.symbol_b,
                                     +0.1, zscore, pair_id, "stop_cover")

                spread_pnl = self._estimate_pair_pnl(active, current_spread)
                active.cumulative_pnl += spread_pnl
                active.state = PairState.FLAT
                active.entry_zscore = 0.0
                active.entry_spread = 0.0
                active.entry_date = None

        # Convert accumulated scores to AlphaScores
        alpha_scores = self._finalize_scores(symbol_scores)

        longs = [s for s in alpha_scores if s.direction == SignalDirection.LONG]
        shorts = [s for s in alpha_scores if s.direction == SignalDirection.SHORT]

        active_positions = sum(
            1 for a in self._active_pairs.values() if a.state != PairState.FLAT
        )

        logger.info(
            "%s: %d signals (%d long, %d short), %d active pair positions, %d total pairs",
            self.name, len(alpha_scores), len(longs), len(shorts),
            active_positions, len(self._active_pairs),
        )

        return StrategyOutput(
            strategy_name=self.name,
            timestamp=datetime.now(timezone.utc),
            scores=alpha_scores,
            strategy_sharpe_63d=self._performance.sharpe_63d,
            strategy_sharpe_21d=self._performance.sharpe_21d,
            metadata={
                "total_pairs": len(self._active_pairs),
                "active_positions": active_positions,
                "long_count": len(longs),
                "short_count": len(shorts),
                "last_scan_date": str(self._last_scan_date),
            },
        )

    def get_performance(self) -> StrategyPerformance:
        return self._performance

    def get_active_pairs_summary(self) -> List[Dict]:
        """Return summary of all active pairs for monitoring."""
        summary = []
        for pair_id, active in self._active_pairs.items():
            summary.append({
                "pair_id": pair_id,
                "symbol_a": active.pair.symbol_a,
                "symbol_b": active.pair.symbol_b,
                "state": active.state,
                "last_zscore": round(active.last_zscore, 3),
                "hedge_ratio": round(active.pair.hedge_ratio, 4),
                "half_life": round(active.pair.half_life, 1),
                "coint_pvalue": round(active.pair.coint_pvalue, 4),
                "cumulative_pnl": round(active.cumulative_pnl, 2),
            })
        return summary

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    def _evaluate_pair(self, active: ActivePair, zscore: float) -> str:
        """
        State machine for a single pair.

        Returns one of: "enter_long_spread", "enter_short_spread",
                        "exit", "stop_loss", "no_action"
        """
        z = zscore
        entry = self.config.entry_zscore
        exit_z = self.config.exit_zscore
        stop = self.config.stop_loss_zscore

        if active.state == PairState.FLAT:
            # Check for entry
            if z > entry:
                return "enter_short_spread"
            elif z < -entry:
                return "enter_long_spread"
            return "no_action"

        else:
            # We have a position — check exit conditions
            # Stop loss: spread diverged further
            if active.state == PairState.SHORT_SPREAD and z > stop:
                return "stop_loss"
            if active.state == PairState.LONG_SPREAD and z < -stop:
                return "stop_loss"

            # Profit target: spread reverted
            if active.state == PairState.SHORT_SPREAD and z < exit_z:
                return "exit"
            if active.state == PairState.LONG_SPREAD and z > -exit_z:
                return "exit"

            return "no_action"

    def _update_spread_stats(
        self, active: ActivePair, data: pd.DataFrame,
    ) -> None:
        """
        Recompute rolling spread mean/std from recent data so z-scores
        stay calibrated as the relationship evolves.
        """
        pair = active.pair
        sym_a_data = data[data["symbol"] == pair.symbol_a].sort_values("timestamp")
        sym_b_data = data[data["symbol"] == pair.symbol_b].sort_values("timestamp")

        if sym_a_data.empty or sym_b_data.empty:
            return

        # Align on common dates
        a_prices = sym_a_data.set_index("timestamp")["close"]
        b_prices = sym_b_data.set_index("timestamp")["close"]
        common = a_prices.index.intersection(b_prices.index)

        if len(common) < self.config.lookback_window:
            return

        a_vals = a_prices.loc[common].values
        b_vals = b_prices.loc[common].values

        spread = a_vals - pair.hedge_ratio * b_vals
        recent = spread[-self.config.lookback_window:]

        pair.spread_mean = float(np.mean(recent))
        pair.spread_std = float(np.std(recent))

    def _estimate_pair_pnl(
        self, active: ActivePair, current_spread: float,
    ) -> float:
        """Estimate P&L from spread change since entry."""
        if active.state == PairState.LONG_SPREAD:
            return current_spread - active.entry_spread
        elif active.state == PairState.SHORT_SPREAD:
            return active.entry_spread - current_spread
        return 0.0

    def _should_rescan(self) -> bool:
        """Check if enough time has passed since last scan."""
        if self._last_scan_date is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_scan_date).days
        return elapsed >= self.config.rescan_interval_days

    def _rescan(self, data: pd.DataFrame) -> None:
        """Re-run pair scanner, preserving active positions."""
        # Remember which pairs have open positions
        positioned_pairs = {
            pid: ap for pid, ap in self._active_pairs.items()
            if ap.state != PairState.FLAT
        }

        # Scan for new pairs
        new_pairs = self._scanner.scan(data)
        new_active = {p.pair_id: ActivePair(pair=p) for p in new_pairs}

        # Merge: keep positioned pairs even if they didn't re-scan
        for pid, ap in positioned_pairs.items():
            if pid not in new_active:
                new_active[pid] = ap
                logger.info("Keeping positioned pair %s despite failed re-scan", pid)
            else:
                # Update pair stats but keep position state
                new_active[pid].state = ap.state
                new_active[pid].entry_zscore = ap.entry_zscore
                new_active[pid].entry_spread = ap.entry_spread
                new_active[pid].entry_date = ap.entry_date
                new_active[pid].cumulative_pnl = ap.cumulative_pnl

        self._active_pairs = new_active
        self._last_scan_date = datetime.now(timezone.utc)
        logger.info("Re-scan complete: %d pairs", len(self._active_pairs))

    @staticmethod
    def _get_latest_prices(data: pd.DataFrame) -> Dict[str, float]:
        """Get the most recent close price per symbol."""
        latest_idx = data.groupby("symbol")["timestamp"].idxmax()
        latest = data.loc[latest_idx]
        return dict(zip(latest["symbol"], latest["close"]))

    @staticmethod
    def _accumulate(
        accum: Dict[str, "_SymbolAccumulator"],
        symbol: str,
        score: float,
        zscore: float,
        pair_id: str,
        role: str,
    ) -> None:
        """Accumulate scores for a symbol across multiple pairs."""
        if symbol not in accum:
            accum[symbol] = _SymbolAccumulator(symbol)
        accum[symbol].add(score, zscore, pair_id, role)

    @staticmethod
    def _finalize_scores(
        accum: Dict[str, "_SymbolAccumulator"],
    ) -> List[AlphaScore]:
        """Convert accumulated per-symbol scores to AlphaScores."""
        results = []
        for symbol, acc in accum.items():
            net_score = acc.net_score
            if abs(net_score) < 0.01:
                continue

            direction = SignalDirection.LONG if net_score > 0 else SignalDirection.SHORT
            confidence = min(abs(net_score), 1.0)

            results.append(AlphaScore(
                symbol=symbol,
                score=net_score,
                raw_score=net_score,
                confidence=confidence,
                direction=direction,
                metadata={
                    "pair_count": acc.pair_count,
                    "pairs": acc.pair_details,
                },
            ))
        return results


class _SymbolAccumulator:
    """Accumulates scores for one symbol across multiple pair signals."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.scores: List[float] = []
        self.pair_details: List[Dict] = []

    def add(self, score: float, zscore: float, pair_id: str, role: str) -> None:
        self.scores.append(score)
        self.pair_details.append({
            "pair_id": pair_id,
            "role": role,
            "zscore": round(zscore, 3),
            "score_contribution": round(score, 4),
        })

    @property
    def net_score(self) -> float:
        return sum(self.scores)

    @property
    def pair_count(self) -> int:
        return len(self.scores)
