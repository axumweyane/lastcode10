"""
Risk-parity portfolio optimizer for the APEX ensemble.

Takes combined alpha scores from the EnsembleCombiner and produces
a final set of target portfolio weights subject to:
    - Volatility targeting (scale positions to hit target annualized vol)
    - Gross leverage cap (sum of |weights| <= max)
    - Net leverage cap (|sum of weights| <= max)
    - Per-position size cap
    - VaR constraint (99% parametric VaR)
    - Regime-based exposure scaling

This replaces the existing PortfolioConstructor for multi-strategy use
while the original remains available for TFT-only operation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from strategies.config import EnsembleConfig
from strategies.ensemble.combiner import CombinedSignal
from strategies.regime.detector import RegimeState

logger = logging.getLogger(__name__)


@dataclass
class PortfolioPosition:
    """Target position for a single symbol."""
    symbol: str
    target_weight: float        # fraction of portfolio (-1 to +1)
    direction: str              # "long" or "short"
    combined_score: float       # from ensemble
    confidence: float
    vol_adjusted_weight: float  # after vol targeting
    contributing_strategies: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioTarget:
    """Complete target portfolio from the optimizer."""
    positions: List[PortfolioPosition]
    timestamp: datetime
    gross_leverage: float       # sum of |weights|
    net_leverage: float         # sum of weights (signed)
    long_weight: float          # sum of positive weights
    short_weight: float         # sum of |negative weights|
    expected_volatility: float  # annualized portfolio vol estimate
    var_99: float               # 99% 1-day Value at Risk (as % of portfolio)
    regime_exposure_scalar: float
    metadata: Dict = field(default_factory=dict)

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def long_count(self) -> int:
        return sum(1 for p in self.positions if p.direction == "long")

    @property
    def short_count(self) -> int:
        return sum(1 for p in self.positions if p.direction == "short")

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for p in self.positions:
            rows.append({
                "symbol": p.symbol,
                "target_weight": p.target_weight,
                "direction": p.direction,
                "combined_score": p.combined_score,
                "confidence": p.confidence,
            })
        return pd.DataFrame(rows)


class PortfolioOptimizer:
    """
    Converts combined signals into risk-constrained portfolio weights.

    Usage:
        optimizer = PortfolioOptimizer(config)
        target = optimizer.optimize(
            signals=combined_signals,
            price_data=recent_prices,
            regime_state=current_regime,
        )
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig.from_env()

    def optimize(
        self,
        signals: List[CombinedSignal],
        price_data: Optional[pd.DataFrame] = None,
        regime_state: Optional[RegimeState] = None,
    ) -> PortfolioTarget:
        """
        Build target portfolio from combined signals.

        Args:
            signals: CombinedSignal list from EnsembleCombiner.
            price_data: Recent price data [symbol, timestamp, close] for
                       volatility estimation. If None, uses equal vol assumption.
            regime_state: Current market regime for exposure scaling.

        Returns:
            PortfolioTarget with constrained position weights.
        """
        if not signals:
            return self._empty_target(regime_state)

        # 1. Score-based raw weights
        raw_weights = self._score_to_weights(signals)

        # 2. Volatility adjustment
        if price_data is not None:
            vol_weights = self._vol_target_weights(raw_weights, price_data)
        else:
            vol_weights = raw_weights.copy()

        # 3. Regime exposure scaling
        exposure_scalar = 1.0
        if regime_state is not None:
            exposure_scalar = regime_state.exposure_scalar
            vol_weights = {
                sym: w * exposure_scalar for sym, w in vol_weights.items()
            }

        # 4. Apply constraints
        constrained = self._apply_constraints(vol_weights)

        # 5. Compute risk metrics
        portfolio_vol = self._estimate_portfolio_vol(constrained, price_data)
        var_99 = self._compute_var(portfolio_vol)

        # 6. Build positions
        signal_map = {s.symbol: s for s in signals}
        positions = []
        for symbol, weight in constrained.items():
            sig = signal_map.get(symbol)
            positions.append(PortfolioPosition(
                symbol=symbol,
                target_weight=weight,
                direction="long" if weight > 0 else "short",
                combined_score=sig.combined_score if sig else 0.0,
                confidence=sig.confidence if sig else 0.0,
                vol_adjusted_weight=vol_weights.get(symbol, weight),
                contributing_strategies=sig.contributing_strategies if sig else {},
            ))

        # Sort by absolute weight descending
        positions.sort(key=lambda p: abs(p.target_weight), reverse=True)

        long_w = sum(w for w in constrained.values() if w > 0)
        short_w = sum(abs(w) for w in constrained.values() if w < 0)

        target = PortfolioTarget(
            positions=positions,
            timestamp=datetime.now(timezone.utc),
            gross_leverage=long_w + short_w,
            net_leverage=long_w - short_w,
            long_weight=long_w,
            short_weight=short_w,
            expected_volatility=portfolio_vol,
            var_99=var_99,
            regime_exposure_scalar=exposure_scalar,
            metadata={
                "signals_input": len(signals),
                "positions_output": len(positions),
                "target_vol": self.config.target_volatility,
            },
        )

        logger.info(
            "Portfolio: %d positions (%d L / %d S), gross=%.2f, net=%.2f, "
            "vol=%.1f%%, VaR99=%.2f%%, regime_scalar=%.0f%%",
            target.position_count, target.long_count, target.short_count,
            target.gross_leverage, target.net_leverage,
            target.expected_volatility * 100, target.var_99 * 100,
            target.regime_exposure_scalar * 100,
        )

        return target

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def _score_to_weights(
        self, signals: List[CombinedSignal],
    ) -> Dict[str, float]:
        """
        Convert combined scores to initial position weights.

        Weight is proportional to score * confidence, then normalized
        so gross leverage = 1.0 as a starting point.
        """
        raw: Dict[str, float] = {}
        for sig in signals:
            if sig.direction.value == "neutral":
                continue
            raw[sig.symbol] = sig.combined_score * sig.confidence

        # Normalize: gross leverage = 1.0
        total_abs = sum(abs(w) for w in raw.values())
        if total_abs > 0:
            raw = {sym: w / total_abs for sym, w in raw.items()}

        return raw

    def _vol_target_weights(
        self,
        weights: Dict[str, float],
        price_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Scale weights to target portfolio volatility.

        Inverse-volatility weighting: stocks with higher vol get smaller
        positions, so each position contributes roughly equal risk.
        Then scale the whole portfolio so estimated vol = target_vol.
        """
        target_vol = self.config.target_volatility

        # Compute per-symbol realized vol
        symbol_vols: Dict[str, float] = {}
        for symbol in weights:
            sym_data = price_data[price_data["symbol"] == symbol]
            if len(sym_data) >= 21:
                returns = sym_data.sort_values("timestamp")["close"].pct_change().dropna()
                vol = returns.tail(63).std() * np.sqrt(252)
                if pd.notna(vol) and vol > 0.01:
                    symbol_vols[symbol] = float(vol)

        if not symbol_vols:
            return weights

        # Inverse-vol weight adjustment
        adjusted: Dict[str, float] = {}
        for sym, w in weights.items():
            vol = symbol_vols.get(sym, 0.20)  # default 20% vol
            inv_vol_factor = 0.20 / vol  # normalize around 20% baseline
            adjusted[sym] = w * inv_vol_factor

        # Estimate portfolio vol with adjusted weights
        portfolio_vol = self._estimate_portfolio_vol(adjusted, price_data)

        # Scale to target
        if portfolio_vol > 0.01:
            scale = target_vol / portfolio_vol
            adjusted = {sym: w * scale for sym, w in adjusted.items()}

        return adjusted

    def _apply_constraints(
        self, weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply hard constraints:
          1. Per-position cap
          2. Gross leverage cap
          3. Net leverage cap
        """
        max_pos = self.config.max_total_positions
        max_gross = self.config.max_gross_leverage
        max_net = self.config.max_net_leverage

        # Cap individual position size at 1/max_positions minimum,
        # or config-implied max (gross_lev / positions)
        per_pos_cap = max_gross / max(max_pos, 1)

        constrained: Dict[str, float] = {}
        for sym, w in weights.items():
            capped = max(-per_pos_cap, min(per_pos_cap, w))
            if abs(capped) > 0.005:  # minimum 0.5% position
                constrained[sym] = capped

        # Trim to max positions (keep highest |weight|)
        if len(constrained) > max_pos:
            sorted_syms = sorted(
                constrained, key=lambda s: abs(constrained[s]), reverse=True,
            )
            constrained = {s: constrained[s] for s in sorted_syms[:max_pos]}

        # Gross leverage constraint
        gross = sum(abs(w) for w in constrained.values())
        if gross > max_gross:
            scale = max_gross / gross
            constrained = {s: w * scale for s, w in constrained.items()}

        # Net leverage constraint
        net = sum(constrained.values())
        if abs(net) > max_net:
            # Reduce the side that's over-exposed
            excess = abs(net) - max_net
            if net > 0:
                # Too long — trim longs proportionally
                long_total = sum(w for w in constrained.values() if w > 0)
                if long_total > 0:
                    trim = excess / long_total
                    constrained = {
                        s: w * (1 - trim) if w > 0 else w
                        for s, w in constrained.items()
                    }
            else:
                # Too short — trim shorts proportionally
                short_total = sum(abs(w) for w in constrained.values() if w < 0)
                if short_total > 0:
                    trim = excess / short_total
                    constrained = {
                        s: w * (1 - trim) if w < 0 else w
                        for s, w in constrained.items()
                    }

        return constrained

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    def _estimate_portfolio_vol(
        self,
        weights: Dict[str, float],
        price_data: Optional[pd.DataFrame],
    ) -> float:
        """
        Estimate annualized portfolio volatility.

        Uses simplified approach: portfolio_vol ≈ sqrt(w'Σw) where Σ is
        estimated from recent returns. Falls back to weighted-average vol
        if correlation data is insufficient.
        """
        if price_data is None or not weights:
            return 0.15  # default assumption

        symbols = list(weights.keys())
        w = np.array([weights[s] for s in symbols])

        # Build returns matrix
        returns_dict: Dict[str, pd.Series] = {}
        for sym in symbols:
            sym_data = price_data[price_data["symbol"] == sym]
            if len(sym_data) >= 21:
                ret = sym_data.sort_values("timestamp")["close"].pct_change().dropna()
                returns_dict[sym] = ret.tail(63).reset_index(drop=True)

        if len(returns_dict) < 2:
            # Not enough data for covariance — use weighted average of vols
            avg_vol = 0.0
            for sym in symbols:
                if sym in returns_dict:
                    avg_vol += abs(weights[sym]) * returns_dict[sym].std() * np.sqrt(252)
                else:
                    avg_vol += abs(weights[sym]) * 0.20
            return avg_vol

        # Align returns and compute covariance
        returns_df = pd.DataFrame(returns_dict)
        # Only keep symbols present in returns
        common_syms = [s for s in symbols if s in returns_df.columns]
        if len(common_syms) < 2:
            return 0.15

        returns_df = returns_df[common_syms].dropna()
        if len(returns_df) < 10:
            return 0.15

        w_common = np.array([weights[s] for s in common_syms])
        cov_matrix = returns_df.cov().values * 252  # annualize

        try:
            port_var = w_common @ cov_matrix @ w_common
            port_vol = float(np.sqrt(max(port_var, 0)))
        except Exception:
            port_vol = 0.15

        return port_vol

    def _compute_var(self, portfolio_vol: float) -> float:
        """
        Parametric VaR at configured confidence level (default 99%).

        Assumes normal distribution (conservative for fat tails — in
        production, supplement with historical VaR).

        VaR = z_score * daily_vol * portfolio_value
        Returns as fraction of portfolio (e.g., 0.03 = 3%).
        """
        z = scipy_stats.norm.ppf(self.config.var_confidence)
        daily_vol = portfolio_vol / np.sqrt(252)
        return float(z * daily_vol)

    def _empty_target(
        self, regime_state: Optional[RegimeState],
    ) -> PortfolioTarget:
        return PortfolioTarget(
            positions=[],
            timestamp=datetime.now(timezone.utc),
            gross_leverage=0.0,
            net_leverage=0.0,
            long_weight=0.0,
            short_weight=0.0,
            expected_volatility=0.0,
            var_99=0.0,
            regime_exposure_scalar=regime_state.exposure_scalar if regime_state else 1.0,
        )
