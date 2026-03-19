"""
Cointegration-based pair scanner for statistical arbitrage.

Scans a universe of stocks to find pairs that are cointegrated — their price
spread is mean-reverting, which means divergences are temporary and tradeable.

Statistical foundation:
  - Engle-Granger (1987) two-step cointegration test
  - Ornstein-Uhlenbeck process for half-life estimation
  - Hedge ratio via OLS regression

Designed to run as a nightly batch job. The output (a list of TradingPair
objects) is consumed by the PairsTrading strategy.
"""

import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats as scipy_stats

from strategies.config import StatArbConfig

logger = logging.getLogger(__name__)


@dataclass
class TradingPair:
    """A validated cointegrated pair ready for trading."""
    symbol_a: str                   # the "Y" leg (dependent variable)
    symbol_b: str                   # the "X" leg (independent variable)
    hedge_ratio: float              # OLS beta: units of B per unit of A
    coint_pvalue: float             # Engle-Granger p-value (lower = stronger)
    half_life: float                # mean-reversion half-life in trading days
    spread_mean: float              # rolling spread mean at scan time
    spread_std: float               # rolling spread std at scan time
    correlation: float              # price correlation
    sector_a: str = ""
    sector_b: str = ""
    scan_date: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def pair_id(self) -> str:
        """Canonical pair identifier (alphabetically sorted)."""
        a, b = sorted([self.symbol_a, self.symbol_b])
        return f"{a}_{b}"

    def spread(self, price_a: float, price_b: float) -> float:
        """Compute the spread: price_a - hedge_ratio * price_b."""
        return price_a - self.hedge_ratio * price_b

    def zscore(self, current_spread: float) -> float:
        """Z-score of the current spread relative to historical distribution."""
        if self.spread_std <= 1e-10:
            return 0.0
        return (current_spread - self.spread_mean) / self.spread_std


class PairScanner:
    """
    Scans a stock universe for cointegrated pairs.

    Usage:
        scanner = PairScanner(config)
        pairs = scanner.scan(price_data, sector_mapping)
    """

    def __init__(self, config: Optional[StatArbConfig] = None):
        self.config = config or StatArbConfig.from_env()

    def scan(
        self,
        prices: pd.DataFrame,
        sector_mapping: Optional[Dict[str, str]] = None,
    ) -> List[TradingPair]:
        """
        Run full cointegration scan on the price universe.

        Args:
            prices: DataFrame with [symbol, timestamp, close], sorted by
                    (symbol, timestamp). Should have >= lookback_window days.
            sector_mapping: Optional {symbol: sector} dict for same-sector filtering.

        Returns:
            List of TradingPair objects, sorted by cointegration p-value (best first),
            capped at max_pairs.
        """
        symbols = prices["symbol"].unique().tolist()
        logger.info(
            "Pair scan starting: %d symbols, %d possible pairs",
            len(symbols),
            len(symbols) * (len(symbols) - 1) // 2,
        )

        # Pivot to wide format: dates x symbols
        wide = prices.pivot_table(
            index="timestamp", columns="symbol", values="close"
        )
        # Drop symbols with too many NaN values
        min_obs = self.config.lookback_window
        wide = wide.dropna(axis=1, thresh=min_obs)
        valid_symbols = wide.columns.tolist()

        if len(valid_symbols) < 2:
            logger.warning("Fewer than 2 valid symbols after NaN filtering")
            return []

        # Generate candidate pairs
        candidates = self._generate_candidates(valid_symbols, sector_mapping)
        logger.info("Testing %d candidate pairs", len(candidates))

        # Test each pair
        validated: List[TradingPair] = []
        sector_counts: Dict[str, int] = {}

        for sym_a, sym_b in candidates:
            series_a = wide[sym_a].dropna()
            series_b = wide[sym_b].dropna()

            # Align on common dates
            common_idx = series_a.index.intersection(series_b.index)
            if len(common_idx) < min_obs:
                continue

            a = series_a.loc[common_idx].values
            b = series_b.loc[common_idx].values

            pair = self._test_pair(sym_a, sym_b, a, b)
            if pair is None:
                continue

            # Apply sector info
            if sector_mapping:
                pair.sector_a = sector_mapping.get(sym_a, "Unknown")
                pair.sector_b = sector_mapping.get(sym_b, "Unknown")

                # Sector pair limit
                sector_key = f"{pair.sector_a}_{pair.sector_b}"
                if sector_counts.get(sector_key, 0) >= self.config.sector_pairs_limit:
                    continue
                sector_counts[sector_key] = sector_counts.get(sector_key, 0) + 1

            pair.scan_date = datetime.now(timezone.utc)
            validated.append(pair)

        # Sort by p-value (best first), cap at max_pairs
        validated.sort(key=lambda p: p.coint_pvalue)
        validated = validated[:self.config.max_pairs]

        logger.info(
            "Pair scan complete: %d valid pairs from %d candidates",
            len(validated), len(candidates),
        )

        for p in validated[:5]:
            logger.info(
                "  %s/%s: pval=%.4f, HL=%.1fd, hedge=%.3f, corr=%.3f",
                p.symbol_a, p.symbol_b, p.coint_pvalue,
                p.half_life, p.hedge_ratio, p.correlation,
            )

        return validated

    def _generate_candidates(
        self,
        symbols: List[str],
        sector_mapping: Optional[Dict[str, str]],
    ) -> List[Tuple[str, str]]:
        """Generate pair candidates, optionally filtered to same-sector."""
        if self.config.same_sector_only and sector_mapping:
            # Group symbols by sector
            sector_groups: Dict[str, List[str]] = {}
            for sym in symbols:
                sector = sector_mapping.get(sym, "Unknown")
                sector_groups.setdefault(sector, []).append(sym)

            candidates = []
            for sector, syms in sector_groups.items():
                if len(syms) >= 2:
                    candidates.extend(itertools.combinations(sorted(syms), 2))
            return candidates
        else:
            return list(itertools.combinations(sorted(symbols), 2))

    def _test_pair(
        self,
        sym_a: str,
        sym_b: str,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
    ) -> Optional[TradingPair]:
        """
        Test a single pair for cointegration.

        Steps:
            1. Engle-Granger cointegration test
            2. OLS hedge ratio
            3. Spread half-life via AR(1)
            4. Filter on p-value and half-life bounds
        """
        # 1. Engle-Granger cointegration test
        try:
            coint_stat, pvalue, crit_values = coint(prices_a, prices_b)
        except Exception as e:
            logger.debug("Cointegration test failed for %s/%s: %s", sym_a, sym_b, e)
            return None

        if pvalue > self.config.cointegration_pvalue:
            return None

        # 2. OLS hedge ratio: A = alpha + beta * B + epsilon
        hedge_ratio, intercept = _ols_hedge_ratio(prices_a, prices_b)

        # 3. Compute spread and its statistics
        spread = prices_a - hedge_ratio * prices_b
        lookback = self.config.lookback_window
        spread_recent = spread[-lookback:]
        spread_mean = float(np.mean(spread_recent))
        spread_std = float(np.std(spread_recent))

        # 4. Half-life of mean reversion via AR(1) on spread
        half_life = _compute_half_life(spread)
        if half_life is None:
            return None

        if half_life < self.config.min_half_life_days:
            return None
        if half_life > self.config.max_half_life_days:
            return None

        # 5. Correlation (for diagnostics)
        correlation = float(np.corrcoef(prices_a, prices_b)[0, 1])

        # 6. Verify spread is stationary (ADF on spread as additional check)
        try:
            adf_stat, adf_pval, *_ = adfuller(spread, maxlag=int(half_life))
            if adf_pval > 0.10:  # relaxed threshold since we already passed coint
                return None
        except Exception:
            return None

        return TradingPair(
            symbol_a=sym_a,
            symbol_b=sym_b,
            hedge_ratio=hedge_ratio,
            coint_pvalue=pvalue,
            half_life=half_life,
            spread_mean=spread_mean,
            spread_std=spread_std,
            correlation=correlation,
            metadata={
                "intercept": float(intercept),
                "coint_stat": float(coint_stat),
                "adf_pval": float(adf_pval),
                "spread_skew": float(scipy_stats.skew(spread_recent)),
                "spread_kurtosis": float(scipy_stats.kurtosis(spread_recent)),
            },
        )


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _ols_hedge_ratio(
    y: np.ndarray, x: np.ndarray,
) -> Tuple[float, float]:
    """
    Ordinary Least Squares: y = intercept + beta * x.
    Returns (beta, intercept).
    """
    x_with_const = np.column_stack([np.ones(len(x)), x])
    # Use lstsq for numerical stability
    result, _, _, _ = np.linalg.lstsq(x_with_const, y, rcond=None)
    intercept, beta = result[0], result[1]
    return float(beta), float(intercept)


def _compute_half_life(spread: np.ndarray) -> Optional[float]:
    """
    Half-life of mean reversion from an Ornstein-Uhlenbeck process.

    Fits AR(1) model: spread_t - spread_{t-1} = phi * spread_{t-1} + epsilon
    Half-life = -ln(2) / ln(1 + phi)

    If phi >= 0 (no mean reversion), returns None.
    """
    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)

    if len(spread_lag) < 10:
        return None

    # OLS: spread_diff = phi * spread_lag
    x = spread_lag.reshape(-1, 1)
    x_with_const = np.column_stack([np.ones(len(x)), x])
    result, _, _, _ = np.linalg.lstsq(x_with_const, spread_diff, rcond=None)
    phi = result[1]

    if phi >= 0:
        # No mean reversion
        return None

    # Half-life = -ln(2) / ln(1 + phi)
    try:
        half_life = -np.log(2) / np.log(1 + phi)
    except (ValueError, RuntimeWarning):
        return None

    if half_life <= 0 or np.isnan(half_life) or np.isinf(half_life):
        return None

    return float(half_life)
