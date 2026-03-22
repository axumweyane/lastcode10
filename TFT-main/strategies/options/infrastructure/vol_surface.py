"""
Implied volatility surface builder.

Constructs a 2D surface (strike x expiry) of implied volatilities,
fitting SABR or cubic spline through market quotes.

The IV surface is the foundation for:
  - Finding mispriced options (market IV vs model IV)
  - Interpolating IV for strikes/expiries without quotes
  - Tracking skew and term structure changes over time
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d

from strategies.options.infrastructure.chain import ChainEntry, OptionsChain
from strategies.options.infrastructure.pricing import PricingEngine

logger = logging.getLogger(__name__)


@dataclass
class VolSurfacePoint:
    """Single point on the IV surface."""

    strike: float
    expiry_days: int
    implied_vol: float
    moneyness: float  # strike / spot


@dataclass
class VolSurface:
    """Implied volatility surface for one underlying."""

    underlying: str
    spot: float
    points: List[VolSurfacePoint] = field(default_factory=list)
    strikes: Optional[np.ndarray] = None  # unique strikes
    expiry_days: Optional[np.ndarray] = None  # unique DTE values
    iv_grid: Optional[np.ndarray] = None  # 2D grid [strikes x expiries]
    _interpolator: Optional[RectBivariateSpline] = None
    build_time: Optional[str] = None

    def get_iv(self, strike: float, dte: int) -> float:
        """Interpolate IV at any strike/DTE point on the surface."""
        if self._interpolator is not None:
            try:
                iv = float(self._interpolator(strike, dte)[0, 0])
                return max(iv, 0.01)
            except Exception:
                pass

        # Fallback: nearest neighbor
        if not self.points:
            return 0.20
        nearest = min(
            self.points,
            key=lambda p: abs(p.strike - strike) + abs(p.expiry_days - dte) * 0.1,
        )
        return nearest.implied_vol

    def get_skew(self, dte: int) -> Optional[Dict]:
        """
        Compute the volatility skew for a given DTE.

        Skew = IV(90% moneyness put) - IV(ATM)
        A negative skew (puts more expensive) is normal and indicates
        demand for downside protection.
        """
        if self._interpolator is None or self.spot <= 0:
            return None

        atm_strike = self.spot
        otm_put_strike = self.spot * 0.90  # 10% OTM put
        otm_call_strike = self.spot * 1.10  # 10% OTM call

        atm_iv = self.get_iv(atm_strike, dte)
        put_iv = self.get_iv(otm_put_strike, dte)
        call_iv = self.get_iv(otm_call_strike, dte)

        return {
            "atm_iv": round(atm_iv, 4),
            "put_10pct_iv": round(put_iv, 4),
            "call_10pct_iv": round(call_iv, 4),
            "put_skew": round(put_iv - atm_iv, 4),
            "call_skew": round(call_iv - atm_iv, 4),
            "skew_ratio": round(put_iv / call_iv, 4) if call_iv > 0 else 0,
        }

    def get_term_structure(self) -> Optional[Dict]:
        """
        IV term structure: ATM IV at each expiry.

        Normal (contango): longer-dated options have higher IV.
        Inverted (backwardation): short-dated IV > long-dated IV (fear).
        """
        if self._interpolator is None or self.expiry_days is None:
            return None

        atm = self.spot
        term = {}
        for dte in sorted(self.expiry_days):
            iv = self.get_iv(atm, int(dte))
            term[int(dte)] = round(iv, 4)

        dtes = sorted(term.keys())
        if len(dtes) >= 2:
            is_contango = term[dtes[-1]] > term[dtes[0]]
        else:
            is_contango = True

        return {
            "term_structure": term,
            "is_contango": is_contango,
            "front_iv": term.get(dtes[0], 0) if dtes else 0,
            "back_iv": term.get(dtes[-1], 0) if dtes else 0,
        }


class VolSurfaceBuilder:
    """
    Builds an IV surface from options chain data.

    Usage:
        builder = VolSurfaceBuilder()
        surface = builder.build("AAPL", chains, spot=150.0)
        iv = surface.get_iv(strike=145, dte=30)
    """

    def __init__(self, pricing_engine: Optional[PricingEngine] = None):
        self.engine = pricing_engine or PricingEngine()

    def build(
        self,
        underlying: str,
        chains: List[OptionsChain],
        spot: float,
        use_mid_prices: bool = True,
    ) -> VolSurface:
        """
        Build IV surface from multiple option chains.

        Extracts IV from market prices, builds a grid, and fits
        a smooth interpolator.
        """
        points: List[VolSurfacePoint] = []

        for chain in chains:
            dte = chain.dte
            if dte <= 0:
                continue

            entries = chain.calls + chain.puts
            for entry in entries:
                iv = entry.implied_vol
                if iv <= 0.005 or iv > 3.0:
                    # Try to compute IV from market price
                    price = entry.mid if use_mid_prices else entry.last
                    if price > 0:
                        computed_iv = self.engine.implied_vol(
                            entry.contract,
                            spot,
                            price,
                        )
                        if computed_iv is not None:
                            iv = computed_iv
                        else:
                            continue
                    else:
                        continue

                moneyness = entry.contract.strike / spot if spot > 0 else 1.0

                # Filter extreme moneyness (deep OTM/ITM have noisy IV)
                if moneyness < 0.7 or moneyness > 1.3:
                    continue

                points.append(
                    VolSurfacePoint(
                        strike=entry.contract.strike,
                        expiry_days=dte,
                        implied_vol=iv,
                        moneyness=moneyness,
                    )
                )

        if len(points) < 6:
            logger.warning(
                "Only %d valid IV points for %s — surface may be sparse",
                len(points),
                underlying,
            )
            return VolSurface(underlying=underlying, spot=spot, points=points)

        # Build grid and interpolator
        surface = self._fit_surface(underlying, spot, points)
        return surface

    def _fit_surface(
        self,
        underlying: str,
        spot: float,
        points: List[VolSurfacePoint],
    ) -> VolSurface:
        """Fit a smooth surface through the IV points."""
        strikes = np.array(sorted(set(p.strike for p in points)))
        dtes = np.array(sorted(set(p.expiry_days for p in points)))

        if len(strikes) < 2 or len(dtes) < 2:
            return VolSurface(
                underlying=underlying,
                spot=spot,
                points=points,
                strikes=strikes,
                expiry_days=dtes,
            )

        # Build 2D grid by averaging IVs at each (strike, dte) node
        iv_grid = np.full((len(strikes), len(dtes)), np.nan)
        strike_idx = {s: i for i, s in enumerate(strikes)}
        dte_idx = {d: i for i, d in enumerate(dtes)}

        for p in points:
            si = strike_idx.get(p.strike)
            di = dte_idx.get(p.expiry_days)
            if si is not None and di is not None:
                if np.isnan(iv_grid[si, di]):
                    iv_grid[si, di] = p.implied_vol
                else:
                    iv_grid[si, di] = (iv_grid[si, di] + p.implied_vol) / 2

        # Interpolate NaN gaps column-by-column (per expiry)
        for j in range(len(dtes)):
            col = iv_grid[:, j]
            valid = ~np.isnan(col)
            if valid.sum() >= 2:
                f = interp1d(
                    strikes[valid],
                    col[valid],
                    kind="linear",
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                iv_grid[:, j] = f(strikes)
            elif valid.sum() == 1:
                iv_grid[:, j] = col[valid][0]

        # Fill remaining NaN rows
        for i in range(len(strikes)):
            row = iv_grid[i, :]
            valid = ~np.isnan(row)
            if valid.sum() >= 2:
                f = interp1d(
                    dtes[valid],
                    row[valid],
                    kind="linear",
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                iv_grid[i, :] = f(dtes)
            elif valid.sum() == 1:
                iv_grid[i, :] = row[valid][0]

        # Final NaN cleanup
        global_mean = np.nanmean(iv_grid)
        if np.isnan(global_mean):
            global_mean = 0.20
        iv_grid = np.nan_to_num(iv_grid, nan=global_mean)
        iv_grid = np.clip(iv_grid, 0.01, 3.0)

        # Fit bicubic spline
        interpolator = None
        try:
            k_s = min(3, len(strikes) - 1)
            k_d = min(3, len(dtes) - 1)
            if k_s >= 1 and k_d >= 1:
                interpolator = RectBivariateSpline(
                    strikes, dtes, iv_grid, kx=k_s, ky=k_d
                )
        except Exception as e:
            logger.debug("Spline fitting failed: %s", e)

        from datetime import datetime

        surface = VolSurface(
            underlying=underlying,
            spot=spot,
            points=points,
            strikes=strikes,
            expiry_days=dtes,
            iv_grid=iv_grid,
            _interpolator=interpolator,
            build_time=datetime.now().isoformat(),
        )

        logger.info(
            "Vol surface built for %s: %d strikes x %d expiries, %d points",
            underlying,
            len(strikes),
            len(dtes),
            len(points),
        )
        return surface
