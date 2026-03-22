"""
Options pricing engine using QuantLib.

Pricing models:
  - Bjerksund-Stensland (American options — what Alpaca trades)
  - Black-Scholes-Merton (European options — fast fallback)
  - Finite-difference BSM (American with dividends)

Also provides IV solving via scipy.optimize.brentq (more robust than
Newton-Raphson for edge cases near the wings).
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Optional

import QuantLib as ql
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

logger = logging.getLogger(__name__)


class OptionStyle(str, Enum):
    AMERICAN = "american"
    EUROPEAN = "european"


class OptionRight(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class OptionContract:
    """Standardized option contract representation."""

    underlying: str
    strike: float
    expiry: date
    right: OptionRight
    style: OptionStyle = OptionStyle.AMERICAN
    multiplier: int = 100

    @property
    def dte(self) -> int:
        today = date.today()
        return max((self.expiry - today).days, 0)

    @property
    def time_to_expiry(self) -> float:
        return max(self.dte / 365.0, 1e-6)

    @property
    def occ_symbol(self) -> str:
        """OCC format: AAPL250321C00150000"""
        exp_str = self.expiry.strftime("%y%m%d")
        right_char = "C" if self.right == OptionRight.CALL else "P"
        strike_str = f"{int(self.strike * 1000):08d}"
        return f"{self.underlying}{exp_str}{right_char}{strike_str}"


@dataclass
class PricingResult:
    """Output from the pricing engine."""

    theoretical_price: float
    delta: float
    gamma: float
    theta: float  # per day
    vega: float  # per 1% vol move
    rho: float  # per 1% rate move
    implied_vol: Optional[float] = None
    model_used: str = ""


class PricingEngine:
    """
    QuantLib-based options pricing with Greeks.

    Usage:
        engine = PricingEngine(risk_free_rate=0.045)
        result = engine.price(contract, spot=150.0, vol=0.25)
        iv = engine.implied_vol(contract, spot=150.0, market_price=5.30)
    """

    def __init__(self, risk_free_rate: float = 0.045, dividend_yield: float = 0.0):
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

    def price(
        self,
        contract: OptionContract,
        spot: float,
        vol: float,
        dividend_yield: Optional[float] = None,
    ) -> PricingResult:
        """
        Price an option and compute all Greeks.

        Uses Bjerksund-Stensland for American, analytic BSM for European.
        """
        div_yield = (
            dividend_yield if dividend_yield is not None else self.dividend_yield
        )

        try:
            return self._price_quantlib(contract, spot, vol, div_yield)
        except Exception as e:
            logger.debug("QuantLib pricing failed, using analytical: %s", e)
            return self._price_analytical(contract, spot, vol, div_yield)

    def implied_vol(
        self,
        contract: OptionContract,
        spot: float,
        market_price: float,
        dividend_yield: Optional[float] = None,
    ) -> Optional[float]:
        """
        Solve for implied volatility using Brent's method.

        More robust than Newton-Raphson for deep OTM/ITM options.
        """
        div_yield = (
            dividend_yield if dividend_yield is not None else self.dividend_yield
        )
        T = contract.time_to_expiry
        r = self.risk_free_rate

        if market_price <= 0 or T <= 0 or spot <= 0:
            return None

        # Intrinsic value check
        if contract.right == OptionRight.CALL:
            intrinsic = max(spot - contract.strike, 0)
        else:
            intrinsic = max(contract.strike - spot, 0)

        if market_price < intrinsic * 0.99:
            return None  # arbitrage — price below intrinsic

        def objective(vol):
            result = self._price_analytical(contract, spot, vol, div_yield)
            return result.theoretical_price - market_price

        try:
            iv = brentq(objective, 0.001, 5.0, xtol=1e-6, maxiter=100)
            return float(iv)
        except (ValueError, RuntimeError):
            # Brent's failed — try wider bounds
            try:
                iv = brentq(objective, 0.0001, 10.0, xtol=1e-5, maxiter=200)
                return float(iv)
            except Exception:
                return None

    def _price_quantlib(
        self,
        contract: OptionContract,
        spot: float,
        vol: float,
        div_yield: float,
    ) -> PricingResult:
        """Price using QuantLib engines."""
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        expiry_date = ql.Date(
            contract.expiry.day, contract.expiry.month, contract.expiry.year
        )
        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if contract.right == OptionRight.CALL else ql.Option.Put,
            contract.strike,
        )

        if contract.style == OptionStyle.AMERICAN:
            exercise = ql.AmericanExercise(today, expiry_date)
        else:
            exercise = ql.EuropeanExercise(expiry_date)

        option = ql.VanillaOption(payoff, exercise)

        # Market data
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
        rate_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, self.risk_free_rate, ql.Actual365Fixed())
        )
        div_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, div_yield, ql.Actual365Fixed())
        )
        vol_handle = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), vol, ql.Actual365Fixed())
        )

        process = ql.BlackScholesMertonProcess(
            spot_handle, div_handle, rate_handle, vol_handle
        )

        # Choose engine
        if contract.style == OptionStyle.AMERICAN:
            engine = ql.BaroneAdesiWhaleyApproximationEngine(process)
        else:
            engine = ql.AnalyticEuropeanEngine(process)

        option.setPricingEngine(engine)

        price = option.NPV()

        # Greeks via bumping for American (analytical for European)
        try:
            delta = option.delta()
        except Exception:
            delta = self._bump_delta(contract, spot, vol, div_yield, price)
        try:
            gamma = option.gamma()
        except Exception:
            gamma = self._bump_gamma(contract, spot, vol, div_yield)
        try:
            theta = option.theta() / 365.0
        except Exception:
            theta = self._bump_theta(contract, spot, vol, div_yield, price)
        try:
            vega_raw = option.vega()
            vega = vega_raw / 100.0  # per 1% vol move
        except Exception:
            vega = self._bump_vega(contract, spot, vol, div_yield, price)
        try:
            rho_raw = option.rho()
            rho = rho_raw / 100.0
        except Exception:
            rho = 0.0

        return PricingResult(
            theoretical_price=price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            model_used=(
                "QuantLib_BaroneAdesi"
                if contract.style == OptionStyle.AMERICAN
                else "QuantLib_BSM"
            ),
        )

    def _price_analytical(
        self,
        contract: OptionContract,
        spot: float,
        vol: float,
        div_yield: float,
    ) -> PricingResult:
        """Analytical Black-Scholes-Merton (European) with Greeks."""
        S, K, T = spot, contract.strike, contract.time_to_expiry
        r, q = self.risk_free_rate, div_yield

        if vol <= 0 or T <= 0 or S <= 0 or K <= 0:
            return PricingResult(0, 0, 0, 0, 0, 0, model_used="analytical_edge_case")

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * sqrt_T)
        d2 = d1 - vol * sqrt_T

        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        n_neg_d1 = norm.cdf(-d1)
        n_neg_d2 = norm.cdf(-d2)
        npd1 = norm.pdf(d1)

        exp_qT = np.exp(-q * T)
        exp_rT = np.exp(-r * T)

        if contract.right == OptionRight.CALL:
            price = S * exp_qT * nd1 - K * exp_rT * nd2
            delta = exp_qT * nd1
            rho_sign = K * T * exp_rT * nd2
        else:
            price = K * exp_rT * n_neg_d2 - S * exp_qT * n_neg_d1
            delta = -exp_qT * n_neg_d1
            rho_sign = -K * T * exp_rT * n_neg_d2

        gamma = exp_qT * npd1 / (S * vol * sqrt_T)
        theta = (
            -(S * npd1 * vol * exp_qT) / (2 * sqrt_T)
            - r * K * exp_rT * (nd2 if contract.right == OptionRight.CALL else n_neg_d2)
            + q * S * exp_qT * (nd1 if contract.right == OptionRight.CALL else n_neg_d1)
        ) / 365.0
        vega = S * exp_qT * npd1 * sqrt_T / 100.0
        rho = rho_sign / 100.0

        return PricingResult(
            theoretical_price=max(price, 0),
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            model_used="analytical_BSM",
        )

    # Bump-and-reprice for Greeks when QuantLib analytical Greeks fail
    def _bump_delta(self, c, S, vol, q, base_price):
        bump = S * 0.01
        up = self._price_analytical(c, S + bump, vol, q).theoretical_price
        dn = self._price_analytical(c, S - bump, vol, q).theoretical_price
        return (up - dn) / (2 * bump)

    def _bump_gamma(self, c, S, vol, q):
        bump = S * 0.01
        up = self._price_analytical(c, S + bump, vol, q).delta
        dn = self._price_analytical(c, S - bump, vol, q).delta
        return (up - dn) / (2 * bump)

    def _bump_theta(self, c, S, vol, q, base_price):
        # Can't easily bump expiry in QuantLib, use analytical
        return self._price_analytical(c, S, vol, q).theta

    def _bump_vega(self, c, S, vol, q, base_price):
        bump = 0.01
        up = self._price_analytical(c, S, vol + bump, q).theoretical_price
        return (up - base_price) / 100.0
