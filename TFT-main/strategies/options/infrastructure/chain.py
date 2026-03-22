"""
Options chain fetcher with yfinance primary and Alpaca fallback.

Fetches available contracts for a given underlying, filters by DTE
and liquidity, and returns standardized OptionContract objects.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.options.infrastructure.pricing import (
    OptionContract,
    OptionRight,
    OptionStyle,
)

logger = logging.getLogger(__name__)


@dataclass
class ChainEntry:
    """Single option in the chain with market data."""

    contract: OptionContract
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_vol: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread_pct(self) -> float:
        if self.mid > 0:
            return (self.ask - self.bid) / self.mid * 100
        return 999.0


@dataclass
class OptionsChain:
    """Full options chain for an underlying on a specific expiry."""

    underlying: str
    expiry: date
    spot_price: float
    calls: List[ChainEntry] = field(default_factory=list)
    puts: List[ChainEntry] = field(default_factory=list)
    fetch_time: Optional[datetime] = None

    @property
    def dte(self) -> int:
        return max((self.expiry - date.today()).days, 0)

    def get_call_by_delta(
        self, target_delta: float, tolerance: float = 0.05
    ) -> Optional[ChainEntry]:
        """Find call closest to target delta."""
        return self._find_by_delta(self.calls, abs(target_delta), tolerance)

    def get_put_by_delta(
        self, target_delta: float, tolerance: float = 0.05
    ) -> Optional[ChainEntry]:
        """Find put closest to target delta (use positive value, we'll negate)."""
        return self._find_by_delta(self.puts, abs(target_delta), tolerance)

    def get_atm_strike(self) -> float:
        """Get the at-the-money strike (closest to spot)."""
        all_strikes = [e.contract.strike for e in self.calls + self.puts]
        if not all_strikes:
            return self.spot_price
        return min(all_strikes, key=lambda k: abs(k - self.spot_price))

    def get_strike_by_std(
        self, std_devs: float, vol: float, right: str = "call"
    ) -> Optional[float]:
        """Get strike N standard deviations from spot."""
        dte_years = max(self.dte / 365.0, 1e-6)
        move = self.spot_price * vol * np.sqrt(dte_years) * std_devs
        target = self.spot_price + move if right == "call" else self.spot_price - move
        entries = self.calls if right == "call" else self.puts
        if not entries:
            return None
        closest = min(entries, key=lambda e: abs(e.contract.strike - target))
        return closest.contract.strike

    @staticmethod
    def _find_by_delta(
        entries: List[ChainEntry], target: float, tolerance: float
    ) -> Optional[ChainEntry]:
        if not entries:
            return None
        best = min(entries, key=lambda e: abs(abs(e.delta) - target))
        if abs(abs(best.delta) - target) <= tolerance:
            return best
        return best  # return closest even if outside tolerance


class ChainFetcher:
    """
    Fetches options chains from yfinance (primary) or Alpaca (fallback).

    Usage:
        fetcher = ChainFetcher()
        chains = fetcher.fetch("AAPL", min_dte=20, max_dte=50)
    """

    def __init__(self, data_source: str = "yfinance"):
        self.data_source = data_source

    def fetch(
        self,
        underlying: str,
        min_dte: int = 7,
        max_dte: int = 60,
        min_volume: int = 0,
        max_spread_pct: float = 999.0,
    ) -> List[OptionsChain]:
        """
        Fetch options chains for an underlying across multiple expiries.

        Returns list of OptionsChain (one per expiry within DTE range).
        """
        if self.data_source == "yfinance":
            return self._fetch_yfinance(
                underlying, min_dte, max_dte, min_volume, max_spread_pct
            )
        else:
            logger.warning(
                "Alpaca options data not yet implemented, falling back to yfinance"
            )
            return self._fetch_yfinance(
                underlying, min_dte, max_dte, min_volume, max_spread_pct
            )

    def _fetch_yfinance(
        self,
        underlying: str,
        min_dte: int,
        max_dte: int,
        min_volume: int,
        max_spread_pct: float,
    ) -> List[OptionsChain]:
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed")
            return []

        try:
            ticker = yf.Ticker(underlying)
            spot = self._get_spot(ticker)
            if spot is None or spot <= 0:
                logger.warning("Cannot get spot price for %s", underlying)
                return []

            expiry_strings = ticker.options
            if not expiry_strings:
                logger.warning("No options expiries found for %s", underlying)
                return []

            today = date.today()
            chains = []

            for exp_str in expiry_strings:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days

                if dte < min_dte or dte > max_dte:
                    continue

                try:
                    chain_data = ticker.option_chain(exp_str)
                except Exception as e:
                    logger.debug(
                        "Failed to fetch chain for %s %s: %s", underlying, exp_str, e
                    )
                    continue

                calls = self._parse_yf_chain(
                    chain_data.calls,
                    underlying,
                    exp_date,
                    OptionRight.CALL,
                    spot,
                    min_volume,
                    max_spread_pct,
                )
                puts = self._parse_yf_chain(
                    chain_data.puts,
                    underlying,
                    exp_date,
                    OptionRight.PUT,
                    spot,
                    min_volume,
                    max_spread_pct,
                )

                if calls or puts:
                    chains.append(
                        OptionsChain(
                            underlying=underlying,
                            expiry=exp_date,
                            spot_price=spot,
                            calls=calls,
                            puts=puts,
                            fetch_time=datetime.now(),
                        )
                    )

            logger.info(
                "Fetched %d expiries for %s (spot=$%.2f)",
                len(chains),
                underlying,
                spot,
            )
            return chains

        except Exception as e:
            logger.error("Failed to fetch options for %s: %s", underlying, e)
            return []

    def _parse_yf_chain(
        self,
        df: pd.DataFrame,
        underlying: str,
        expiry: date,
        right: OptionRight,
        spot: float,
        min_volume: int,
        max_spread_pct: float,
    ) -> List[ChainEntry]:
        entries = []
        if df is None or df.empty:
            return entries

        for _, row in df.iterrows():
            strike = float(row.get("strike", 0))
            if strike <= 0:
                continue

            bid = float(row.get("bid", 0))
            ask = float(row.get("ask", 0))
            last = float(row.get("lastPrice", 0))
            volume = int(row.get("volume", 0) or 0)
            oi = int(row.get("openInterest", 0) or 0)
            iv = float(row.get("impliedVolatility", 0) or 0)

            # Filters
            if volume < min_volume:
                continue
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
            if mid > 0 and max_spread_pct < 999:
                spread = (ask - bid) / mid * 100
                if spread > max_spread_pct:
                    continue

            contract = OptionContract(
                underlying=underlying,
                strike=strike,
                expiry=expiry,
                right=right,
            )

            # Approximate delta from IV if available
            delta = self._approx_delta(spot, strike, iv, expiry, right)

            entries.append(
                ChainEntry(
                    contract=contract,
                    bid=bid,
                    ask=ask,
                    last=last,
                    volume=volume,
                    open_interest=oi,
                    implied_vol=iv,
                    delta=delta,
                )
            )

        return entries

    @staticmethod
    def _approx_delta(spot, strike, iv, expiry, right) -> float:
        """Quick analytical delta estimate."""
        if iv <= 0 or spot <= 0 or strike <= 0:
            return 0.5 if right == OptionRight.CALL else -0.5

        T = max((expiry - date.today()).days / 365.0, 1e-6)
        d1 = (np.log(spot / strike) + 0.5 * iv**2 * T) / (iv * np.sqrt(T))
        if right == OptionRight.CALL:
            return float(norm.cdf(d1))
        else:
            return float(norm.cdf(d1) - 1)

    @staticmethod
    def _get_spot(ticker) -> Optional[float]:
        try:
            info = ticker.fast_info
            return float(info.get("lastPrice", 0) or info.get("previousClose", 0))
        except Exception:
            try:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    return float(hist["Close"].iloc[-1])
            except Exception:
                pass
        return None


from scipy.stats import norm
