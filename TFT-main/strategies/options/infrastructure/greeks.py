"""
Portfolio-level Greeks aggregator.

Computes net Greeks across all options positions and provides
risk metrics for the options risk manager.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from strategies.options.infrastructure.pricing import (
    OptionContract,
    PricingEngine,
    PricingResult,
)

logger = logging.getLogger(__name__)


@dataclass
class PositionGreeks:
    """Greeks for a single options position."""

    contract: OptionContract
    quantity: int  # positive = long, negative = short
    delta: float
    gamma: float
    theta: float  # daily
    vega: float  # per 1% vol move
    rho: float
    notional: float  # |quantity| * multiplier * spot
    market_value: float  # quantity * mid_price * multiplier
    max_loss: float  # worst-case loss for this position

    @property
    def dollar_delta(self) -> float:
        return self.delta * self.quantity * self.contract.multiplier

    @property
    def dollar_gamma(self) -> float:
        return self.gamma * self.quantity * self.contract.multiplier

    @property
    def dollar_theta(self) -> float:
        return self.theta * self.quantity * self.contract.multiplier

    @property
    def dollar_vega(self) -> float:
        return self.vega * self.quantity * self.contract.multiplier


@dataclass
class PortfolioGreeks:
    """Aggregated Greeks across all options positions."""

    net_delta: float = 0.0  # net directional exposure in shares
    net_gamma: float = 0.0  # rate of delta change
    net_theta: float = 0.0  # daily time decay ($)
    net_vega: float = 0.0  # volatility exposure ($)
    net_rho: float = 0.0
    gross_delta: float = 0.0  # total |delta|
    total_notional: float = 0.0
    total_market_value: float = 0.0
    total_max_loss: float = 0.0
    position_count: int = 0
    positions: List[PositionGreeks] = field(default_factory=list)

    @property
    def beta_weighted_delta(self) -> float:
        """Approximate SPY-equivalent delta exposure."""
        return self.net_delta  # caller can multiply by beta

    def summary(self) -> Dict:
        return {
            "net_delta": round(self.net_delta, 2),
            "net_gamma": round(self.net_gamma, 4),
            "net_theta": round(self.net_theta, 2),
            "net_vega": round(self.net_vega, 2),
            "gross_delta": round(self.gross_delta, 2),
            "total_max_loss": round(self.total_max_loss, 2),
            "position_count": self.position_count,
        }


class GreeksCalculator:
    """
    Calculates per-position and portfolio-level Greeks.

    Usage:
        calc = GreeksCalculator(pricing_engine)
        pos = calc.calculate_position(contract, quantity=1, spot=150.0, vol=0.25)
        portfolio = calc.aggregate([pos1, pos2, pos3])
    """

    def __init__(self, pricing_engine: Optional[PricingEngine] = None):
        self.engine = pricing_engine or PricingEngine()

    def calculate_position(
        self,
        contract: OptionContract,
        quantity: int,
        spot: float,
        vol: float,
        mid_price: Optional[float] = None,
    ) -> PositionGreeks:
        """Calculate Greeks for a single position."""
        result = self.engine.price(contract, spot, vol)

        if mid_price is None:
            mid_price = result.theoretical_price

        market_value = quantity * mid_price * contract.multiplier
        notional = abs(quantity) * contract.multiplier * spot

        # Max loss calculation
        if quantity > 0:
            # Long option: max loss = premium paid
            max_loss = abs(market_value)
        else:
            # Short option: depends on type
            if contract.right.value == "call":
                # Short naked call: theoretically unlimited, cap at 3x spot
                max_loss = abs(quantity) * contract.multiplier * spot * 3
            else:
                # Short naked put: max loss = strike * contracts * multiplier
                max_loss = abs(quantity) * contract.multiplier * contract.strike

        return PositionGreeks(
            contract=contract,
            quantity=quantity,
            delta=result.delta,
            gamma=result.gamma,
            theta=result.theta,
            vega=result.vega,
            rho=result.rho,
            notional=notional,
            market_value=market_value,
            max_loss=max_loss,
        )

    def aggregate(self, positions: List[PositionGreeks]) -> PortfolioGreeks:
        """Aggregate Greeks across all positions into portfolio-level view."""
        pf = PortfolioGreeks(positions=positions, position_count=len(positions))

        for pos in positions:
            pf.net_delta += pos.dollar_delta
            pf.net_gamma += pos.dollar_gamma
            pf.net_theta += pos.dollar_theta
            pf.net_vega += pos.dollar_vega
            pf.net_rho += pos.rho * pos.quantity * pos.contract.multiplier
            pf.gross_delta += abs(pos.dollar_delta)
            pf.total_notional += pos.notional
            pf.total_market_value += pos.market_value
            pf.total_max_loss += pos.max_loss

        return pf

    def stress_test(
        self,
        positions: List[PositionGreeks],
        spot_shock_pct: float = 0.05,
        vol_shock_pct: float = 0.10,
    ) -> Dict:
        """
        Quick stress test: reprice portfolio with shocked spot and vol.

        Returns P&L impact of each scenario.
        """
        base_value = sum(p.market_value for p in positions)

        # Spot up
        spot_up_pnl = sum(
            p.dollar_delta * spot_shock_pct * 100
            + 0.5 * p.dollar_gamma * (spot_shock_pct * 100) ** 2
            for p in positions
        )

        # Spot down
        spot_dn_pnl = sum(
            -p.dollar_delta * spot_shock_pct * 100
            + 0.5 * p.dollar_gamma * (spot_shock_pct * 100) ** 2
            for p in positions
        )

        # Vol up
        vol_up_pnl = sum(p.dollar_vega * vol_shock_pct * 100 for p in positions)

        # Vol down
        vol_dn_pnl = sum(-p.dollar_vega * vol_shock_pct * 100 for p in positions)

        # 1-day theta decay
        theta_1d = sum(p.dollar_theta for p in positions)

        return {
            "base_value": round(base_value, 2),
            f"spot_+{spot_shock_pct:.0%}": round(spot_up_pnl, 2),
            f"spot_-{spot_shock_pct:.0%}": round(spot_dn_pnl, 2),
            f"vol_+{vol_shock_pct:.0%}": round(vol_up_pnl, 2),
            f"vol_-{vol_shock_pct:.0%}": round(vol_dn_pnl, 2),
            "theta_1d": round(theta_1d, 2),
        }
