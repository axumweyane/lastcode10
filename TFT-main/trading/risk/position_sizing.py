"""
Position sizing strategies: Fixed Fractional, Half-Kelly, Volatility-Scaled.
All sizers floor shares to 0 and cap at max_position_size.
"""

import logging
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SizingStrategy(str, Enum):
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_SCALED = "volatility_scaled"


@dataclass
class SizingInput:
    portfolio_value: float
    current_price: float
    # Risk params
    risk_per_trade_percent: float = 1.0
    stop_loss_distance_percent: float = 2.0
    max_position_size: float = 0.05  # fraction of portfolio
    # TFT confidence params (for Kelly)
    win_probability: Optional[float] = None  # from TFT quantile spread
    avg_win_ratio: Optional[float] = None  # avg win / avg loss
    # Volatility params (for vol-scaled)
    atr: Optional[float] = None
    atr_multiplier: float = 2.0


@dataclass
class SizingResult:
    shares: int
    position_value: float
    position_percent: float
    risk_amount: float
    risk_percent: float
    rationale: str


class BasePositionSizer(ABC):
    @abstractmethod
    def calculate(self, inputs: SizingInput) -> SizingResult:
        """Calculate the number of shares to trade."""


class FixedFractionalSizer(BasePositionSizer):
    """shares = (portfolio * risk_pct) / (price * stop_loss_pct)"""

    def calculate(self, inputs: SizingInput) -> SizingResult:
        risk_amount = inputs.portfolio_value * (inputs.risk_per_trade_percent / 100)
        dollar_risk_per_share = inputs.current_price * (
            inputs.stop_loss_distance_percent / 100
        )

        if dollar_risk_per_share <= 0 or inputs.current_price <= 0:
            return _zero_result("Invalid price or stop loss")

        raw_shares = risk_amount / dollar_risk_per_share
        shares = _apply_caps(raw_shares, inputs)

        position_value = shares * inputs.current_price
        return SizingResult(
            shares=shares,
            position_value=position_value,
            position_percent=(
                (position_value / inputs.portfolio_value * 100)
                if inputs.portfolio_value > 0
                else 0
            ),
            risk_amount=shares * dollar_risk_per_share,
            risk_percent=inputs.risk_per_trade_percent,
            rationale=(
                f"Fixed fractional: {inputs.risk_per_trade_percent}% risk, "
                f"{inputs.stop_loss_distance_percent}% stop -> {shares} shares"
            ),
        )


class KellyCriterionSizer(BasePositionSizer):
    """Half-Kelly: f = 0.5 * (p * b - q) / b, capped at max_position_size."""

    def calculate(self, inputs: SizingInput) -> SizingResult:
        p = inputs.win_probability
        b = inputs.avg_win_ratio

        if p is None or b is None or b <= 0:
            return _zero_result("Missing win_probability or avg_win_ratio for Kelly")

        q = 1.0 - p
        kelly_fraction = (p * b - q) / b
        # Half-Kelly for safety
        half_kelly = kelly_fraction / 2.0

        if half_kelly <= 0:
            return _zero_result(f"Kelly fraction non-positive ({kelly_fraction:.4f})")

        # Cap at max_position_size
        position_fraction = min(half_kelly, inputs.max_position_size)
        position_value = inputs.portfolio_value * position_fraction

        if inputs.current_price <= 0:
            return _zero_result("Invalid current price")

        raw_shares = position_value / inputs.current_price
        shares = _apply_caps(raw_shares, inputs)

        actual_value = shares * inputs.current_price
        risk_amount = inputs.portfolio_value * (inputs.risk_per_trade_percent / 100)
        return SizingResult(
            shares=shares,
            position_value=actual_value,
            position_percent=(
                (actual_value / inputs.portfolio_value * 100)
                if inputs.portfolio_value > 0
                else 0
            ),
            risk_amount=risk_amount,
            risk_percent=inputs.risk_per_trade_percent,
            rationale=(
                f"Half-Kelly: p={p:.2f}, b={b:.2f}, "
                f"full_f={kelly_fraction:.4f}, half_f={half_kelly:.4f}, "
                f"capped={position_fraction:.4f} -> {shares} shares"
            ),
        )


class VolatilityScaledSizer(BasePositionSizer):
    """shares = dollar_risk / (ATR * multiplier)"""

    def calculate(self, inputs: SizingInput) -> SizingResult:
        if inputs.atr is None or inputs.atr <= 0:
            return _zero_result("Missing or invalid ATR for volatility sizing")

        if inputs.current_price <= 0:
            return _zero_result("Invalid current price")

        risk_amount = inputs.portfolio_value * (inputs.risk_per_trade_percent / 100)
        dollar_risk_per_share = inputs.atr * inputs.atr_multiplier

        if dollar_risk_per_share <= 0:
            return _zero_result("ATR * multiplier is zero or negative")

        raw_shares = risk_amount / dollar_risk_per_share
        shares = _apply_caps(raw_shares, inputs)

        actual_value = shares * inputs.current_price
        return SizingResult(
            shares=shares,
            position_value=actual_value,
            position_percent=(
                (actual_value / inputs.portfolio_value * 100)
                if inputs.portfolio_value > 0
                else 0
            ),
            risk_amount=shares * dollar_risk_per_share,
            risk_percent=inputs.risk_per_trade_percent,
            rationale=(
                f"Volatility-scaled: ATR={inputs.atr:.2f}, "
                f"mult={inputs.atr_multiplier}, "
                f"risk/share=${dollar_risk_per_share:.2f} -> {shares} shares"
            ),
        )


class PositionSizerFactory:
    _registry = {
        SizingStrategy.FIXED_FRACTIONAL: FixedFractionalSizer,
        SizingStrategy.KELLY_CRITERION: KellyCriterionSizer,
        SizingStrategy.VOLATILITY_SCALED: VolatilityScaledSizer,
    }

    @classmethod
    def create(cls, strategy: SizingStrategy) -> BasePositionSizer:
        sizer_cls = cls._registry.get(strategy)
        if sizer_cls is None:
            raise ValueError(f"Unknown sizing strategy: {strategy}")
        return sizer_cls()

    @classmethod
    def from_config(cls, strategy_name: str) -> BasePositionSizer:
        try:
            strategy = SizingStrategy(strategy_name)
        except ValueError:
            logger.warning(
                "Unknown strategy '%s', falling back to fixed_fractional", strategy_name
            )
            strategy = SizingStrategy.FIXED_FRACTIONAL
        return cls.create(strategy)


def _apply_caps(raw_shares: float, inputs: SizingInput) -> int:
    """Floor to 0, round down to int, cap at max_position_size."""
    shares = max(0, int(math.floor(raw_shares)))
    if inputs.current_price > 0 and inputs.portfolio_value > 0:
        max_shares = int(
            math.floor(
                inputs.portfolio_value * inputs.max_position_size / inputs.current_price
            )
        )
        shares = min(shares, max_shares)
    return shares


def _zero_result(reason: str) -> SizingResult:
    return SizingResult(
        shares=0,
        position_value=0.0,
        position_percent=0.0,
        risk_amount=0.0,
        risk_percent=0.0,
        rationale=reason,
    )
