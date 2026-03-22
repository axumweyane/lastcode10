"""
Options strategy configuration — all toggleable via environment variables.
Every strategy is DISABLED by default.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_list(key: str, default: str = "") -> List[str]:
    raw = os.getenv(key, default)
    return [s.strip() for s in raw.split(",") if s.strip()] if raw else []


@dataclass
class CoveredCallConfig:
    enabled: bool = False
    target_delta: float = 0.25  # sell calls at ~25 delta
    min_delta: float = 0.15
    max_delta: float = 0.35
    min_dte: int = 25  # minimum days to expiry
    max_dte: int = 45
    roll_dte: int = 7  # auto-roll when DTE <= 7
    profit_target_pct: float = 50.0  # close at 50% of max profit
    min_iv_rank: float = 20.0  # don't sell when IV is cheap
    max_position_pct: float = 0.10
    kill_max_drawdown: float = 0.15
    kill_min_sharpe: float = -1.0

    @classmethod
    def from_env(cls) -> "CoveredCallConfig":
        return cls(
            enabled=_env_bool("OPT_COVERED_CALL_ENABLED"),
            target_delta=_env_float("OPT_CC_TARGET_DELTA", 0.25),
            min_dte=_env_int("OPT_CC_MIN_DTE", 25),
            max_dte=_env_int("OPT_CC_MAX_DTE", 45),
            roll_dte=_env_int("OPT_CC_ROLL_DTE", 7),
            profit_target_pct=_env_float("OPT_CC_PROFIT_TARGET", 50.0),
            min_iv_rank=_env_float("OPT_CC_MIN_IV_RANK", 20.0),
        )


@dataclass
class IronCondorConfig:
    enabled: bool = False
    underlyings: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])
    wing_width_std: float = 1.0  # wings at 1 standard deviation
    min_dte: int = 30
    max_dte: int = 50
    min_iv_rank: float = 50.0  # only sell when IV is elevated
    profit_target_pct: float = 50.0
    stop_loss_pct: float = 200.0  # close at 2x credit received
    max_position_pct: float = 0.05
    kill_max_drawdown: float = 0.20
    kill_min_sharpe: float = -1.0

    @classmethod
    def from_env(cls) -> "IronCondorConfig":
        return cls(
            enabled=_env_bool("OPT_IRON_CONDOR_ENABLED"),
            underlyings=_env_list("OPT_IC_UNDERLYINGS", "SPY,QQQ"),
            wing_width_std=_env_float("OPT_IC_WING_STD", 1.0),
            min_dte=_env_int("OPT_IC_MIN_DTE", 30),
            max_dte=_env_int("OPT_IC_MAX_DTE", 50),
            min_iv_rank=_env_float("OPT_IC_MIN_IV_RANK", 50.0),
            profit_target_pct=_env_float("OPT_IC_PROFIT_TARGET", 50.0),
            stop_loss_pct=_env_float("OPT_IC_STOP_LOSS", 200.0),
        )


@dataclass
class ProtectivePutConfig:
    enabled: bool = False
    target_delta: float = -0.20  # buy puts at ~20 delta
    min_dte: int = 30
    max_dte: int = 60
    hedge_ratio: float = 0.50  # hedge 50% of portfolio
    only_volatile_regime: bool = True
    max_premium_pct: float = 2.0  # max 2% of portfolio on puts
    kill_max_drawdown: float = 0.25
    kill_min_sharpe: float = -2.0  # higher tolerance (insurance)

    @classmethod
    def from_env(cls) -> "ProtectivePutConfig":
        return cls(
            enabled=_env_bool("OPT_PROTECTIVE_PUT_ENABLED"),
            target_delta=_env_float("OPT_PP_TARGET_DELTA", -0.20),
            min_dte=_env_int("OPT_PP_MIN_DTE", 30),
            max_dte=_env_int("OPT_PP_MAX_DTE", 60),
            hedge_ratio=_env_float("OPT_PP_HEDGE_RATIO", 0.50),
            only_volatile_regime=_env_bool("OPT_PP_VOLATILE_ONLY", True),
            max_premium_pct=_env_float("OPT_PP_MAX_PREMIUM_PCT", 2.0),
        )


@dataclass
class VolArbConfig:
    enabled: bool = False
    iv_rv_entry_threshold: float = 0.05  # enter when |IV - RV| > 5 vol points
    iv_rv_exit_threshold: float = 0.02
    garch_lookback_days: int = 252
    min_dte: int = 20
    max_dte: int = 45
    target_delta: float = 0.25
    use_tft_timing: bool = True  # use TFT forecast to time entries
    max_position_pct: float = 0.05
    kill_max_drawdown: float = 0.20
    kill_min_sharpe: float = -1.0

    @classmethod
    def from_env(cls) -> "VolArbConfig":
        return cls(
            enabled=_env_bool("OPT_VOL_ARB_ENABLED"),
            iv_rv_entry_threshold=_env_float("OPT_VA_ENTRY_THRESH", 0.05),
            iv_rv_exit_threshold=_env_float("OPT_VA_EXIT_THRESH", 0.02),
            garch_lookback_days=_env_int("OPT_VA_GARCH_LOOKBACK", 252),
            use_tft_timing=_env_bool("OPT_VA_USE_TFT", True),
        )


@dataclass
class EarningsPlayConfig:
    enabled: bool = False
    min_confidence: float = 0.60  # TFT + sentiment confidence threshold
    directional_spread_width: int = 5  # $5 wide spreads
    neutral_wing_std: float = 1.0
    max_risk_per_play: float = 0.02  # 2% max risk per earnings play
    entry_days_before: int = 3  # enter 3 days before earnings
    exit_days_after: int = 1  # exit 1 day after earnings
    min_iv_rank: float = 40.0
    kill_max_drawdown: float = 0.25
    kill_min_sharpe: float = -1.5

    @classmethod
    def from_env(cls) -> "EarningsPlayConfig":
        return cls(
            enabled=_env_bool("OPT_EARNINGS_ENABLED"),
            min_confidence=_env_float("OPT_EARN_MIN_CONF", 0.60),
            directional_spread_width=_env_int("OPT_EARN_SPREAD_WIDTH", 5),
            max_risk_per_play=_env_float("OPT_EARN_MAX_RISK", 0.02),
            entry_days_before=_env_int("OPT_EARN_ENTRY_DAYS", 3),
            exit_days_after=_env_int("OPT_EARN_EXIT_DAYS", 1),
        )


@dataclass
class GammaScalpConfig:
    enabled: bool = False
    rv_iv_threshold: float = 0.03  # enter when RV - IV > 3 vol points
    hedge_frequency_hours: int = 4  # delta hedge every 4 hours
    min_dte: int = 14
    max_dte: int = 30
    max_position_pct: float = 0.05
    profit_target_pct: float = 100.0  # close at 100% of premium paid
    stop_loss_pct: float = 80.0  # close if 80% of premium lost
    kill_max_drawdown: float = 0.20
    kill_min_sharpe: float = -1.0

    @classmethod
    def from_env(cls) -> "GammaScalpConfig":
        return cls(
            enabled=_env_bool("OPT_GAMMA_SCALP_ENABLED"),
            rv_iv_threshold=_env_float("OPT_GS_RV_IV_THRESH", 0.03),
            hedge_frequency_hours=_env_int("OPT_GS_HEDGE_FREQ_HR", 4),
            min_dte=_env_int("OPT_GS_MIN_DTE", 14),
            max_dte=_env_int("OPT_GS_MAX_DTE", 30),
        )


@dataclass
class OptionsInfraConfig:
    """Infrastructure-level config for the options system."""

    risk_free_rate: float = 0.045
    data_source: str = "yfinance"  # "alpaca" or "yfinance"
    min_option_volume: int = 10
    max_bid_ask_spread_pct: float = 10.0  # max 10% of mid
    iv_rank_lookback_days: int = 252
    vol_surface_strikes: int = 20  # strikes per expiry for surface
    vol_surface_expiries: int = 6  # expiry dates for surface

    @classmethod
    def from_env(cls) -> "OptionsInfraConfig":
        return cls(
            risk_free_rate=_env_float("OPT_RISK_FREE_RATE", 0.045),
            data_source=_env_str("OPT_DATA_SOURCE", "yfinance"),
            min_option_volume=_env_int("OPT_MIN_VOLUME", 10),
            max_bid_ask_spread_pct=_env_float("OPT_MAX_SPREAD_PCT", 10.0),
            iv_rank_lookback_days=_env_int("OPT_IV_RANK_LOOKBACK", 252),
        )


@dataclass
class OptionsMasterConfig:
    infra: OptionsInfraConfig = field(default_factory=OptionsInfraConfig)
    covered_calls: CoveredCallConfig = field(default_factory=CoveredCallConfig)
    iron_condors: IronCondorConfig = field(default_factory=IronCondorConfig)
    protective_puts: ProtectivePutConfig = field(default_factory=ProtectivePutConfig)
    vol_arb: VolArbConfig = field(default_factory=VolArbConfig)
    earnings: EarningsPlayConfig = field(default_factory=EarningsPlayConfig)
    gamma_scalp: GammaScalpConfig = field(default_factory=GammaScalpConfig)

    @classmethod
    def from_env(cls) -> "OptionsMasterConfig":
        cfg = cls(
            infra=OptionsInfraConfig.from_env(),
            covered_calls=CoveredCallConfig.from_env(),
            iron_condors=IronCondorConfig.from_env(),
            protective_puts=ProtectivePutConfig.from_env(),
            vol_arb=VolArbConfig.from_env(),
            earnings=EarningsPlayConfig.from_env(),
            gamma_scalp=GammaScalpConfig.from_env(),
        )
        enabled = []
        for name in [
            "covered_calls",
            "iron_condors",
            "protective_puts",
            "vol_arb",
            "earnings",
            "gamma_scalp",
        ]:
            if getattr(getattr(cfg, name), "enabled", False):
                enabled.append(name)
        if enabled:
            logger.info("Enabled options strategies: %s", ", ".join(enabled))
        else:
            logger.info("No options strategies enabled")
        return cfg
