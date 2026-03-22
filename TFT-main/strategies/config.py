"""
Strategy configuration — all toggleable via environment variables.
Every strategy is DISABLED by default. Enable explicitly in .env.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

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
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Momentum / Mean-Reversion Factor Strategy
# ---------------------------------------------------------------------------


@dataclass
class MomentumConfig:
    enabled: bool = False

    # Momentum parameters
    momentum_lookback_days: int = 252  # 12-month lookback
    momentum_skip_days: int = 21  # skip last month (avoid reversal)
    momentum_weight: float = 0.5  # weight in combo score

    # Mean-reversion parameters
    mean_reversion_lookback_days: int = 5  # 5-day reversal
    mean_reversion_weight: float = 0.3  # weight in combo score

    # Quality factor (profitability)
    quality_weight: float = 0.2

    # Signal thresholds
    long_threshold_zscore: float = 1.0  # go long above this z-score
    short_threshold_zscore: float = -1.0  # go short below this z-score
    max_positions_per_side: int = 10

    # Minimum data requirements
    min_history_days: int = 280  # need 252+21 days minimum
    min_avg_dollar_volume: float = 1_000_000  # $1M avg daily dollar volume

    # Risk
    strategy_max_drawdown: float = 0.20
    strategy_kill_sharpe: float = -1.0

    @classmethod
    def from_env(cls) -> "MomentumConfig":
        return cls(
            enabled=_env_bool("STRATEGY_MOMENTUM_ENABLED", False),
            momentum_lookback_days=_env_int("STRATEGY_MOMENTUM_LOOKBACK", 252),
            momentum_skip_days=_env_int("STRATEGY_MOMENTUM_SKIP", 21),
            momentum_weight=_env_float("STRATEGY_MOMENTUM_WEIGHT", 0.5),
            mean_reversion_lookback_days=_env_int("STRATEGY_MEANREV_LOOKBACK", 5),
            mean_reversion_weight=_env_float("STRATEGY_MEANREV_WEIGHT", 0.3),
            quality_weight=_env_float("STRATEGY_QUALITY_WEIGHT", 0.2),
            long_threshold_zscore=_env_float("STRATEGY_MOMENTUM_LONG_ZSCORE", 1.0),
            short_threshold_zscore=_env_float("STRATEGY_MOMENTUM_SHORT_ZSCORE", -1.0),
            max_positions_per_side=_env_int("STRATEGY_MOMENTUM_MAX_POS", 10),
            min_history_days=_env_int("STRATEGY_MOMENTUM_MIN_HISTORY", 280),
            min_avg_dollar_volume=_env_float("STRATEGY_MOMENTUM_MIN_ADV", 1_000_000),
            strategy_max_drawdown=_env_float("STRATEGY_MOMENTUM_MAX_DD", 0.20),
            strategy_kill_sharpe=_env_float("STRATEGY_MOMENTUM_KILL_SHARPE", -1.0),
        )


# ---------------------------------------------------------------------------
# Statistical Arbitrage (Pairs Trading)
# ---------------------------------------------------------------------------


@dataclass
class StatArbConfig:
    enabled: bool = False

    # Pair selection
    cointegration_pvalue: float = 0.05  # Engle-Granger threshold
    max_half_life_days: int = 30  # reject slow mean-reverting pairs
    min_half_life_days: int = 2  # reject too-fast pairs (noise)
    max_pairs: int = 20  # max simultaneous pairs
    rescan_interval_days: int = 7  # re-run cointegration weekly

    # Trading thresholds
    entry_zscore: float = 2.0  # enter when spread |z| > 2.0
    exit_zscore: float = 0.5  # exit when spread |z| < 0.5
    stop_loss_zscore: float = 4.0  # stop-loss at |z| > 4.0
    lookback_window: int = 63  # spread z-score lookback

    # Sector constraints
    same_sector_only: bool = True
    sector_pairs_limit: int = 3  # max pairs from one sector

    # Risk
    strategy_max_drawdown: float = 0.20
    strategy_kill_sharpe: float = -1.0
    max_position_per_pair: float = 0.05  # 5% of portfolio per pair

    @classmethod
    def from_env(cls) -> "StatArbConfig":
        return cls(
            enabled=_env_bool("STRATEGY_STATARB_ENABLED", False),
            cointegration_pvalue=_env_float("STRATEGY_STATARB_COINT_PVAL", 0.05),
            max_half_life_days=_env_int("STRATEGY_STATARB_MAX_HL", 30),
            min_half_life_days=_env_int("STRATEGY_STATARB_MIN_HL", 2),
            max_pairs=_env_int("STRATEGY_STATARB_MAX_PAIRS", 20),
            rescan_interval_days=_env_int("STRATEGY_STATARB_RESCAN_DAYS", 7),
            entry_zscore=_env_float("STRATEGY_STATARB_ENTRY_Z", 2.0),
            exit_zscore=_env_float("STRATEGY_STATARB_EXIT_Z", 0.5),
            stop_loss_zscore=_env_float("STRATEGY_STATARB_STOP_Z", 4.0),
            lookback_window=_env_int("STRATEGY_STATARB_LOOKBACK", 63),
            same_sector_only=_env_bool("STRATEGY_STATARB_SAME_SECTOR", True),
            sector_pairs_limit=_env_int("STRATEGY_STATARB_SECTOR_LIMIT", 3),
            strategy_max_drawdown=_env_float("STRATEGY_STATARB_MAX_DD", 0.20),
            strategy_kill_sharpe=_env_float("STRATEGY_STATARB_KILL_SHARPE", -1.0),
            max_position_per_pair=_env_float("STRATEGY_STATARB_MAX_POS_PAIR", 0.05),
        )


# ---------------------------------------------------------------------------
# Ensemble / Signal Combiner
# ---------------------------------------------------------------------------


@dataclass
class EnsembleConfig:
    enabled: bool = False

    # Weighting method
    weighting_method: str = "bayesian"  # "equal", "sharpe", "bayesian"
    sharpe_lookback_days: int = 63  # rolling window for weight calc
    min_weight: float = 0.05  # floor weight (no strategy drops to 0)
    max_weight: float = 0.50  # cap weight (no strategy dominates)

    # Portfolio constraints
    max_total_positions: int = 30
    max_gross_leverage: float = 2.0  # long + short exposure
    max_net_leverage: float = 0.5  # |long - short| exposure
    target_volatility: float = 0.15  # 15% annualized target vol

    # Risk
    portfolio_max_drawdown: float = 0.20
    var_confidence: float = 0.99  # 99% VaR
    correlation_alert_threshold: float = 0.6  # alert if strategies correlate

    # Bayesian weight updater (Beta-Binomial adaptive weights)
    use_bayesian_updater: bool = False  # ENSEMBLE_USE_BAYESIAN_WEIGHTS=true to enable
    bayesian_decay: float = 0.995  # exponential forgetting factor

    @classmethod
    def from_env(cls) -> "EnsembleConfig":
        return cls(
            enabled=_env_bool("STRATEGY_ENSEMBLE_ENABLED", False),
            weighting_method=_env_str("STRATEGY_ENSEMBLE_WEIGHTING", "bayesian"),
            sharpe_lookback_days=_env_int("STRATEGY_ENSEMBLE_SHARPE_LOOKBACK", 63),
            min_weight=_env_float("STRATEGY_ENSEMBLE_MIN_WEIGHT", 0.05),
            max_weight=_env_float("STRATEGY_ENSEMBLE_MAX_WEIGHT", 0.50),
            max_total_positions=_env_int("STRATEGY_ENSEMBLE_MAX_POSITIONS", 30),
            max_gross_leverage=_env_float("STRATEGY_ENSEMBLE_MAX_GROSS_LEV", 2.0),
            max_net_leverage=_env_float("STRATEGY_ENSEMBLE_MAX_NET_LEV", 0.5),
            target_volatility=_env_float("STRATEGY_ENSEMBLE_TARGET_VOL", 0.15),
            portfolio_max_drawdown=_env_float("STRATEGY_ENSEMBLE_MAX_DD", 0.20),
            var_confidence=_env_float("STRATEGY_ENSEMBLE_VAR_CONF", 0.99),
            correlation_alert_threshold=_env_float("STRATEGY_ENSEMBLE_CORR_ALERT", 0.6),
            use_bayesian_updater=_env_bool("ENSEMBLE_USE_BAYESIAN_WEIGHTS", False),
            bayesian_decay=_env_float("ENSEMBLE_BAYESIAN_DECAY", 0.995),
        )


# ---------------------------------------------------------------------------
# Regime Detection
# ---------------------------------------------------------------------------


@dataclass
class RegimeConfig:
    enabled: bool = False

    # VIX thresholds (aligned with existing config_manager.py values)
    vix_low_threshold: float = 20.0
    vix_high_threshold: float = 30.0

    # Market breadth
    breadth_lookback_days: int = 50  # % stocks above 50-day MA
    breadth_trending_threshold: float = 0.6
    breadth_choppy_threshold: float = 0.4

    # Realized vol
    realized_vol_window: int = 21

    # Regime allocation weights: [momentum, mean_rev, pairs, tft]
    # Calm Trending: momentum does well
    calm_trending_weights: List[float] = field(
        default_factory=lambda: [0.40, 0.15, 0.20, 0.25]
    )
    # Calm Choppy: mean reversion does well
    calm_choppy_weights: List[float] = field(
        default_factory=lambda: [0.15, 0.40, 0.25, 0.20]
    )
    # Volatile Trending: reduce exposure, favor trend
    volatile_trending_weights: List[float] = field(
        default_factory=lambda: [0.30, 0.10, 0.30, 0.30]
    )
    # Volatile Choppy: defensive, favor pairs (market-neutral)
    volatile_choppy_weights: List[float] = field(
        default_factory=lambda: [0.10, 0.20, 0.45, 0.25]
    )

    @classmethod
    def from_env(cls) -> "RegimeConfig":
        return cls(
            enabled=_env_bool("STRATEGY_REGIME_ENABLED", False),
            vix_low_threshold=_env_float("STRATEGY_REGIME_VIX_LOW", 20.0),
            vix_high_threshold=_env_float("STRATEGY_REGIME_VIX_HIGH", 30.0),
            breadth_lookback_days=_env_int("STRATEGY_REGIME_BREADTH_LOOKBACK", 50),
            breadth_trending_threshold=_env_float("STRATEGY_REGIME_BREADTH_TREND", 0.6),
            breadth_choppy_threshold=_env_float("STRATEGY_REGIME_BREADTH_CHOP", 0.4),
            realized_vol_window=_env_int("STRATEGY_REGIME_RVOL_WINDOW", 21),
        )


# ---------------------------------------------------------------------------
# Mean Reversion (OU-based)
# ---------------------------------------------------------------------------


@dataclass
class MeanReversionConfig:
    enabled: bool = False

    hurst_threshold: float = 0.45  # only trade when Hurst < this
    min_half_life: int = 2  # reject too-fast reversion
    max_half_life: int = 30  # reject too-slow reversion
    entry_zscore: float = 1.5  # enter when |deviation_z| > this
    exit_zscore: float = 0.5  # exit when |deviation_z| < this
    max_positions_per_side: int = 8

    # Risk
    strategy_max_drawdown: float = 0.20
    strategy_kill_sharpe: float = -1.0

    @classmethod
    def from_env(cls) -> "MeanReversionConfig":
        return cls(
            enabled=_env_bool("STRATEGY_MEAN_REVERSION_ENABLED", False),
            hurst_threshold=_env_float("STRATEGY_MR_HURST_THRESH", 0.45),
            min_half_life=_env_int("STRATEGY_MR_MIN_HL", 2),
            max_half_life=_env_int("STRATEGY_MR_MAX_HL", 30),
            entry_zscore=_env_float("STRATEGY_MR_ENTRY_Z", 1.5),
            exit_zscore=_env_float("STRATEGY_MR_EXIT_Z", 0.5),
            max_positions_per_side=_env_int("STRATEGY_MR_MAX_POS", 8),
            strategy_max_drawdown=_env_float("STRATEGY_MR_MAX_DD", 0.20),
            strategy_kill_sharpe=_env_float("STRATEGY_MR_KILL_SHARPE", -1.0),
        )


# ---------------------------------------------------------------------------
# Sector Rotation (macro regime driven)
# ---------------------------------------------------------------------------


@dataclass
class SectorRotationConfig:
    enabled: bool = False

    min_tilt_threshold: float = 0.1  # ignore sector tilts below this
    max_positions_per_side: int = 8
    rebalance_interval_days: int = 21  # re-evaluate monthly

    # Risk
    strategy_max_drawdown: float = 0.20
    strategy_kill_sharpe: float = -1.0

    @classmethod
    def from_env(cls) -> "SectorRotationConfig":
        return cls(
            enabled=_env_bool("STRATEGY_SECTOR_ROTATION_ENABLED", False),
            min_tilt_threshold=_env_float("STRATEGY_SR_MIN_TILT", 0.1),
            max_positions_per_side=_env_int("STRATEGY_SR_MAX_POS", 8),
            rebalance_interval_days=_env_int("STRATEGY_SR_REBAL_DAYS", 21),
            strategy_max_drawdown=_env_float("STRATEGY_SR_MAX_DD", 0.20),
            strategy_kill_sharpe=_env_float("STRATEGY_SR_KILL_SHARPE", -1.0),
        )


# ---------------------------------------------------------------------------
# FX Momentum (time-series momentum on currency pairs)
# ---------------------------------------------------------------------------


@dataclass
class FXMomentumConfig:
    enabled: bool = False

    min_lookback_days: int = 63  # need at least 3 months of data
    signal_threshold: float = 0.5  # minimum z-score to generate signal
    max_pairs_long: int = 3
    max_pairs_short: int = 3

    # Risk
    strategy_max_drawdown: float = 0.20
    strategy_kill_sharpe: float = -1.0

    @classmethod
    def from_env(cls) -> "FXMomentumConfig":
        return cls(
            enabled=_env_bool("STRATEGY_FX_MOMENTUM_ENABLED", False),
            min_lookback_days=_env_int("STRATEGY_FXM_MIN_LOOKBACK", 63),
            signal_threshold=_env_float("STRATEGY_FXM_SIGNAL_THRESH", 0.5),
            max_pairs_long=_env_int("STRATEGY_FXM_MAX_LONG", 3),
            max_pairs_short=_env_int("STRATEGY_FXM_MAX_SHORT", 3),
            strategy_max_drawdown=_env_float("STRATEGY_FXM_MAX_DD", 0.20),
            strategy_kill_sharpe=_env_float("STRATEGY_FXM_KILL_SHARPE", -1.0),
        )


# ---------------------------------------------------------------------------
# FX Volatility Breakout (Bollinger squeeze to expansion)
# ---------------------------------------------------------------------------


@dataclass
class FXVolBreakoutConfig:
    enabled: bool = False

    bb_window: int = 20  # Bollinger Band window
    lookback_days: int = 126  # need 6 months for bandwidth history
    squeeze_lookback: int = 126  # lookback for bandwidth percentile
    squeeze_percentile: float = 0.10  # squeeze = bandwidth in bottom 10%
    momentum_window: int = 10  # momentum during squeeze for direction

    # Risk
    strategy_max_drawdown: float = 0.20
    strategy_kill_sharpe: float = -1.0

    @classmethod
    def from_env(cls) -> "FXVolBreakoutConfig":
        return cls(
            enabled=_env_bool("STRATEGY_FX_VOL_BREAKOUT_ENABLED", False),
            bb_window=_env_int("STRATEGY_FXVB_BB_WINDOW", 20),
            lookback_days=_env_int("STRATEGY_FXVB_LOOKBACK", 126),
            squeeze_lookback=_env_int("STRATEGY_FXVB_SQUEEZE_LOOKBACK", 126),
            squeeze_percentile=_env_float("STRATEGY_FXVB_SQUEEZE_PCT", 0.10),
            momentum_window=_env_int("STRATEGY_FXVB_MOM_WINDOW", 10),
            strategy_max_drawdown=_env_float("STRATEGY_FXVB_MAX_DD", 0.20),
            strategy_kill_sharpe=_env_float("STRATEGY_FXVB_KILL_SHARPE", -1.0),
        )


# ---------------------------------------------------------------------------
# FX Carry + Trend (placeholder for Phase 2)
# ---------------------------------------------------------------------------


@dataclass
class FXConfig:
    enabled: bool = False
    pairs: List[str] = field(
        default_factory=lambda: [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "AUDUSD",
            "USDCAD",
            "USDCHF",
        ]
    )
    carry_weight: float = 0.5
    trend_weight: float = 0.5
    trend_lookback_days: int = 63
    max_pairs_long: int = 2
    max_pairs_short: int = 2
    broker: str = "alpaca"  # "alpaca" or "oanda"

    @classmethod
    def from_env(cls) -> "FXConfig":
        return cls(
            enabled=_env_bool("STRATEGY_FX_ENABLED", False),
            pairs=_env_list(
                "STRATEGY_FX_PAIRS", "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF"
            ),
            carry_weight=_env_float("STRATEGY_FX_CARRY_WEIGHT", 0.5),
            trend_weight=_env_float("STRATEGY_FX_TREND_WEIGHT", 0.5),
            trend_lookback_days=_env_int("STRATEGY_FX_TREND_LOOKBACK", 63),
            broker=_env_str("STRATEGY_FX_BROKER", "alpaca"),
        )


# ---------------------------------------------------------------------------
# Kronos — Pre-trained Foundation Model (Strategy #12)
# ---------------------------------------------------------------------------


@dataclass
class KronosConfig:
    enabled: bool = False
    model_name: str = "NeoQuasar/Kronos-base"  # mini, small, or base
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base"
    repo_path: str = "/opt/kronos"
    max_context: int = 512
    num_samples: int = 100
    prediction_length: int = 5
    initial_weight: float = 0.10

    # Risk
    strategy_max_drawdown: float = 0.20
    strategy_kill_sharpe: float = -1.0

    @classmethod
    def from_env(cls) -> "KronosConfig":
        return cls(
            enabled=_env_bool("STRATEGY_KRONOS_ENABLED", False),
            model_name=_env_str("KRONOS_MODEL_NAME", "NeoQuasar/Kronos-base"),
            tokenizer_name=_env_str(
                "KRONOS_TOKENIZER_NAME", "NeoQuasar/Kronos-Tokenizer-base"
            ),
            repo_path=_env_str("KRONOS_REPO_PATH", "/opt/kronos"),
            max_context=_env_int("KRONOS_MAX_CONTEXT", 512),
            num_samples=_env_int("KRONOS_NUM_SAMPLES", 100),
            prediction_length=_env_int("KRONOS_PREDICTION_LENGTH", 5),
            initial_weight=_env_float("STRATEGY_KRONOS_INITIAL_WEIGHT", 0.10),
            strategy_max_drawdown=_env_float("STRATEGY_KRONOS_MAX_DD", 0.20),
            strategy_kill_sharpe=_env_float("STRATEGY_KRONOS_KILL_SHARPE", -1.0),
        )


# ---------------------------------------------------------------------------
# Deep Surrogates — Neural Option Pricing (Strategy #13)
# ---------------------------------------------------------------------------


@dataclass
class DeepSurrogateConfig:
    enabled: bool = False
    repo_path: str = "/opt/deep_surrogate"
    model_type: str = "heston"  # "heston" or "bdjm"
    initial_weight: float = 0.10

    # Tail risk monitoring
    tail_risk_enabled: bool = True
    tail_risk_alert_threshold: float = 0.7

    # Risk
    strategy_max_drawdown: float = 0.20
    strategy_kill_sharpe: float = -1.0

    @classmethod
    def from_env(cls) -> "DeepSurrogateConfig":
        return cls(
            enabled=_env_bool("STRATEGY_DEEP_SURROGATES_ENABLED", False),
            repo_path=_env_str("DEEP_SURROGATE_REPO_PATH", "/opt/deep_surrogate"),
            model_type=_env_str("DEEP_SURROGATE_MODEL_TYPE", "heston"),
            initial_weight=_env_float("STRATEGY_DEEP_SURROGATES_INITIAL_WEIGHT", 0.10),
            tail_risk_enabled=_env_bool("DEEP_SURROGATE_TAIL_RISK_ENABLED", True),
            tail_risk_alert_threshold=_env_float("DEEP_SURROGATE_TAIL_RISK_ALERT", 0.7),
            strategy_max_drawdown=_env_float("STRATEGY_DEEP_SURROGATES_MAX_DD", 0.20),
            strategy_kill_sharpe=_env_float(
                "STRATEGY_DEEP_SURROGATES_KILL_SHARPE", -1.0
            ),
        )


# ---------------------------------------------------------------------------
# TDGF — American Options Pricing (Strategy #14)
# ---------------------------------------------------------------------------


@dataclass
class TDGFConfig:
    enabled: bool = False
    repo_path: str = "/opt/tdgf"
    pde_model: str = "heston"  # "black_scholes", "heston", "lifted_heston"
    hidden_layers: int = 3
    hidden_units: int = 50
    learning_rate: float = 0.001
    max_epochs: int = 5000
    initial_weight: float = 0.10

    # Risk
    strategy_max_drawdown: float = 0.20
    strategy_kill_sharpe: float = -1.0

    @classmethod
    def from_env(cls) -> "TDGFConfig":
        return cls(
            enabled=_env_bool("STRATEGY_TDGF_ENABLED", False),
            repo_path=_env_str("TDGF_REPO_PATH", "/opt/tdgf"),
            pde_model=_env_str("TDGF_PDE_MODEL", "heston"),
            hidden_layers=_env_int("TDGF_HIDDEN_LAYERS", 3),
            hidden_units=_env_int("TDGF_HIDDEN_UNITS", 50),
            learning_rate=_env_float("TDGF_LEARNING_RATE", 0.001),
            max_epochs=_env_int("TDGF_MAX_EPOCHS", 5000),
            initial_weight=_env_float("STRATEGY_TDGF_INITIAL_WEIGHT", 0.10),
            strategy_max_drawdown=_env_float("STRATEGY_TDGF_MAX_DD", 0.20),
            strategy_kill_sharpe=_env_float("STRATEGY_TDGF_KILL_SHARPE", -1.0),
        )


# ---------------------------------------------------------------------------
# Sentiment (contrarian / momentum confirmation)
# ---------------------------------------------------------------------------


@dataclass
class SentimentConfig:
    enabled: bool = False

    max_positions_per_side: int = 10
    initial_weight: float = 0.10  # 10% default ensemble weight

    # Risk
    strategy_max_drawdown: float = 0.20
    strategy_kill_sharpe: float = -1.0

    @classmethod
    def from_env(cls) -> "SentimentConfig":
        return cls(
            enabled=_env_bool("STRATEGY_SENTIMENT_ENABLED", False),
            max_positions_per_side=_env_int("STRATEGY_SENTIMENT_MAX_POS", 10),
            initial_weight=_env_float("STRATEGY_SENTIMENT_INITIAL_WEIGHT", 0.10),
            strategy_max_drawdown=_env_float("STRATEGY_SENTIMENT_MAX_DD", 0.20),
            strategy_kill_sharpe=_env_float("STRATEGY_SENTIMENT_KILL_SHARPE", -1.0),
        )


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardConfig:
    """Walk-forward cross-validation configuration."""

    is_window: int = 252  # in-sample window (trading days)
    oos_window: int = 63  # out-of-sample window (trading days)
    embargo_bars: int = 5  # embargo gap between IS and OOS
    min_sharpe: float = 0.0  # minimum Sharpe to consider a fold viable
    frequency: str = "daily"  # "daily" or "minute"
    norm_stats_dir: str = "models/norm_stats"  # where to save normalization sidecars
    sharpe_warning_threshold: float = (
        0.5  # warn if latest fold Sharpe is this much below best
    )

    @classmethod
    def from_env(cls) -> "WalkForwardConfig":
        return cls(
            is_window=_env_int("WF_IS_WINDOW", 252),
            oos_window=_env_int("WF_OOS_WINDOW", 63),
            embargo_bars=_env_int("WF_EMBARGO_BARS", 5),
            min_sharpe=_env_float("WF_MIN_SHARPE", 0.0),
            frequency=_env_str("WF_FREQUENCY", "daily"),
            norm_stats_dir=_env_str("WF_NORM_STATS_DIR", "models/norm_stats"),
            sharpe_warning_threshold=_env_float("WF_SHARPE_WARNING_THRESHOLD", 0.5),
        )


# ---------------------------------------------------------------------------
# Master config loader
# ---------------------------------------------------------------------------


@dataclass
class StrategyMasterConfig:
    """Loads all strategy configs from environment."""

    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    statarb: StatArbConfig = field(default_factory=StatArbConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    fx: FXConfig = field(default_factory=FXConfig)
    kronos: KronosConfig = field(default_factory=KronosConfig)
    deep_surrogates: DeepSurrogateConfig = field(default_factory=DeepSurrogateConfig)
    tdgf: TDGFConfig = field(default_factory=TDGFConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    sector_rotation: SectorRotationConfig = field(default_factory=SectorRotationConfig)
    fx_momentum: FXMomentumConfig = field(default_factory=FXMomentumConfig)
    fx_vol_breakout: FXVolBreakoutConfig = field(default_factory=FXVolBreakoutConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)

    @classmethod
    def from_env(cls) -> "StrategyMasterConfig":
        cfg = cls(
            momentum=MomentumConfig.from_env(),
            statarb=StatArbConfig.from_env(),
            ensemble=EnsembleConfig.from_env(),
            regime=RegimeConfig.from_env(),
            fx=FXConfig.from_env(),
            kronos=KronosConfig.from_env(),
            deep_surrogates=DeepSurrogateConfig.from_env(),
            tdgf=TDGFConfig.from_env(),
            mean_reversion=MeanReversionConfig.from_env(),
            sector_rotation=SectorRotationConfig.from_env(),
            fx_momentum=FXMomentumConfig.from_env(),
            fx_vol_breakout=FXVolBreakoutConfig.from_env(),
            sentiment=SentimentConfig.from_env(),
        )
        enabled = []
        if cfg.momentum.enabled:
            enabled.append("momentum")
        if cfg.statarb.enabled:
            enabled.append("statarb")
        if cfg.ensemble.enabled:
            enabled.append("ensemble")
        if cfg.regime.enabled:
            enabled.append("regime")
        if cfg.fx.enabled:
            enabled.append("fx")
        if cfg.kronos.enabled:
            enabled.append("kronos")
        if cfg.deep_surrogates.enabled:
            enabled.append("deep_surrogates")
        if cfg.tdgf.enabled:
            enabled.append("tdgf")
        if cfg.mean_reversion.enabled:
            enabled.append("mean_reversion")
        if cfg.sector_rotation.enabled:
            enabled.append("sector_rotation")
        if cfg.fx_momentum.enabled:
            enabled.append("fx_momentum")
        if cfg.fx_vol_breakout.enabled:
            enabled.append("fx_vol_breakout")
        if cfg.sentiment.enabled:
            enabled.append("sentiment")

        if enabled:
            logger.info("Enabled strategies: %s", ", ".join(enabled))
        else:
            logger.info(
                "No strategies enabled. Set STRATEGY_*_ENABLED=true to activate."
            )

        return cfg
