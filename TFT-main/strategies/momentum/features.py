"""
Cross-sectional factor computation for the momentum/mean-reversion strategy.

Factors computed:
  1. 12-1 Momentum  — 12-month return, skip last 1 month (avoids short-term reversal)
  2. 5-day Reversal  — inverted 5-day return (short-term mean reversion)
  3. Quality          — ROE proxy from price stability + profitability
  4. Volatility       — realized vol (used for risk adjustment, not as alpha)
  5. Dollar Volume    — liquidity filter

All factors are cross-sectionally z-scored (mean 0, std 1 across all symbols on
each date) so they can be combined with equal scale.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_momentum_factor(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """
    12-1 momentum: total return over [t-lookback, t-skip].

    The "skip month" is critical — without it, the factor captures short-term
    reversal (a negative signal) mixed with medium-term momentum (positive).
    Academic literature (Jegadeesh & Titman 1993) confirms this.

    Args:
        prices: DataFrame with columns [symbol, timestamp, close], sorted by
                (symbol, timestamp).
        lookback: total lookback window in trading days (default 252 = ~12 months).
        skip: days to skip at the end (default 21 = ~1 month).

    Returns:
        DataFrame with columns [symbol, timestamp, momentum_raw, momentum_zscore].
    """
    result_frames = []

    for symbol, group in prices.groupby("symbol"):
        group = group.sort_values("timestamp").copy()
        close = group["close"]

        # Return from t-lookback to t-skip
        past_price = close.shift(lookback)
        skip_price = close.shift(skip)

        # Avoid division by zero
        momentum_raw = np.where(
            past_price > 0,
            (skip_price / past_price) - 1.0,
            np.nan,
        )

        group["momentum_raw"] = momentum_raw
        result_frames.append(group[["symbol", "timestamp", "momentum_raw"]])

    df = pd.concat(result_frames, ignore_index=True)
    df["momentum_zscore"] = _cross_sectional_zscore(df, "momentum_raw")
    return df


def compute_mean_reversion_factor(
    prices: pd.DataFrame,
    lookback: int = 5,
) -> pd.DataFrame:
    """
    Short-term mean reversion: inverted 5-day return.

    Stocks that dropped the most over 5 days tend to bounce (liquidity provision
    effect). We invert the return so a large negative 5-day return becomes a
    positive signal (buy the dip).

    Returns:
        DataFrame with columns [symbol, timestamp, meanrev_raw, meanrev_zscore].
    """
    result_frames = []

    for symbol, group in prices.groupby("symbol"):
        group = group.sort_values("timestamp").copy()
        close = group["close"]

        ret_5d = close.pct_change(lookback)
        # Invert: stocks that fell the most get the highest score
        group["meanrev_raw"] = -ret_5d

        result_frames.append(group[["symbol", "timestamp", "meanrev_raw"]])

    df = pd.concat(result_frames, ignore_index=True)
    df["meanrev_zscore"] = _cross_sectional_zscore(df, "meanrev_raw")
    return df


def compute_quality_factor(
    prices: pd.DataFrame,
    fundamentals: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Quality factor proxy.

    If fundamentals are available (from Polygon), use ROE and profit margin.
    Otherwise, use a price-based proxy: stocks with higher Sharpe ratio over
    the past 63 days and lower max drawdown are "higher quality."

    This captures the quality premium documented by Asness, Frazzini & Pedersen
    (2013) — profitable, stable companies earn excess returns.

    Returns:
        DataFrame with columns [symbol, timestamp, quality_raw, quality_zscore].
    """
    result_frames = []

    for symbol, group in prices.groupby("symbol"):
        group = group.sort_values("timestamp").copy()
        close = group["close"]
        daily_ret = close.pct_change()

        # Rolling 63-day Sharpe as quality proxy
        rolling_mean = daily_ret.rolling(63, min_periods=21).mean()
        rolling_std = daily_ret.rolling(63, min_periods=21).std()
        rolling_sharpe = np.where(rolling_std > 0, rolling_mean / rolling_std, 0.0)

        # Rolling max drawdown (lower = better quality)
        rolling_max = close.rolling(63, min_periods=21).max()
        rolling_dd = (close - rolling_max) / rolling_max  # negative values

        # Quality = high Sharpe + low drawdown (negate DD so higher = better)
        quality_raw = rolling_sharpe * np.sqrt(252) - rolling_dd * 2.0

        group["quality_raw"] = quality_raw
        result_frames.append(group[["symbol", "timestamp", "quality_raw"]])

    df = pd.concat(result_frames, ignore_index=True)

    # If fundamentals available, blend in ROE
    if fundamentals is not None and "roe" in fundamentals.columns:
        df = _blend_fundamental_quality(df, fundamentals)

    df["quality_zscore"] = _cross_sectional_zscore(df, "quality_raw")
    return df


def compute_realized_volatility(
    prices: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """
    Realized volatility (annualized) for risk adjustment.

    Not an alpha factor itself, but used by position sizing and the ensemble
    optimizer for volatility targeting.

    Returns:
        DataFrame with columns [symbol, timestamp, realized_vol].
    """
    result_frames = []

    for symbol, group in prices.groupby("symbol"):
        group = group.sort_values("timestamp").copy()
        daily_ret = group["close"].pct_change()
        realized_vol = daily_ret.rolling(window, min_periods=10).std() * np.sqrt(252)
        group["realized_vol"] = realized_vol
        result_frames.append(group[["symbol", "timestamp", "realized_vol"]])

    return pd.concat(result_frames, ignore_index=True)


def compute_dollar_volume(
    prices: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Average daily dollar volume for liquidity filtering.

    Returns:
        DataFrame with columns [symbol, timestamp, avg_dollar_volume].
    """
    result_frames = []

    for symbol, group in prices.groupby("symbol"):
        group = group.sort_values("timestamp").copy()
        dollar_vol = group["close"] * group["volume"]
        group["avg_dollar_volume"] = dollar_vol.rolling(window, min_periods=10).mean()
        result_frames.append(group[["symbol", "timestamp", "avg_dollar_volume"]])

    return pd.concat(result_frames, ignore_index=True)


def compute_all_factors(
    prices: pd.DataFrame,
    momentum_lookback: int = 252,
    momentum_skip: int = 21,
    meanrev_lookback: int = 5,
    quality_fundamentals: Optional[pd.DataFrame] = None,
    vol_window: int = 21,
    dollar_vol_window: int = 20,
) -> pd.DataFrame:
    """
    Compute all factors and merge into a single DataFrame.

    Args:
        prices: DataFrame with [symbol, timestamp, open, high, low, close, volume].

    Returns:
        DataFrame with all factor columns merged on (symbol, timestamp).
    """
    required_cols = {"symbol", "timestamp", "close", "volume"}
    missing = required_cols - set(prices.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(
        "Computing factors for %d symbols, %d rows",
        prices["symbol"].nunique(),
        len(prices),
    )

    # Compute each factor independently
    mom = compute_momentum_factor(prices, momentum_lookback, momentum_skip)
    mr = compute_mean_reversion_factor(prices, meanrev_lookback)
    qual = compute_quality_factor(prices, quality_fundamentals)
    vol = compute_realized_volatility(prices, vol_window)
    dvol = compute_dollar_volume(prices, dollar_vol_window)

    # Merge all on (symbol, timestamp)
    merge_keys = ["symbol", "timestamp"]
    result = prices[["symbol", "timestamp", "close", "volume"]].copy()
    result = result.merge(mom, on=merge_keys, how="left")
    result = result.merge(mr, on=merge_keys, how="left")
    result = result.merge(qual, on=merge_keys, how="left")
    result = result.merge(vol, on=merge_keys, how="left")
    result = result.merge(dvol, on=merge_keys, how="left")

    logger.info(
        "Factor computation complete. Columns: %s",
        [
            c
            for c in result.columns
            if c not in ("symbol", "timestamp", "close", "volume")
        ],
    )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cross_sectional_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Z-score a column cross-sectionally (across all symbols on each date).

    This is the standard quant approach: on each day, rank all stocks relative
    to each other. A stock's z-score tells you how extreme it is vs. peers,
    not vs. its own history.
    """
    grouped = df.groupby("timestamp")[col]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    # Avoid division by zero on dates with a single stock or zero variance
    zscore = np.where(std > 1e-10, (df[col] - mean) / std, 0.0)
    return pd.Series(zscore, index=df.index)


def _blend_fundamental_quality(
    df: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    If fundamentals contain ROE, blend it with the price-based quality score.
    Weight: 60% price-based, 40% fundamental.
    """
    if "symbol" not in fundamentals.columns or "roe" not in fundamentals.columns:
        return df

    # Get latest ROE per symbol
    latest_roe = fundamentals.groupby("symbol")["roe"].last().reset_index()
    latest_roe.columns = ["symbol", "fund_roe"]

    df = df.merge(latest_roe, on="symbol", how="left")

    # Z-score ROE cross-sectionally
    roe_mean = df["fund_roe"].mean()
    roe_std = df["fund_roe"].std()
    if roe_std > 1e-10:
        roe_z = (df["fund_roe"] - roe_mean) / roe_std
    else:
        roe_z = 0.0

    # Blend: where fundamental data exists, use 60/40; otherwise 100% price
    has_roe = df["fund_roe"].notna()
    df.loc[has_roe, "quality_raw"] = df.loc[has_roe, "quality_raw"] * 0.6 + roe_z * 0.4

    df = df.drop(columns=["fund_roe"])
    return df
