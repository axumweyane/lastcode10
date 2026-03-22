#!/usr/bin/env python3
"""
TFT-Volatility Training Script.

Downloads stock + VIX data via yfinance and trains the TFT-Vol model
to forecast 5-day realized volatility.

Usage:
    python -m models.train_volatility
    python -m models.train_volatility --symbols AAPL MSFT NVDA SPY --epochs 30
    python -m models.train_volatility --period 5y --save-path models/tft_vol_v2.pth
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.volatility_model import TFTVolatilityModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_volatility")


def download_stock_data(symbols: list, period: str = "5y") -> pd.DataFrame:
    """Download stock data via yfinance Ticker API for vol model training."""
    import yfinance as yf

    # Always include VIX and SPY for regime context
    all_symbols = list(set(symbols + ["SPY", "^VIX"]))
    logger.info("Downloading %d symbols (%s)...", len(all_symbols), period)

    rows = []
    for sym in all_symbols:
        clean_sym = sym.replace("^", "")
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period=period, auto_adjust=True)
            if hist.empty:
                logger.warning("No data for %s", sym)
                continue
            for dt, row in hist.dropna().iterrows():
                rows.append(
                    {
                        "symbol": clean_sym,
                        "timestamp": dt.tz_localize(None) if dt.tzinfo else dt,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row.get("Volume", 0)),
                    }
                )
        except Exception as e:
            logger.warning("Failed to get %s: %s", sym, e)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("No data downloaded")
        return df

    # If VIX data is available, add as a feature to all other symbols
    if "VIX" in df["symbol"].unique():
        vix_data = df[df["symbol"] == "VIX"][["timestamp", "close"]].rename(
            columns={"close": "vix_level"}
        )
        df = df[df["symbol"] != "VIX"]  # remove VIX as a standalone symbol
        df = df.merge(vix_data, on="timestamp", how="left")
        df["vix_level"] = df["vix_level"].ffill().fillna(20.0)

    logger.info("Downloaded %d rows for %d symbols", len(df), df["symbol"].nunique())
    return df


def main():
    parser = argparse.ArgumentParser(description="Train TFT-Volatility model")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=[
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "JPM",
            "BAC",
            "XOM",
            "SPY",
            "QQQ",
        ],
    )
    parser.add_argument("--period", default="5y", help="yfinance period")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.0008)
    parser.add_argument("--save-path", default="models/tft_volatility.pth")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  TFT-VOLATILITY TRAINING")
    logger.info("  Symbols: %s", ", ".join(args.symbols))
    logger.info("  Period: %s | Epochs: %d", args.period, args.epochs)
    logger.info("=" * 60)

    # Download data
    data = download_stock_data(args.symbols, args.period)
    if data.empty:
        logger.error("No data downloaded, aborting")
        return

    # Configure and train
    config = {
        "max_encoder_length": 63,
        "max_prediction_length": 5,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size,
        "lstm_layers": 2,
        "attention_head_size": 4,
        "dropout": 0.15,
        "max_epochs": args.epochs,
        "patience": 8,
        "quantiles": [0.1, 0.5, 0.9],
    }

    model = TFTVolatilityModel(config)
    metrics = model.train(data)

    # Save
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.save_path)

    logger.info("=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("  Val loss: %.6f", metrics.get("val_loss", 0))
    logger.info("  Saved to: %s", args.save_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
