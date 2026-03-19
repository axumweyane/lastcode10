#!/usr/bin/env python3
"""
TFT-Forex Training Script.

Downloads FX data via yfinance and trains the TFT-Forex model.

Usage:
    python -m models.train_forex
    python -m models.train_forex --pairs EURUSD GBPUSD USDJPY --epochs 30
    python -m models.train_forex --period 3y --save-path models/tft_forex_v2.pth
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.forex_model import TFTForexModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("train_forex")

FX_YF_MAP = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X", "USDCHF": "CHF=X",
}


def download_fx_data(pairs: list, period: str = "3y") -> pd.DataFrame:
    """Download FX data via yfinance."""
    import yfinance as yf

    yf_symbols = [FX_YF_MAP.get(p, f"{p}=X") for p in pairs]
    logger.info("Downloading %d FX pairs (%s)...", len(pairs), period)

    data = yf.download(yf_symbols, period=period, group_by="ticker",
                       auto_adjust=True, progress=False)

    rows = []
    for pair, yf_sym in zip(pairs, yf_symbols):
        try:
            sym_data = data[yf_sym].dropna() if len(yf_symbols) > 1 else data.dropna()
            for dt, row in sym_data.iterrows():
                rows.append({
                    "symbol": pair, "timestamp": dt,
                    "open": float(row["Open"]), "high": float(row["High"]),
                    "low": float(row["Low"]), "close": float(row["Close"]),
                    "volume": int(row.get("Volume", 0)),
                })
        except Exception as e:
            logger.warning("Failed to get %s: %s", pair, e)

    df = pd.DataFrame(rows)
    logger.info("Downloaded %d rows for %d pairs", len(df), df["symbol"].nunique())
    return df


def main():
    parser = argparse.ArgumentParser(description="Train TFT-Forex model")
    parser.add_argument("--pairs", nargs="+",
                        default=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF"])
    parser.add_argument("--period", default="3y", help="yfinance period (1y, 2y, 3y, 5y)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--save-path", default="models/tft_forex.pth")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  TFT-FOREX TRAINING")
    logger.info("  Pairs: %s", ", ".join(args.pairs))
    logger.info("  Period: %s | Epochs: %d", args.period, args.epochs)
    logger.info("=" * 60)

    # Download data
    data = download_fx_data(args.pairs, args.period)
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
        "dropout": 0.2,
        "max_epochs": args.epochs,
        "patience": 8,
        "quantiles": [0.1, 0.5, 0.9],
    }

    model = TFTForexModel(config)
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
