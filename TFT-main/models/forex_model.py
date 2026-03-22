"""
TFT-Forex Model — specialized TFT for currency pair forecasting.

Features (asset-class specific):
  - Cross-pair correlations (e.g., EURUSD vs GBPUSD)
  - Interest rate differentials (carry signal)
  - Currency-specific RSI, MACD, Bollinger Bands
  - USD index proxy (average of USD pairs)
  - Momentum at multiple horizons (5d, 21d, 63d)

Target: 5-day FX pair returns.
Used by: FX carry + trend strategy.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from models.base import BaseTFTModel, ModelInfo, ModelPrediction

logger = logging.getLogger(__name__)

# Interest rate differentials (base - USD for xxxUSD pairs, USD - base for USDxxx)
DEFAULT_RATE_DIFFS = {
    "EURUSD": -1.75,  # EUR 2.75% - USD 4.50%
    "GBPUSD": 0.00,  # GBP 4.50% - USD 4.50%
    "USDJPY": 4.00,  # USD 4.50% - JPY 0.50%
    "AUDUSD": -0.40,  # AUD 4.10% - USD 4.50%
    "USDCAD": 1.25,  # USD 4.50% - CAD 3.25%
    "USDCHF": 4.00,  # USD 4.50% - CHF 0.50%
}


class TFTForexModel(BaseTFTModel):

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            "max_encoder_length": 63,
            "max_prediction_length": 5,
            "batch_size": 32,
            "learning_rate": 0.001,
            "hidden_size": 48,
            "lstm_layers": 2,
            "attention_head_size": 4,
            "dropout": 0.2,
            "max_epochs": 50,
            "patience": 8,
            "quantiles": [0.1, 0.5, 0.9],
        }
        self._model = None
        self._training_dataset = None
        self._is_loaded = False
        self._trained_at: Optional[datetime] = None

    @property
    def name(self) -> str:
        return "tft_forex"

    @property
    def asset_class(self) -> str:
        return "forex"

    def prepare_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        FX-specific feature engineering.

        Input: DataFrame with [symbol, timestamp, close] (and optionally open/high/low/volume).
        Output: DataFrame ready for TimeSeriesDataSet.
        """
        df = raw_data.copy()
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        all_frames = []
        for pair, group in df.groupby("symbol"):
            g = group.copy()
            close = g["close"]

            # Returns at multiple horizons
            g["returns_1d"] = close.pct_change(1)
            g["returns_5d"] = close.pct_change(5)
            g["returns_21d"] = close.pct_change(21)
            g["returns_63d"] = close.pct_change(63)

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            g["rsi"] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            g["macd"] = ema12 - ema26
            g["macd_signal"] = g["macd"].ewm(span=9).mean()

            # Bollinger ratio
            ma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            g["bollinger_ratio"] = (close - ma20) / std20.where(std20 > 0, 1)

            # Realized vol (21d annualized)
            g["realized_vol"] = close.pct_change().rolling(21).std() * np.sqrt(252)

            # Interest rate differential (static per pair, known in advance)
            g["rate_diff"] = DEFAULT_RATE_DIFFS.get(pair, 0.0)

            # Carry signal: rate_diff normalized
            g["carry_score"] = g["rate_diff"] / 5.0

            all_frames.append(g)

        df = pd.concat(all_frames, ignore_index=True)

        # Cross-pair correlation feature: USD strength index
        # Average return of all USD-long pairs minus USD-short pairs
        usd_long_pairs = ["USDJPY", "USDCAD", "USDCHF"]
        usd_short_pairs = ["EURUSD", "GBPUSD", "AUDUSD"]

        pivot = df.pivot_table(index="timestamp", columns="symbol", values="returns_1d")
        usd_strength = pd.Series(0.0, index=pivot.index)
        for p in usd_long_pairs:
            if p in pivot.columns:
                usd_strength += pivot[p].fillna(0)
        for p in usd_short_pairs:
            if p in pivot.columns:
                usd_strength -= pivot[p].fillna(0)
        usd_strength /= max(len(usd_long_pairs) + len(usd_short_pairs), 1)

        usd_map = dict(zip(pivot.index, usd_strength))
        df["usd_strength"] = df["timestamp"].map(usd_map).fillna(0)

        # Temporal features
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Target: 5-day forward return
        for pair, group in df.groupby("symbol"):
            mask = df["symbol"] == pair
            df.loc[mask, "target"] = (
                df.loc[mask, "close"].shift(-5) / df.loc[mask, "close"] - 1
            )

        # Time index
        df["time_idx"] = df.groupby("symbol").cumcount()

        # Drop NaN rows
        feature_cols = [
            "returns_1d",
            "returns_5d",
            "rsi",
            "macd",
            "bollinger_ratio",
            "realized_vol",
            "target",
        ]
        df = df.dropna(subset=[c for c in feature_cols if c in df.columns])

        # Fill remaining NaN
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(0)

        logger.info(
            "FX features: %d rows, %d pairs, %d columns",
            len(df),
            df["symbol"].nunique(),
            len(df.columns),
        )
        return df

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Train TFT-Forex on prepared FX data."""
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import NaNLabelEncoder
        from pytorch_forecasting.metrics import QuantileLoss
        import lightning.pytorch as pl

        df = self.prepare_features(data)

        max_time_idx = df["time_idx"].max()
        training_cutoff = int(max_time_idx * 0.8)

        time_varying_known = [
            c
            for c in [
                "time_idx",
                "day_sin",
                "day_cos",
                "month_sin",
                "month_cos",
                "rate_diff",
                "carry_score",
            ]
            if c in df.columns
        ]

        time_varying_unknown = [
            c
            for c in [
                "close",
                "returns_1d",
                "returns_5d",
                "returns_21d",
                "rsi",
                "macd",
                "macd_signal",
                "bollinger_ratio",
                "realized_vol",
                "usd_strength",
            ]
            if c in df.columns
        ]

        training_ds = TimeSeriesDataSet(
            df[df["time_idx"] <= training_cutoff],
            time_idx="time_idx",
            target="target",
            group_ids=["symbol"],
            max_encoder_length=self.config["max_encoder_length"],
            max_prediction_length=self.config["max_prediction_length"],
            static_categoricals=["symbol"],
            time_varying_known_reals=time_varying_known,
            time_varying_unknown_reals=time_varying_unknown,
            add_relative_time_idx=True,
            add_target_scales=True,
            allow_missing_timesteps=True,
            categorical_encoders={"symbol": NaNLabelEncoder(add_nan=True)},
        )

        val_ds = TimeSeriesDataSet.from_dataset(
            training_ds,
            df[df["time_idx"] > training_cutoff],
            predict=True,
            stop_randomization=True,
        )

        self._training_dataset = training_ds

        train_dl = training_ds.to_dataloader(
            train=True, batch_size=self.config["batch_size"], num_workers=0
        )
        val_dl = val_ds.to_dataloader(
            train=False, batch_size=self.config["batch_size"], num_workers=0
        )

        model = TemporalFusionTransformer.from_dataset(
            training_ds,
            learning_rate=self.config["learning_rate"],
            hidden_size=self.config["hidden_size"],
            lstm_layers=self.config["lstm_layers"],
            attention_head_size=self.config["attention_head_size"],
            dropout=self.config["dropout"],
            loss=QuantileLoss(quantiles=self.config["quantiles"]),
            optimizer="ranger",
            reduce_on_plateau_patience=4,
        )

        trainer = pl.Trainer(
            max_epochs=self.config["max_epochs"],
            accelerator="auto",
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss", patience=self.config["patience"]
                ),
                pl.callbacks.ModelCheckpoint(
                    monitor="val_loss", save_top_k=1, mode="min"
                ),
            ],
            gradient_clip_val=0.1,
            enable_progress_bar=True,
        )

        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        self._model = model
        self._is_loaded = True
        self._trained_at = datetime.now()

        val_loss = float(trainer.callback_metrics.get("val_loss", 0))
        logger.info("TFT-Forex trained: val_loss=%.6f", val_loss)
        return {"val_loss": val_loss}

    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        if not self._is_loaded or self._model is None:
            return []

        try:
            df = self.prepare_features(data)

            val_ds = self._training_dataset.from_dataset(
                self._training_dataset,
                df,
                predict=True,
                stop_randomization=True,
            )
            dl = val_ds.to_dataloader(
                train=False, batch_size=self.config["batch_size"], num_workers=0
            )

            raw_preds = self._model.predict(dl, mode="prediction")
            preds_np = (
                raw_preds.numpy()
                if hasattr(raw_preds, "numpy")
                else np.array(raw_preds)
            )

            pairs = df["symbol"].unique()
            predictions = []

            for i, pair in enumerate(pairs):
                if i >= len(preds_np):
                    break

                pred = preds_np[i]
                if pred.ndim >= 2 and pred.shape[-1] >= 3:
                    lower, median, upper = (
                        float(pred[-1, 0]),
                        float(pred[-1, 1]),
                        float(pred[-1, 2]),
                    )
                else:
                    median = float(pred.flatten()[-1])
                    lower, upper = median * 0.8, median * 1.2

                spread = abs(upper - lower)
                confidence = max(0.1, 1.0 - spread * 10)

                predictions.append(
                    ModelPrediction(
                        symbol=pair,
                        predicted_value=median,
                        lower_bound=lower,
                        upper_bound=upper,
                        confidence=min(confidence, 0.95),
                        horizon_days=5,
                        model_name=self.name,
                        metadata={"rate_diff": DEFAULT_RATE_DIFFS.get(pair, 0)},
                    )
                )

            logger.info("TFT-Forex: %d predictions", len(predictions))
            return predictions

        except Exception as e:
            logger.error("TFT-Forex prediction failed: %s", e)
            return []

    def save(self, path: str) -> None:
        if self._model is None:
            raise ValueError("No model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "config": self.config,
                "training_dataset": self._training_dataset,
                "trained_at": self._trained_at,
            },
            path,
        )
        logger.info("TFT-Forex saved to %s", path)

    def load(self, path: str) -> bool:
        if not Path(path).exists():
            logger.info("TFT-Forex model not found at %s", path)
            return False
        try:
            from pytorch_forecasting import TemporalFusionTransformer

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.config = checkpoint["config"]
            self._training_dataset = checkpoint["training_dataset"]
            self._trained_at = checkpoint.get("trained_at")
            from pytorch_forecasting.metrics import QuantileLoss

            self._model = TemporalFusionTransformer.from_dataset(
                self._training_dataset,
                learning_rate=self.config["learning_rate"],
                hidden_size=self.config["hidden_size"],
                lstm_layers=self.config["lstm_layers"],
                attention_head_size=self.config["attention_head_size"],
                dropout=self.config["dropout"],
                loss=QuantileLoss(quantiles=self.config["quantiles"]),
            )
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._is_loaded = True
            logger.info("TFT-Forex loaded from %s", path)
            return True
        except Exception as e:
            logger.warning("Failed to load TFT-Forex: %s", e)
            return False

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            asset_class=self.asset_class,
            version="1.0",
            trained_at=self._trained_at,
            is_loaded=self._is_loaded,
        )
