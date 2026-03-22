"""
TFT-Volatility Model — specialized TFT for realized volatility forecasting.

Features (vol-specific):
  - Realized vol at multiple windows (5d, 10d, 21d, 63d)
  - Estimated implied vol and IV-RV spread
  - IV rank and IV percentile
  - GARCH(1,1) residuals (surprise component)
  - Absolute returns (vol proxy)
  - VIX level (if available)
  - Intraday range (high-low / close)

Target: 5-day forward realized volatility (annualized).
Used by: vol arbitrage, iron condors, gamma scalping, protective puts.
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


class TFTVolatilityModel(BaseTFTModel):

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            "max_encoder_length": 63,
            "max_prediction_length": 5,
            "batch_size": 32,
            "learning_rate": 0.0008,
            "hidden_size": 48,
            "lstm_layers": 2,
            "attention_head_size": 4,
            "dropout": 0.15,
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
        return "tft_volatility"

    @property
    def asset_class(self) -> str:
        return "volatility"

    def prepare_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Vol-specific feature engineering.

        Input: DataFrame with [symbol, timestamp, open, high, low, close, volume].
        Output: DataFrame with vol features and target = 5-day forward realized vol.
        """
        df = raw_data.copy()
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        all_frames = []
        for symbol, group in df.groupby("symbol"):
            g = group.copy()
            close = g["close"]
            log_returns = np.log(close / close.shift(1))

            # Realized vol at multiple windows (annualized)
            g["rv_5d"] = log_returns.rolling(5).std() * np.sqrt(252)
            g["rv_10d"] = log_returns.rolling(10).std() * np.sqrt(252)
            g["rv_21d"] = log_returns.rolling(21).std() * np.sqrt(252)
            g["rv_63d"] = log_returns.rolling(63).std() * np.sqrt(252)

            # Absolute returns (vol proxy)
            g["abs_return"] = log_returns.abs()
            g["abs_return_5d_avg"] = g["abs_return"].rolling(5).mean()

            # Intraday range (Parkinson vol proxy)
            if "high" in g.columns and "low" in g.columns:
                hl_ratio = np.log(g["high"] / g["low"])
                g["parkinson_vol"] = hl_ratio.rolling(21).apply(
                    lambda x: np.sqrt((x**2).sum() / (4 * len(x) * np.log(2)))
                    * np.sqrt(252),
                    raw=True,
                )
            else:
                g["parkinson_vol"] = g["rv_21d"]

            # Estimated IV (RV * premium multiplier)
            g["est_iv"] = g["rv_21d"] * 1.15

            # IV-RV spread
            g["iv_rv_spread"] = g["est_iv"] - g["rv_21d"]

            # IV rank (rolling 252-day)
            rolling_high = g["est_iv"].rolling(252, min_periods=63).max()
            rolling_low = g["est_iv"].rolling(252, min_periods=63).min()
            iv_range = rolling_high - rolling_low
            g["iv_rank"] = np.where(
                iv_range > 0.001,
                (g["est_iv"] - rolling_low) / iv_range * 100,
                50.0,
            )

            # Vol of vol (volatility clustering measure)
            g["vol_of_vol"] = g["rv_5d"].rolling(21).std()

            # GARCH residuals
            g["garch_residual"] = self._compute_garch_residual(log_returns)

            # Momentum of vol (is vol trending up or down?)
            g["rv_momentum"] = g["rv_5d"] - g["rv_21d"]
            g["rv_trend"] = g["rv_21d"].pct_change(5)

            # Temporal features
            ts = pd.to_datetime(g["timestamp"])
            g["day_of_week"] = ts.dt.dayofweek
            g["month"] = ts.dt.month
            g["day_sin"] = np.sin(2 * np.pi * g["day_of_week"] / 7)
            g["day_cos"] = np.cos(2 * np.pi * g["day_of_week"] / 7)
            g["month_sin"] = np.sin(2 * np.pi * g["month"] / 12)
            g["month_cos"] = np.cos(2 * np.pi * g["month"] / 12)

            # Target: 5-day forward realized vol
            future_rets = log_returns.shift(-5).rolling(5).std() * np.sqrt(252)
            # Shift back to align: target at time t = vol realized over [t+1, t+5]
            g["target"] = future_rets.shift(-4)

            all_frames.append(g)

        df = pd.concat(all_frames, ignore_index=True)

        # Time index
        df["time_idx"] = df.groupby("symbol").cumcount()

        # Drop NaN
        key_cols = ["rv_5d", "rv_21d", "target"]
        df = df.dropna(subset=[c for c in key_cols if c in df.columns])

        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(0)

        logger.info(
            "Vol features: %d rows, %d symbols, %d columns",
            len(df),
            df["symbol"].nunique(),
            len(df.columns),
        )
        return df

    def _compute_garch_residual(self, returns: pd.Series) -> pd.Series:
        """GARCH(1,1) standardized residual as surprise measure."""
        try:
            from arch import arch_model

            clean = returns.dropna().values * 100  # scale for numerical stability
            if len(clean) < 100:
                return pd.Series(0.0, index=returns.index)

            am = arch_model(
                clean[-252:], vol="Garch", p=1, q=1, mean="Constant", rescale=False
            )
            res = am.fit(disp="off", show_warning=False)
            std_resid = res.std_resid

            # Map back to original index
            out = pd.Series(0.0, index=returns.index)
            out.iloc[-len(std_resid) :] = std_resid
            return out
        except Exception:
            return pd.Series(0.0, index=returns.index)

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Train TFT-Volatility on vol features."""
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import NaNLabelEncoder
        from pytorch_forecasting.metrics import QuantileLoss
        import lightning.pytorch as pl

        df = self.prepare_features(data)

        # Remove target outliers (vol can spike)
        q99 = df["target"].quantile(0.99)
        df.loc[df["target"] > q99, "target"] = q99

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
            ]
            if c in df.columns
        ]

        time_varying_unknown = [
            c
            for c in [
                "close",
                "rv_5d",
                "rv_10d",
                "rv_21d",
                "rv_63d",
                "abs_return",
                "abs_return_5d_avg",
                "parkinson_vol",
                "est_iv",
                "iv_rv_spread",
                "iv_rank",
                "vol_of_vol",
                "garch_residual",
                "rv_momentum",
                "rv_trend",
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
        logger.info("TFT-Volatility trained: val_loss=%.6f", val_loss)
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

            symbols = df["symbol"].unique()
            predictions = []

            for i, symbol in enumerate(symbols):
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

                # For vol predictions, ensure non-negative
                median = max(median, 0.01)
                lower = max(lower, 0.01)
                upper = max(upper, 0.01)

                spread = abs(upper - lower)
                confidence = max(0.1, 1.0 - spread / median)

                predictions.append(
                    ModelPrediction(
                        symbol=symbol,
                        predicted_value=median,
                        lower_bound=lower,
                        upper_bound=upper,
                        confidence=min(confidence, 0.95),
                        horizon_days=5,
                        model_name=self.name,
                        metadata={"predicted_metric": "realized_vol_5d_annualized"},
                    )
                )

            logger.info("TFT-Volatility: %d predictions", len(predictions))
            return predictions

        except Exception as e:
            logger.error("TFT-Volatility prediction failed: %s", e)
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
        logger.info("TFT-Volatility saved to %s", path)

    def load(self, path: str) -> bool:
        if not Path(path).exists():
            logger.info("TFT-Volatility model not found at %s", path)
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
            logger.info("TFT-Volatility loaded from %s", path)
            return True
        except Exception as e:
            logger.warning("Failed to load TFT-Volatility: %s", e)
            return False

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            asset_class=self.asset_class,
            version="1.0",
            trained_at=self._trained_at,
            is_loaded=self._is_loaded,
        )
