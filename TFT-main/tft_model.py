"""
TFT Model Implementation with Advanced Features
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss, SMAPE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


class EnhancedTFTModel:
    """
    Enhanced Temporal Fusion Transformer for stock prediction
    with advanced configuration and training capabilities
    """

    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.model = None
        self.training_dataset = None
        self.validation_dataset = None
        self.trainer = None

    def _get_default_config(self) -> Dict:
        """Default model configuration"""
        return {
            "max_encoder_length": 63,  # ~3 months of trading days
            "max_prediction_length": 5,  # 5-day forecast
            "batch_size": 64,
            "learning_rate": 0.001,
            "hidden_size": 64,
            "lstm_layers": 2,
            "attention_head_size": 4,
            "dropout": 0.2,
            "hidden_continuous_size": 32,
            "max_epochs": 100,
            "patience": 10,
            "quantiles": [0.1, 0.5, 0.9],
            "loss_type": "quantile",  # 'quantile', 'mse', 'crossentropy'
        }

    def create_datasets(
        self, df: pd.DataFrame, validation_split: float = 0.2
    ) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """Create training and validation datasets"""

        # Determine split point
        max_time_idx = df["time_idx"].max()
        training_cutoff = int(max_time_idx * (1 - validation_split))

        # Define categorical and continuous variables
        static_categoricals = ["symbol"]
        if "sector" in df.columns:
            static_categoricals.append("sector")

        time_varying_known_reals = [
            "time_idx",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
            "day_of_week",
            "month",
            "quarter",
            "is_month_end",
            "is_quarter_end",
        ]

        # Add earnings/event indicators if available
        event_cols = [col for col in df.columns if "earnings" in col or "event" in col]
        time_varying_known_reals.extend(event_cols)

        time_varying_unknown_reals = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "macd",
            "macd_signal",
            "bollinger_ratio",
            "volume_ratio",
            "returns_1d",
            "returns_5d",
            "returns_20d",
        ]

        # Add sentiment if available
        if "sentiment" in df.columns:
            time_varying_unknown_reals.append("sentiment")

        # Filter available columns
        time_varying_known_reals = [
            col for col in time_varying_known_reals if col in df.columns
        ]
        time_varying_unknown_reals = [
            col for col in time_varying_unknown_reals if col in df.columns
        ]
        static_categoricals = [col for col in static_categoricals if col in df.columns]

        print(f"Static categoricals: {static_categoricals}")
        print(f"Time varying known reals: {time_varying_known_reals}")
        print(f"Time varying unknown reals: {time_varying_unknown_reals}")

        # Create training dataset
        training = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="target",
            group_ids=["symbol"],
            max_encoder_length=self.config["max_encoder_length"],
            max_prediction_length=self.config["max_prediction_length"],
            static_categoricals=static_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            add_relative_time_idx=True,
            add_target_scales=True,
            allow_missing_timesteps=True,
            categorical_encoders={"symbol": NaNLabelEncoder(add_nan=True)},
        )

        # Create validation dataset
        validation = TimeSeriesDataSet.from_dataset(
            training,
            df[lambda x: x.time_idx > training_cutoff],
            predict=True,
            stop_randomization=True,
        )

        self.training_dataset = training
        self.validation_dataset = validation

        print(f"Training dataset length: {len(training)}")
        print(f"Validation dataset length: {len(validation)}")

        return training, validation

    def create_model(
        self, training_dataset: TimeSeriesDataSet
    ) -> TemporalFusionTransformer:
        """Create TFT model with specified configuration"""

        # Choose loss function
        if self.config["loss_type"] == "quantile":
            loss = QuantileLoss(quantiles=self.config["quantiles"])
        elif self.config["loss_type"] == "mse":
            loss = nn.MSELoss()
        elif self.config["loss_type"] == "crossentropy":
            loss = nn.CrossEntropyLoss()
        else:
            loss = QuantileLoss(quantiles=self.config["quantiles"])

        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=self.config["learning_rate"],
            hidden_size=self.config["hidden_size"],
            lstm_layers=self.config["lstm_layers"],
            attention_head_size=self.config["attention_head_size"],
            dropout=self.config["dropout"],
            hidden_continuous_size=self.config["hidden_continuous_size"],
            loss=loss,
            log_interval=10,
            reduce_on_plateau_patience=4,
            optimizer="ranger",  # Modern optimizer
        )

        self.model = tft
        return tft

    def train(
        self,
        training_dataset: TimeSeriesDataSet,
        validation_dataset: TimeSeriesDataSet,
        use_gpu: bool = True,
    ) -> pl.Trainer:
        """Train the TFT model"""

        # Create data loaders
        train_dataloader = training_dataset.to_dataloader(
            train=True, batch_size=self.config["batch_size"], num_workers=4
        )

        val_dataloader = validation_dataset.to_dataloader(
            train=False, batch_size=self.config["batch_size"], num_workers=4
        )

        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=self.config["max_epochs"],
            accelerator="auto",
            devices="auto" if use_gpu else None,
            precision="16-mixed" if use_gpu else 32,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss", patience=self.config["patience"]
                ),
                pl.callbacks.LearningRateMonitor(logging_interval="step"),
                pl.callbacks.ModelCheckpoint(
                    monitor="val_loss", save_top_k=3, mode="min"
                ),
            ],
            gradient_clip_val=0.1,
            logger=True,
        )

        # Train model
        if self.model is None:
            self.model = self.create_model(training_dataset)

        trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        self.trainer = trainer
        return trainer

    def predict(
        self, dataset: TimeSeriesDataSet, mode: str = "prediction"
    ) -> np.ndarray:
        """Generate predictions"""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        dataloader = dataset.to_dataloader(
            train=False, batch_size=self.config["batch_size"], num_workers=4
        )

        if mode == "raw":
            predictions, x = self.model.predict(dataloader, mode="raw", return_x=True)
            return predictions, x
        else:
            predictions = self.model.predict(dataloader, mode="prediction")
            return predictions

    def save_model(self, path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
                "training_dataset": self.training_dataset,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location="cpu")
        self.config = checkpoint["config"]
        self.training_dataset = checkpoint["training_dataset"]

        # Recreate model
        self.model = self.create_model(self.training_dataset)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {path}")

    def optimize_hyperparameters(
        self,
        training_dataset: TimeSeriesDataSet,
        validation_dataset: TimeSeriesDataSet,
        n_trials: int = 20,
    ) -> Dict:
        """Optimize hyperparameters using Optuna"""

        study = optimize_hyperparameters(
            training_dataset,
            validation_dataset,
            model_path="optuna_test",
            n_trials=n_trials,
            max_epochs=50,
            gradient_clip_val=0.1,
            hidden_size_range=(16, 128),
            hidden_continuous_size_range=(8, 64),
            attention_head_size_range=(1, 8),
            learning_rate_range=(0.0001, 0.1),
            dropout_range=(0.1, 0.4),
            trainer_kwargs=dict(limit_train_batches=30, accelerator="auto"),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
        )

        # Update config with best parameters
        best_params = study.best_trial.params
        self.config.update(best_params)

        print(f"Best hyperparameters: {best_params}")
        return best_params


class TFTTrainingPipeline:
    """Complete training pipeline for TFT stock prediction"""

    def __init__(self, config: Dict = None):
        self.config = config
        self.model = EnhancedTFTModel(config)
        self.training_history = []

    def run_pipeline(
        self,
        df: pd.DataFrame,
        validation_split: float = 0.2,
        optimize_hyperparams: bool = False,
        n_trials: int = 20,
    ) -> EnhancedTFTModel:
        """Run complete training pipeline"""

        print("=== TFT Training Pipeline ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Number of symbols: {df['symbol'].nunique()}")

        # Create datasets
        print("\n1. Creating datasets...")
        training_dataset, validation_dataset = self.model.create_datasets(
            df, validation_split
        )

        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            print(f"\n2. Optimizing hyperparameters ({n_trials} trials)...")
            best_params = self.model.optimize_hyperparameters(
                training_dataset, validation_dataset, n_trials
            )
            print(f"Best parameters: {best_params}")

        # Train model
        print("\n3. Training model...")
        trainer = self.model.train(training_dataset, validation_dataset)

        # Store training history
        self.training_history.append(
            {
                "timestamp": pd.Timestamp.now(),
                "train_loss": trainer.logged_metrics.get("train_loss", None),
                "val_loss": trainer.logged_metrics.get("val_loss", None),
                "config": self.config,
            }
        )

        print("\n4. Training completed!")
        print(
            f"Best validation loss: {trainer.callback_metrics.get('val_loss', 'N/A')}"
        )

        return self.model


if __name__ == "__main__":
    # Test the model with sample data
    from data_preprocessing import StockDataPreprocessor, load_sample_data

    print("Loading and preprocessing sample data...")
    df = load_sample_data()

    preprocessor = StockDataPreprocessor()
    processed_df = preprocessor.fit_transform(df, target_type="returns")

    print("\nInitializing TFT model...")
    config = {
        "max_encoder_length": 30,
        "max_prediction_length": 1,
        "batch_size": 32,
        "max_epochs": 5,  # Reduced for testing
    }

    pipeline = TFTTrainingPipeline(config)
    model = pipeline.run_pipeline(processed_df, validation_split=0.2)

    print("\nModel training completed!")
