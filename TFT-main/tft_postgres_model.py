"""
TFT Model with PostgreSQL Integration
Enhanced TFT implementation using PostgreSQL as data source
"""

"""
# COPILOT PROMPT: Add Polygon-specific features to TFT:
# - vwap_ratio: vwap relative to close price
# - news_sentiment_momentum: 3-day sentiment change
# - fundamental_zscore: sector-adjusted fundamental metrics
# Quantize model weights for faster Polygon real-time predictions
# EXPECTED OUTPUT: Enhanced TFT model with Polygon-optimized features
# POLYGON INTEGRATION: VWAP features, sentiment data, model optimization
"""


import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
    GroupNormalizer,
    EncoderNormalizer,
)
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)


class AdvancedOptionsModel:
    """
    ENHANCED COPILOT PROMPT: Advanced options pricing with Greeks calculation
    Integrates Black-Scholes-Merton model with TFT predictions for volatility-adjusted signals
    """

    def __init__(self):
        self.risk_free_rate = 0.05  # 5% risk-free rate (update from Fed data)

    def calculate_black_scholes_iv(self, option_data: Dict) -> float:
        """
        ENHANCED COPILOT PROMPT: Calculate Black-Scholes implied volatility
        Your enhanced Copilot should generate:
        1. Implement Black-Scholes formula for call/put options
        2. Use Newton-Raphson method to solve for implied volatility
        3. Handle dividend adjustments and early exercise premiums
        4. Add American vs European option style detection
        5. Return calibrated implied volatility with confidence bounds
        """
        try:
            S = option_data["stock_price"]
            K = option_data["strike_price"]
            T = option_data["time_to_expiry"]
            r = option_data.get("risk_free_rate", self.risk_free_rate)
            q = option_data.get("dividend_yield", 0.0)
            market_price = option_data["option_price"]
            option_type = option_data["option_type"].lower()

            def black_scholes_price(vol):
                """Calculate theoretical option price using Black-Scholes"""
                d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
                d2 = d1 - vol * np.sqrt(T)

                if option_type == "call":
                    price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(
                        -r * T
                    ) * norm.cdf(d2)
                else:  # put
                    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(
                        -q * T
                    ) * norm.cdf(-d1)

                return price

            def vega(vol):
                """Calculate vega (sensitivity to volatility)"""
                d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
                return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

            def objective(vol):
                return black_scholes_price(vol) - market_price

            # Newton-Raphson method for implied volatility
            vol_guess = 0.2  # Initial guess: 20% volatility
            max_iterations = 100
            tolerance = 1e-6

            for i in range(max_iterations):
                price_diff = objective(vol_guess)

                if abs(price_diff) < tolerance:
                    break

                vega_val = vega(vol_guess)
                if abs(vega_val) < 1e-8:
                    break

                vol_guess = vol_guess - price_diff / vega_val

                # Keep volatility in reasonable bounds
                vol_guess = max(0.001, min(vol_guess, 5.0))

            return vol_guess

        except Exception as e:
            logger.error(f"Failed to calculate implied volatility: {e}")
            return 0.2  # Default to 20% volatility

    def calculate_option_greeks(
        self, option_symbol: str, underlying_data: Dict
    ) -> Dict:
        """
        ENHANCED COPILOT PROMPT: Calculate complete option Greeks suite
        Your enhanced Copilot should generate:
        1. Delta: price sensitivity to underlying movement
        2. Gamma: rate of change of delta
        3. Theta: time decay
        4. Vega: volatility sensitivity
        5. Rho: interest rate sensitivity
        6. Handle dividend adjustments and early exercise
        """
        try:
            # Parse Polygon options symbol (e.g., O:SPY230818C00325000)
            parts = (
                option_symbol.split(":")[1] if ":" in option_symbol else option_symbol
            )

            # Extract components from options symbol
            # This is a simplified parser - production would use more robust parsing
            underlying_symbol = parts[:3] if len(parts) >= 3 else "SPY"
            expiry_str = parts[3:9] if len(parts) >= 9 else "230818"
            option_type = parts[9] if len(parts) > 9 else "C"
            strike_str = parts[10:] if len(parts) > 10 else "325000"

            # Convert to standard format
            strike_price = float(strike_str) / 1000  # Convert from milli-dollars
            option_type_full = "call" if option_type == "C" else "put"

            # Calculate time to expiry (simplified - would use actual date parsing)
            expiry_date = datetime.strptime(f"20{expiry_str}", "%Y%m%d")
            time_to_expiry = (expiry_date - datetime.now()).days / 365.0

            S = underlying_data.get("current_price", 100.0)
            K = strike_price
            T = max(time_to_expiry, 0.001)  # Prevent division by zero
            r = self.risk_free_rate
            q = underlying_data.get("dividend_yield", 0.0)
            vol = underlying_data.get("implied_volatility", 0.2)

            # Calculate d1 and d2
            d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)

            # Calculate Greeks
            if option_type_full == "call":
                delta = np.exp(-q * T) * norm.cdf(d1)
                price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(
                    -r * T
                ) * norm.cdf(d2)
            else:  # put
                delta = -np.exp(-q * T) * norm.cdf(-d1)
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(
                    -q * T
                ) * norm.cdf(-d1)

            gamma = np.exp(-q * T) * norm.pdf(d1) / (S * vol * np.sqrt(T))
            theta = (
                -(S * norm.pdf(d1) * vol * np.exp(-q * T)) / (2 * np.sqrt(T))
                - r
                * K
                * np.exp(-r * T)
                * norm.cdf(d2 if option_type_full == "call" else -d2)
                + q
                * S
                * np.exp(-q * T)
                * norm.cdf(d1 if option_type_full == "call" else -d1)
            ) / 365
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
            rho = (
                K
                * T
                * np.exp(-r * T)
                * norm.cdf(d2 if option_type_full == "call" else -d2)
            ) / 100

            greeks = {
                "symbol": option_symbol,
                "underlying": underlying_symbol,
                "price": price,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "rho": rho,
                "implied_volatility": vol,
                "time_to_expiry": T,
                "moneyness": S / K,
            }

            logger.info(
                f"✅ Calculated Greeks for {option_symbol}: Delta={delta:.3f}, Gamma={gamma:.3f}"
            )
            return greeks

        except Exception as e:
            logger.error(f"Failed to calculate Greeks for {option_symbol}: {e}")
            return {"error": str(e)}


from pytorch_forecasting.metrics import QuantileLoss, SMAPE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional, Tuple
import warnings
import os
import pickle
from postgres_data_pipeline import PostgresDataPipeline

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class TFTPostgresModel:
    """TFT Model with PostgreSQL integration"""

    def __init__(
        self, db_config: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize TFT model with PostgreSQL data source

        Args:
            db_config: Database configuration
            config: Model configuration parameters
        """
        self.db_config = db_config
        self.pipeline = PostgresDataPipeline(db_config)
        self.model = None
        self.training_dataset = None
        self.config = config or self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration"""
        return {
            "max_encoder_length": 63,  # ~3 months lookback
            "max_prediction_length": 5,  # 5-day forecast
            "batch_size": 64,
            "learning_rate": 0.001,
            "hidden_size": 64,
            "lstm_layers": 2,
            "attention_head_size": 4,
            "dropout": 0.2,
            "hidden_continuous_size": 32,
            "quantiles": [0.1, 0.5, 0.9],
            "max_epochs": 100,
            "target_type": "returns",
            "prediction_horizon": 5,
            "validation_split": 0.2,
            "early_stopping_patience": 10,
        }

    def prepare_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        **kwargs,
    ) -> Tuple[pd.DataFrame, TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        Prepare data for training/validation

        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Additional arguments for pipeline

        Returns:
            Tuple of (dataframe, training_dataset, validation_dataset)
        """
        logger.info(f"Preparing data for {len(symbols)} symbols")

        # Build dataset from PostgreSQL
        df = self.pipeline.build_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            target_type=self.config["target_type"],
            prediction_horizon=self.config["prediction_horizon"],
            **kwargs,
        )

        # Get feature columns
        features = self.pipeline.get_feature_columns(df)

        # Calculate training cutoff
        max_time_idx = df["time_idx"].max()
        training_cutoff = int(max_time_idx * (1 - self.config["validation_split"]))

        logger.info(f"Training cutoff at time_idx: {training_cutoff}")
        logger.info(f"Max time_idx: {max_time_idx}")

        # Create TimeSeriesDataSet for training
        training_data = df[df["time_idx"] <= training_cutoff].copy()

        # Handle target normalizer based on target type
        if self.config["target_type"] == "returns":
            target_normalizer = GroupNormalizer(
                groups=["symbol"], transformation="softplus"
            )
        elif self.config["target_type"] == "classification":
            target_normalizer = None  # No normalization for classification
        else:
            target_normalizer = EncoderNormalizer()

        self.training_dataset = TimeSeriesDataSet(
            training_data,
            time_idx="time_idx",
            target="target",
            group_ids=["symbol"],
            max_encoder_length=self.config["max_encoder_length"],
            max_prediction_length=self.config["max_prediction_length"],
            static_categoricals=features["static_categoricals"],
            static_reals=features["static_reals"],
            time_varying_known_reals=features["time_varying_known_reals"],
            time_varying_unknown_reals=features["time_varying_unknown_reals"],
            target_normalizer=target_normalizer,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        # Create validation dataset
        validation_data = df[df["time_idx"] > training_cutoff].copy()
        validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset,
            validation_data,
            predict=True,
            stop_randomization=True,
        )

        logger.info(f"Training samples: {len(self.training_dataset)}")
        logger.info(f"Validation samples: {len(validation_dataset)}")

        return df, self.training_dataset, validation_dataset

    def train(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        optimize_hyperparams: bool = False,
        n_trials: int = 20,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train TFT model

        Args:
            symbols: List of stock symbols
            start_date: Start date for training data
            end_date: End date for training data
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of optimization trials
            **kwargs: Additional arguments for data preparation

        Returns:
            Training metrics and results
        """
        logger.info("Starting TFT model training...")

        # Prepare data
        df, training_dataset, validation_dataset = self.prepare_data(
            symbols, start_date, end_date, **kwargs
        )

        # Create data loaders
        train_dataloader = training_dataset.to_dataloader(
            train=True,
            batch_size=self.config["batch_size"],
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
        )

        val_dataloader = validation_dataset.to_dataloader(
            train=False, batch_size=self.config["batch_size"] * 2, num_workers=0
        )

        if optimize_hyperparams:
            logger.info(f"Optimizing hyperparameters with {n_trials} trials...")

            # Define search space
            study = optimize_hyperparameters(
                train_dataloader,
                val_dataloader,
                model_path="optuna_test",
                n_trials=n_trials,
                max_epochs=30,
                gradient_clip_val_range=(0.01, 1.0),
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
            logger.info(f"Best hyperparameters: {best_params}")

        # Choose loss function based on target type
        if self.config["target_type"] == "classification":
            loss = torch.nn.BCEWithLogitsLoss()
        else:
            loss = QuantileLoss(quantiles=self.config["quantiles"])

        # Initialize model
        self.model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=self.config["learning_rate"],
            hidden_size=self.config["hidden_size"],
            attention_head_size=self.config["attention_head_size"],
            dropout=self.config["dropout"],
            hidden_continuous_size=self.config["hidden_continuous_size"],
            loss=loss,
            lstm_layers=self.config["lstm_layers"],
            output_size=(
                1
                if self.config["target_type"] == "classification"
                else len(self.config["quantiles"])
            ),
            reduce_on_plateau_patience=4,
            optimizer="adamw",
        )

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.config["early_stopping_patience"],
                verbose=True,
            ),
            ModelCheckpoint(
                monitor="val_loss",
                dirpath="models/",
                filename="tft_postgres_model",
                save_top_k=1,
                mode="min",
            ),
        ]

        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=self.config["max_epochs"],
            accelerator="auto",
            callbacks=callbacks,
            gradient_clip_val=0.1,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
        )

        # Train model
        trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # Save model and training dataset
        self.save_model("models/tft_model.pth")

        # Return training metrics
        training_metrics = {
            "final_train_loss": float(trainer.callback_metrics.get("train_loss", 0)),
            "final_val_loss": float(trainer.callback_metrics.get("val_loss", 0)),
            "best_val_loss": float(trainer.callback_metrics.get("val_loss", 0)),
            "epochs_trained": trainer.current_epoch,
            "config": self.config,
        }

        logger.info(
            f"Training completed. Final validation loss: {training_metrics['final_val_loss']:.4f}"
        )

        return training_metrics

    def predict(
        self, symbols: List[str], prediction_date: str, lookback_days: int = 90
    ) -> Dict[str, Any]:
        """
        Generate predictions for given symbols and date

        Args:
            symbols: List of stock symbols
            prediction_date: Date to generate predictions for
            lookback_days: Number of days to look back for features

        Returns:
            Dictionary with predictions and metadata
        """
        if self.model is None:
            raise ValueError(
                "Model not trained. Call train() first or load a trained model."
            )

        logger.info(
            f"Generating predictions for {len(symbols)} symbols on {prediction_date}"
        )

        # Calculate lookback start date
        pred_date = datetime.strptime(prediction_date, "%Y-%m-%d")
        start_date = (pred_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        # Build prediction dataset
        df = self.pipeline.build_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=prediction_date,
            target_type=self.config["target_type"],
            prediction_horizon=self.config["prediction_horizon"],
        )

        if df.empty:
            raise ValueError(f"No data available for prediction on {prediction_date}")

        # Filter to latest available data for each symbol
        latest_data = df.groupby("symbol").tail(self.config["max_encoder_length"])

        # Create prediction dataset
        prediction_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, latest_data, predict=True, stop_randomization=True
        )

        # Generate predictions
        prediction_dataloader = prediction_dataset.to_dataloader(
            train=False, batch_size=len(symbols), num_workers=0
        )

        # Get raw predictions
        raw_predictions = self.model.predict(prediction_dataloader, mode="raw")

        # Process predictions based on target type
        if self.config["target_type"] == "classification":
            # Binary classification
            probabilities = torch.sigmoid(raw_predictions["prediction"]).cpu().numpy()
            predictions = {
                "predictions": probabilities.flatten(),
                "prediction_type": "classification",
                "symbols": symbols,
                "prediction_date": prediction_date,
            }
        else:
            # Quantile predictions
            quantile_predictions = raw_predictions["prediction"].cpu().numpy()

            predictions = {
                "predictions": {
                    "quantile_0.1": quantile_predictions[:, :, 0].flatten(),
                    "quantile_0.5": quantile_predictions[:, :, 1].flatten(),
                    "quantile_0.9": quantile_predictions[:, :, 2].flatten(),
                },
                "prediction_type": "quantile",
                "quantiles": self.config["quantiles"],
                "symbols": symbols,
                "prediction_date": prediction_date,
                "prediction_horizon": self.config["prediction_horizon"],
            }

        # Add attention weights if available
        if hasattr(raw_predictions, "attention"):
            predictions["attention_weights"] = (
                raw_predictions["attention"].cpu().numpy()
            )

        logger.info(f"Generated predictions for {len(symbols)} symbols")

        return predictions

    def evaluate(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test data

        Args:
            symbols: List of stock symbols
            start_date: Start date for evaluation
            end_date: End date for evaluation

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError(
                "Model not trained. Call train() first or load a trained model."
            )

        logger.info(f"Evaluating model performance from {start_date} to {end_date}")

        # Build evaluation dataset
        df = self.pipeline.build_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            target_type=self.config["target_type"],
            prediction_horizon=self.config["prediction_horizon"],
        )

        # Create evaluation dataset
        eval_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, df, predict=True, stop_randomization=True
        )

        eval_dataloader = eval_dataset.to_dataloader(
            train=False, batch_size=self.config["batch_size"], num_workers=0
        )

        # Calculate metrics
        metrics = {}

        # Get predictions and actuals
        predictions = self.model.predict(eval_dataloader)
        actuals = torch.cat([batch[1][0] for batch in eval_dataloader])

        if self.config["target_type"] == "classification":
            # Classification metrics
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score,
            )

            pred_probs = torch.sigmoid(predictions).cpu().numpy()
            pred_classes = (pred_probs > 0.5).astype(int)
            actual_classes = actuals.cpu().numpy().astype(int)

            metrics = {
                "accuracy": accuracy_score(actual_classes, pred_classes),
                "precision": precision_score(
                    actual_classes, pred_classes, average="weighted"
                ),
                "recall": recall_score(
                    actual_classes, pred_classes, average="weighted"
                ),
                "f1_score": f1_score(actual_classes, pred_classes, average="weighted"),
                "auc_roc": roc_auc_score(actual_classes, pred_probs),
            }
        else:
            # Regression metrics
            from sklearn.metrics import (
                mean_squared_error,
                mean_absolute_error,
                r2_score,
            )

            pred_values = predictions[:, 1].cpu().numpy()  # Use median prediction
            actual_values = actuals.cpu().numpy()

            metrics = {
                "mse": mean_squared_error(actual_values, pred_values),
                "mae": mean_absolute_error(actual_values, pred_values),
                "rmse": np.sqrt(mean_squared_error(actual_values, pred_values)),
                "r2_score": r2_score(actual_values, pred_values),
                "directional_accuracy": np.mean(
                    np.sign(pred_values) == np.sign(actual_values)
                ),
            }

        logger.info(f"Evaluation completed. Key metrics: {metrics}")

        return metrics

    def save_model(self, filepath: str):
        """Save trained model and configuration.

        Saves a checkpoint dict with keys: config, model_state_dict,
        training_dataset — matching the format expected by
        EnhancedTFTModel.load_model() and TFTStocksAdapter.
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        checkpoint = {
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "training_dataset": self.training_dataset,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model and configuration"""
        # Load model checkpoint
        checkpoint_path = filepath.replace(".pth", ".ckpt")
        self.model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)

        # Load additional information
        info_path = filepath.replace(".pth", "_info.pkl")
        if os.path.exists(info_path):
            with open(info_path, "rb") as f:
                model_info = pickle.load(f)
            self.config = model_info.get("config", self.config)

        logger.info(f"Model loaded from {filepath}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError(
                "Model not trained. Call train() first or load a trained model."
            )

        # Get variable importance
        importance = self.model.log_interval()

        return importance


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Database configuration
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "database": os.getenv("POSTGRES_DB", "stock_trading_analysis"),
        "user": os.getenv("POSTGRES_USER", "trading_user"),
        "password": os.environ["POSTGRES_PASSWORD"],
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "schema": os.getenv("POSTGRES_SCHEMA", "public"),
    }

    try:
        # Initialize model
        tft_model = TFTPostgresModel(db_config)

        # Test with sample symbols (would need actual data in PostgreSQL)
        test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

        # Define date range for training
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=730)).strftime(
            "%Y-%m-%d"
        )  # 2 years

        print(f"Testing TFT model with symbols: {test_symbols}")
        print(f"Training date range: {start_date} to {end_date}")

        # Note: This would require actual data in PostgreSQL to run
        # Training would be done like this:
        # training_results = tft_model.train(
        #     symbols=test_symbols,
        #     start_date=start_date,
        #     end_date=end_date,
        #     optimize_hyperparams=False
        # )

        print("TFT PostgreSQL model initialized successfully")
        print(
            "To train the model, ensure PostgreSQL database has the required data and run:"
        )
        print("training_results = tft_model.train(symbols, start_date, end_date)")

    except Exception as e:
        print(f"Error initializing TFT PostgreSQL model: {e}")
        import traceback

        traceback.print_exc()
