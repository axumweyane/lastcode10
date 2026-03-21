"""
TDGF — Time Deep Gradient Flow for American option pricing.

Strategy #14: Solves option pricing PDEs using deep learning, supporting
Black-Scholes, Heston, and lifted Heston (rough volatility) models.
Handles free-boundary PDEs for American options (early exercise).

NEEDS LIGHT TRAINING on specific option parameters (minutes to hours on GPU).

Repository: https://github.com/jgrou/TDGF
Architecture: 3 layers x 50 neurons, DGM-style, softplus output (no-arbitrage)
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from models.base import BaseTFTModel, ModelInfo, ModelPrediction

logger = logging.getLogger(__name__)

TDGF_REPO_PATH = os.getenv("TDGF_REPO_PATH", "/opt/tdgf")
TDGF_HIDDEN_LAYERS = int(os.getenv("TDGF_HIDDEN_LAYERS", "3"))
TDGF_HIDDEN_UNITS = int(os.getenv("TDGF_HIDDEN_UNITS", "50"))
TDGF_LEARNING_RATE = float(os.getenv("TDGF_LEARNING_RATE", "0.001"))
TDGF_MAX_EPOCHS = int(os.getenv("TDGF_MAX_EPOCHS", "5000"))
TDGF_PDE_MODEL = os.getenv("TDGF_PDE_MODEL", "heston")


class TDGFModel(BaseTFTModel):
    """
    Time Deep Gradient Flow for American option pricing.

    Key advantage over PINN: handles free-boundary PDEs for American options
    (early exercise boundary). Uses DGM-style architecture with softplus
    output to enforce no-arbitrage constraints.

    Supports:
      - Black-Scholes (1D)
      - Heston stochastic volatility (2D)
      - Lifted Heston / rough volatility (higher-D)
      - American options up to 5 dimensions
    """

    def __init__(self):
        self._solver = None
        self._is_loaded = False
        self._is_trained = False
        self._symbols: List[str] = []
        self._trained_at: Optional[datetime] = None
        self._pde_model = TDGF_PDE_MODEL
        self._training_metrics: Dict[str, float] = {}

    @property
    def name(self) -> str:
        return "tdgf"

    @property
    def asset_class(self) -> str:
        return "volatility"

    def prepare_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare option parameters for TDGF pricing.

        Expected columns for Heston model:
          spot, strike, rate, tau, kappa, theta, sigma, rho, v0

        For Black-Scholes:
          spot, strike, rate, tau, sigma
        """
        df = raw_data.copy()
        df.columns = [c.lower() for c in df.columns]

        # Compute moneyness if not present
        if "moneyness" not in df.columns and "spot" in df.columns and "strike" in df.columns:
            df["moneyness"] = df["spot"] / df["strike"]

        # Compute time to maturity in years if dates are given
        if "tau" not in df.columns and "expiry" in df.columns:
            today = pd.Timestamp.now()
            df["tau"] = (pd.to_datetime(df["expiry"]) - today).dt.days / 365.0
            df["tau"] = df["tau"].clip(lower=1 / 365.0)

        return df

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Train TDGF network on specific option parameters.

        The TDGF solver learns the PDE solution for a given set of model
        parameters. Training takes minutes to hours depending on problem
        dimension and desired accuracy.

        Args:
            data: DataFrame with option parameters
            **kwargs: Optional overrides for epochs, lr, etc.
        """
        if not self._is_loaded:
            logger.error("TDGF solver not loaded. Call load() first.")
            return {"error": "not_loaded"}

        epochs = kwargs.get("epochs", TDGF_MAX_EPOCHS)
        lr = kwargs.get("learning_rate", TDGF_LEARNING_RATE)

        df = self.prepare_features(data)

        try:
            # Extract model parameters for training
            params = self._extract_pde_params(df)
            if params is None:
                return {"error": "invalid_params"}

            logger.info(
                "Training TDGF (%s model, %d epochs, lr=%.6f)",
                self._pde_model,
                epochs,
                lr,
            )

            # Train the TDGF solver
            result = self._solver.train(
                model_type=self._pde_model,
                params=params,
                n_epochs=epochs,
                learning_rate=lr,
                hidden_layers=TDGF_HIDDEN_LAYERS,
                hidden_units=TDGF_HIDDEN_UNITS,
            )

            self._is_trained = True
            self._trained_at = datetime.now()

            # Extract training metrics
            metrics = {}
            if isinstance(result, dict):
                metrics = {
                    "final_loss": float(result.get("loss", 0)),
                    "epochs_run": int(result.get("epochs", epochs)),
                    "pde_residual": float(result.get("pde_residual", 0)),
                    "boundary_error": float(result.get("boundary_error", 0)),
                }
            self._training_metrics = metrics

            logger.info("TDGF training complete: %s", metrics)
            return metrics

        except Exception as e:
            logger.error("TDGF training failed: %s", e)
            return {"error": str(e)}

    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """
        Price American options and compute Greeks using the trained TDGF network.
        """
        if not self._is_loaded:
            logger.debug("TDGF not loaded, returning empty predictions")
            return []

        if not self._is_trained:
            logger.debug("TDGF not trained, returning empty predictions")
            return []

        predictions = []
        symbols = (
            data["symbol"].unique() if "symbol" in data.columns else ["OPT"]
        )

        for symbol in symbols:
            try:
                sym_data = (
                    data[data["symbol"] == symbol]
                    if "symbol" in data.columns
                    else data
                )
                preds = self._predict_symbol(symbol, sym_data)
                predictions.extend(preds)
            except Exception as e:
                logger.error("TDGF prediction failed for %s: %s", symbol, e)

        logger.info("TDGF generated %d predictions", len(predictions))
        return predictions

    def _predict_symbol(
        self, symbol: str, sym_data: pd.DataFrame
    ) -> List[ModelPrediction]:
        """Price options for a single underlying."""
        df = self.prepare_features(sym_data)
        results: List[ModelPrediction] = []

        params = self._extract_pde_params(df)
        if params is None:
            return results

        try:
            # Get option prices from TDGF solver
            prices = self._solver.price(
                model_type=self._pde_model,
                params=params,
            )

            # Get Greeks if available
            greeks = None
            if hasattr(self._solver, "greeks"):
                greeks = self._solver.greeks(
                    model_type=self._pde_model,
                    params=params,
                )

            if prices is None:
                return results

            # Convert prices to array
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices).flatten()

            # Generate signal: compare TDGF price to market price for mispricing
            for i, row in df.iterrows():
                if i >= len(prices):
                    break

                tdgf_price = float(prices[i]) if i < len(prices) else 0.0
                market_price = float(row.get("market_price", tdgf_price))

                if market_price > 0 and tdgf_price > 0:
                    # Mispricing ratio: positive = option undervalued
                    mispricing = (tdgf_price - market_price) / market_price
                else:
                    mispricing = 0.0

                # Confidence depends on PDE solution quality
                confidence = min(
                    0.90,
                    0.5 + (1.0 - self._training_metrics.get("pde_residual", 0.5)) * 0.4,
                )

                metadata: Dict[str, Any] = {
                    "signal_type": "american_option_pricing",
                    "tdgf_price": tdgf_price,
                    "market_price": market_price,
                    "pde_model": self._pde_model,
                    "asset_class": "options",
                }

                # Add Greeks to metadata if available
                if greeks is not None and isinstance(greeks, dict):
                    for greek_name, greek_vals in greeks.items():
                        if isinstance(greek_vals, (list, np.ndarray)):
                            if i < len(greek_vals):
                                metadata[f"greek_{greek_name}"] = float(
                                    greek_vals[i]
                                )
                        else:
                            metadata[f"greek_{greek_name}"] = float(greek_vals)

                results.append(
                    ModelPrediction(
                        symbol=symbol,
                        predicted_value=mispricing,
                        lower_bound=mispricing - 0.05,
                        upper_bound=mispricing + 0.05,
                        confidence=confidence,
                        horizon_days=int(
                            row.get("tau", 0.1) * 365
                        ),
                        model_name=self.name,
                        metadata=metadata,
                    )
                )

        except Exception as e:
            logger.error("TDGF pricing failed for %s: %s", symbol, e)

        return results

    def _extract_pde_params(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Extract PDE parameters based on model type."""
        if df.empty:
            return None

        if self._pde_model == "black_scholes":
            required = ["spot", "strike", "rate", "tau", "sigma"]
        elif self._pde_model in ("heston", "lifted_heston"):
            required = [
                "spot",
                "strike",
                "rate",
                "tau",
                "kappa",
                "theta",
                "sigma",
                "rho",
                "v0",
            ]
        else:
            required = ["spot", "strike", "rate", "tau"]

        available = [col for col in required if col in df.columns]
        if len(available) < len(required):
            missing = set(required) - set(available)
            logger.debug("TDGF: missing PDE parameters: %s", missing)
            return None

        params = {}
        for col in required:
            vals = df[col].dropna().values
            if len(vals) > 0:
                params[col] = vals
            else:
                return None

        # Add option type (American put by default)
        params["option_type"] = df.get("option_type", pd.Series(["put"])).iloc[0]
        params["exercise_type"] = "american"

        return params

    def save(self, path: str) -> None:
        """Save trained TDGF network."""
        if not self._is_trained:
            logger.info("TDGF not trained yet; nothing to save")
            return

        try:
            import torch

            save_data = {
                "pde_model": self._pde_model,
                "trained_at": self._trained_at,
                "training_metrics": self._training_metrics,
            }
            if hasattr(self._solver, "state_dict"):
                save_data["state_dict"] = self._solver.state_dict()
            elif hasattr(self._solver, "save"):
                self._solver.save(path)
                logger.info("TDGF model saved via solver.save() to %s", path)
                return

            torch.save(save_data, path)
            logger.info("TDGF model saved to %s", path)
        except Exception as e:
            logger.error("Failed to save TDGF model: %s", e)

    def load(self, path: str = None) -> bool:
        """Load TDGF solver from the cloned repo."""
        try:
            repo_path = Path(TDGF_REPO_PATH)
            if repo_path.exists() and str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))

            from tdgf import TDGFSolver  # type: ignore

            self._solver = TDGFSolver(
                hidden_layers=TDGF_HIDDEN_LAYERS,
                hidden_units=TDGF_HIDDEN_UNITS,
            )
            self._is_loaded = True

            # Try to load pre-trained weights
            if path and Path(path).exists():
                try:
                    import torch

                    checkpoint = torch.load(path, weights_only=False)
                    if "state_dict" in checkpoint:
                        self._solver.load_state_dict(checkpoint["state_dict"])
                    self._pde_model = checkpoint.get(
                        "pde_model", self._pde_model
                    )
                    self._trained_at = checkpoint.get("trained_at")
                    self._training_metrics = checkpoint.get(
                        "training_metrics", {}
                    )
                    self._is_trained = True
                    logger.info("TDGF loaded pre-trained weights from %s", path)
                except Exception as e:
                    logger.info("No pre-trained TDGF weights at %s: %s", path, e)

            logger.info("TDGF solver loaded (model=%s)", self._pde_model)
            return True
        except ImportError as e:
            logger.warning(
                "TDGF not available (clone repo to %s): %s",
                TDGF_REPO_PATH,
                e,
            )
            return False
        except Exception as e:
            logger.error("Failed to load TDGF: %s", e)
            return False

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            asset_class=self.asset_class,
            version="1.0",
            trained_at=self._trained_at,
            symbols=self._symbols,
            model_path=TDGF_REPO_PATH,
            is_loaded=self._is_loaded,
            metrics=self._training_metrics,
        )
