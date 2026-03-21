"""
Deep Surrogates — Neural network surrogate for Heston/Bates option pricing.

Strategy #13: Options pricing acceleration + crash prediction via tail risk index.
Pre-trained neural surrogates for IV, Greeks, and prices — 100-1000x faster
than traditional numerical methods.

NO TRAINING NEEDED — pre-trained surrogates included in repo.

Repository: https://github.com/DeepSurrogate/OptionPricing

Key feature: Build a TAIL RISK INDEX from daily recalibrated Heston parameters
that predicts market crashes by tracking parameter evolution.
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

DEEP_SURROGATE_REPO_PATH = os.getenv(
    "DEEP_SURROGATE_REPO_PATH", "/opt/deep_surrogate"
)
DEEP_SURROGATE_MODEL_TYPE = os.getenv("DEEP_SURROGATE_MODEL_TYPE", "heston")

# Tail risk index thresholds
TAIL_RISK_VOL_OF_VOL_WEIGHT = 0.35
TAIL_RISK_CORRELATION_WEIGHT = 0.25
TAIL_RISK_SKEW_WEIGHT = 0.25
TAIL_RISK_JUMP_WEIGHT = 0.15


class DeepSurrogateModel(BaseTFTModel):
    """
    Wraps the DeepSurrogate option pricing models for fast IV/Greeks computation
    and tail risk monitoring.

    Produces two types of signals:
      1. Options signals: IV surface anomalies, mispricing detection
      2. Risk signals: Tail risk index from calibrated Heston parameters
    """

    def __init__(self):
        self._surrogate = None
        self._is_loaded = False
        self._symbols: List[str] = []
        self._model_type = DEEP_SURROGATE_MODEL_TYPE
        self._param_history: List[Dict[str, float]] = []
        self._tail_risk_history: List[float] = []

    @property
    def name(self) -> str:
        return "deep_surrogates"

    @property
    def asset_class(self) -> str:
        return "volatility"

    def prepare_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare Heston model parameters for the surrogate.

        Expected columns: kappa, theta, sigma, rho, v0, rate, tau, moneyness
        Or: symbol, strike, expiry, spot, rate — we compute the rest.
        """
        df = raw_data.copy()
        df.columns = [c.lower() for c in df.columns]

        # If raw options chain data, compute Heston input parameters
        if "moneyness" not in df.columns and "strike" in df.columns:
            if "spot" in df.columns:
                df["moneyness"] = df["strike"] / df["spot"]

        return df

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Deep Surrogates are pre-trained. No training required."""
        logger.info("DeepSurrogate is a pre-trained model. No training needed.")
        return {"status": "pre_trained", "val_loss": 0.0}

    def predict(self, data: pd.DataFrame) -> List[ModelPrediction]:
        """
        Generate option pricing predictions and tail risk signals.

        For each symbol/underlying:
        1. Compute IV surface using fast surrogate
        2. Calculate Greeks (delta, gamma, vega, theta)
        3. Calibrate Heston parameters
        4. Compute tail risk index from parameter evolution
        """
        if not self._is_loaded:
            logger.debug("DeepSurrogate not loaded, returning empty predictions")
            return []

        predictions = []
        symbols = (
            data["symbol"].unique() if "symbol" in data.columns else ["SPY"]
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
                logger.error(
                    "DeepSurrogate prediction failed for %s: %s", symbol, e
                )

        logger.info("DeepSurrogate generated %d predictions", len(predictions))
        return predictions

    def _predict_symbol(
        self, symbol: str, sym_data: pd.DataFrame
    ) -> List[ModelPrediction]:
        """Compute IV surface and tail risk for a single underlying."""
        results: List[ModelPrediction] = []
        df = self.prepare_features(sym_data)

        # Compute IV using surrogate
        iv_surface = self._compute_iv_surface(df)
        if iv_surface is None:
            return results

        # Calibrate Heston parameters from the IV surface
        heston_params = self._calibrate_heston(df, iv_surface)
        if heston_params is not None:
            self._param_history.append(heston_params)
            # Keep last 252 trading days of params
            if len(self._param_history) > 252:
                self._param_history = self._param_history[-252:]

            # Compute tail risk index
            tail_risk = self._compute_tail_risk_index(heston_params)
            self._tail_risk_history.append(tail_risk)

            # Generate tail risk signal as a ModelPrediction
            # Higher tail_risk = more crash risk = bearish signal
            risk_signal = -tail_risk  # negative = bearish
            confidence = min(0.95, 0.3 + abs(tail_risk) * 0.5)

            results.append(
                ModelPrediction(
                    symbol=symbol,
                    predicted_value=risk_signal,
                    lower_bound=risk_signal - 0.1,
                    upper_bound=risk_signal + 0.1,
                    confidence=confidence,
                    horizon_days=5,
                    model_name=self.name,
                    metadata={
                        "signal_type": "tail_risk",
                        "tail_risk_index": tail_risk,
                        "heston_kappa": heston_params.get("kappa", 0),
                        "heston_theta": heston_params.get("theta", 0),
                        "heston_sigma": heston_params.get("sigma", 0),
                        "heston_rho": heston_params.get("rho", 0),
                        "heston_v0": heston_params.get("v0", 0),
                        "asset_class": "options",
                    },
                )
            )

        # Generate IV-based mispricing signal
        iv_signal = self._compute_iv_signal(iv_surface, symbol)
        if iv_signal is not None:
            results.append(iv_signal)

        return results

    def _compute_iv_surface(
        self, df: pd.DataFrame
    ) -> Optional[Dict[str, np.ndarray]]:
        """Use surrogate to compute IV for all strikes/expiries."""
        required = ["kappa", "theta", "sigma", "rho", "v0", "rate", "tau", "moneyness"]
        if not all(col in df.columns for col in required):
            logger.debug("DeepSurrogate: missing columns for IV computation")
            return None

        try:
            input_df = df[required].dropna()
            if input_df.empty:
                return None

            ivs = self._surrogate.get_iv(input_df, model_type=self._model_type)
            deltas = self._surrogate.get_iv_delta(
                input_df, model_type=self._model_type
            )

            return {
                "iv": np.array(ivs) if not isinstance(ivs, np.ndarray) else ivs,
                "delta": (
                    np.array(deltas)
                    if not isinstance(deltas, np.ndarray)
                    else deltas
                ),
                "moneyness": input_df["moneyness"].values,
                "tau": input_df["tau"].values,
            }
        except Exception as e:
            logger.error("IV surface computation failed: %s", e)
            return None

    def _calibrate_heston(
        self, df: pd.DataFrame, iv_surface: Dict[str, np.ndarray]
    ) -> Optional[Dict[str, float]]:
        """Extract/calibrate Heston parameters from the data or IV surface."""
        # If parameters are directly in the data, use them
        param_cols = ["kappa", "theta", "sigma", "rho", "v0"]
        if all(col in df.columns for col in param_cols):
            row = df[param_cols].iloc[-1]
            return {col: float(row[col]) for col in param_cols}

        # Otherwise, use median of available parameters
        params = {}
        for col in param_cols:
            if col in df.columns:
                params[col] = float(df[col].median())
        return params if params else None

    def _compute_tail_risk_index(self, params: Dict[str, float]) -> float:
        """
        Build a tail risk index from Heston parameters.

        The index captures:
        - Vol-of-vol (sigma): higher = more tail risk
        - Correlation (rho): more negative = more crash risk (leverage effect)
        - Variance skew: deviation from long-run variance
        - Jump intensity: if using Bates model

        Returns a score in [0, 1] where 1 = maximum crash risk.
        """
        sigma = params.get("sigma", 0.3)
        rho = params.get("rho", -0.7)
        theta = params.get("theta", 0.04)
        v0 = params.get("v0", 0.04)
        kappa = params.get("kappa", 2.0)

        # Vol-of-vol component: normalize sigma to [0,1]
        # Typical range: 0.1 to 1.5
        vol_of_vol_score = min(1.0, max(0.0, (sigma - 0.1) / 1.4))

        # Correlation component: more negative = higher risk
        # Typical range: -1.0 to 0.0
        correlation_score = min(1.0, max(0.0, -rho))

        # Skew component: v0 much higher than theta = elevated risk
        if theta > 0:
            variance_ratio = v0 / theta
            skew_score = min(1.0, max(0.0, (variance_ratio - 1.0) / 3.0))
        else:
            skew_score = 0.5

        # Mean reversion speed: lower kappa = risk persists longer
        jump_score = min(1.0, max(0.0, 1.0 - kappa / 10.0))

        # Weighted combination
        tail_risk = (
            TAIL_RISK_VOL_OF_VOL_WEIGHT * vol_of_vol_score
            + TAIL_RISK_CORRELATION_WEIGHT * correlation_score
            + TAIL_RISK_SKEW_WEIGHT * skew_score
            + TAIL_RISK_JUMP_WEIGHT * jump_score
        )

        # Add momentum component if we have history
        if len(self._tail_risk_history) >= 5:
            recent = np.mean(self._tail_risk_history[-5:])
            older = np.mean(self._tail_risk_history[-21:-5]) if len(self._tail_risk_history) >= 21 else recent
            momentum = max(0.0, recent - older)
            tail_risk = min(1.0, tail_risk + momentum * 0.5)

        return round(tail_risk, 4)

    def _compute_iv_signal(
        self, iv_surface: Dict[str, np.ndarray], symbol: str
    ) -> Optional[ModelPrediction]:
        """Generate a signal from IV surface shape (skew, term structure)."""
        ivs = iv_surface.get("iv")
        moneyness = iv_surface.get("moneyness")
        if ivs is None or moneyness is None or len(ivs) == 0:
            return None

        atm_mask = np.abs(moneyness - 1.0) < 0.05
        if not np.any(atm_mask):
            return None

        atm_iv = float(np.mean(ivs[atm_mask]))

        # IV level signal: high IV relative to history suggests mean reversion
        # (simplified — a full implementation would use IV rank)
        iv_signal = -(atm_iv - 0.20) * 2.0  # mean-revert around 20% vol
        iv_signal = max(-1.0, min(1.0, iv_signal))

        return ModelPrediction(
            symbol=symbol,
            predicted_value=iv_signal,
            lower_bound=iv_signal - 0.15,
            upper_bound=iv_signal + 0.15,
            confidence=min(0.85, 0.4 + abs(iv_signal) * 0.3),
            horizon_days=5,
            model_name=self.name,
            metadata={
                "signal_type": "iv_surface",
                "atm_iv": atm_iv,
                "asset_class": "options",
            },
        )

    def calibrate_heston(
        self,
        market_data: pd.DataFrame,
        n_restarts: int = 3,
    ) -> Dict[str, float]:
        """
        Calibrate Heston model parameters from market option prices.

        Uses multi-start L-BFGS-B optimization with the DeepSurrogate for
        fast evaluation (replaces slow Monte Carlo). Best result across
        restarts is returned. Validates the Feller condition on the output.

        Args:
            market_data: DataFrame with columns: strike, spot, rate, tau,
                         market_price (observed option prices).
            n_restarts: Number of random starting points to try.

        Returns:
            Dict with calibrated parameters: kappa, theta, sigma, rho, v0,
            plus calibration_error, feller_satisfied, and success flag.
            Empty dict on failure.
        """
        if not self._is_loaded or self._surrogate is None:
            logger.error("calibrate_heston: surrogate not loaded")
            return {}

        from scipy.optimize import minimize

        df = self.prepare_features(market_data)
        required = ["strike", "spot", "rate", "tau", "market_price"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error("calibrate_heston: missing columns %s", missing)
            return {}

        df = df.dropna(subset=required)
        if len(df) < 3:
            logger.warning("calibrate_heston: need >= 3 option prices, got %d", len(df))
            return {}

        market_prices = df["market_price"].values
        # Pre-compute moneyness once
        if "moneyness" not in df.columns:
            df = df.copy()
            df["moneyness"] = df["strike"] / df["spot"]

        surrogate_cols = ["kappa", "theta", "sigma", "rho", "v0", "rate", "tau", "moneyness"]

        def objective(params):
            kappa, theta, sigma, rho, v0 = params
            cal_df = df.copy()
            cal_df["kappa"] = kappa
            cal_df["theta"] = theta
            cal_df["sigma"] = sigma
            cal_df["rho"] = rho
            cal_df["v0"] = v0

            try:
                model_prices = self._surrogate.get_price(
                    cal_df[surrogate_cols], model_type=self._model_type,
                )
                model_prices = np.array(model_prices).flatten()
                if len(model_prices) != len(market_prices):
                    return 1e10
                # Relative RMSE to handle different price magnitudes
                denom = np.maximum(market_prices, 0.01)
                return float(np.sum(((model_prices - market_prices) / denom) ** 2))
            except Exception:
                return 1e10

        bounds = [
            (0.01, 20.0),   # kappa (mean reversion speed)
            (0.001, 1.0),   # theta (long-run variance)
            (0.01, 2.0),    # sigma (vol of vol)
            (-0.99, 0.0),   # rho (correlation, negative for equities)
            (0.001, 1.0),   # v0 (initial variance)
        ]

        # Multi-start: deterministic first, then random
        starting_points = [
            [2.0, 0.04, 0.3, -0.7, 0.04],  # typical equity params
        ]
        rng = np.random.default_rng(42)
        for _ in range(n_restarts - 1):
            starting_points.append([
                rng.uniform(0.5, 10.0),    # kappa
                rng.uniform(0.01, 0.15),   # theta
                rng.uniform(0.1, 1.0),     # sigma
                rng.uniform(-0.95, -0.2),  # rho
                rng.uniform(0.01, 0.15),   # v0
            ])

        best_result = None
        best_fun = float("inf")

        for x0 in starting_points:
            try:
                result = minimize(
                    objective, x0, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 200, "ftol": 1e-10},
                )
                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
            except Exception as e:
                logger.debug("Calibration restart failed: %s", e)

        if best_result is None:
            logger.error("All calibration restarts failed")
            return {}

        kappa, theta, sigma, rho, v0 = best_result.x

        # Feller condition: 2*kappa*theta > sigma^2
        # When satisfied, variance process stays strictly positive
        feller = 2.0 * kappa * theta > sigma ** 2

        calibrated = {
            "kappa": round(kappa, 6),
            "theta": round(theta, 6),
            "sigma": round(sigma, 6),
            "rho": round(rho, 6),
            "v0": round(v0, 6),
            "calibration_error": round(best_fun, 8),
            "feller_satisfied": feller,
            "success": best_result.success,
            "n_options": len(df),
        }

        if not feller:
            logger.warning(
                "Heston calibration: Feller condition NOT satisfied "
                "(2*kappa*theta=%.4f < sigma^2=%.4f). "
                "Variance process may hit zero.",
                2 * kappa * theta, sigma ** 2,
            )

        logger.info(
            "Heston calibrated from %d options: kappa=%.3f theta=%.4f "
            "sigma=%.3f rho=%.3f v0=%.4f (error=%.6f, feller=%s)",
            len(df), kappa, theta, sigma, rho, v0, best_fun, feller,
        )
        return calibrated

    def save(self, path: str) -> None:
        """DeepSurrogate is pre-trained — nothing to save."""
        logger.info("DeepSurrogate is pre-trained; no local model to save")

    def load(self, path: str = None) -> bool:
        """Load DeepSurrogate from the cloned repo."""
        try:
            repo_path = Path(DEEP_SURROGATE_REPO_PATH)
            if repo_path.exists() and str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))

            from source.deepsurrogate import DeepSurrogate  # type: ignore

            self._surrogate = DeepSurrogate(self._model_type)
            self._is_loaded = True
            logger.info(
                "DeepSurrogate loaded (model_type=%s)", self._model_type
            )
            return True
        except ImportError as e:
            logger.warning(
                "DeepSurrogate not available (clone repo to %s): %s",
                DEEP_SURROGATE_REPO_PATH,
                e,
            )
            return False
        except Exception as e:
            logger.error("Failed to load DeepSurrogate: %s", e)
            return False

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            asset_class=self.asset_class,
            version="1.0",
            symbols=self._symbols,
            model_path=DEEP_SURROGATE_REPO_PATH,
            is_loaded=self._is_loaded,
            metrics={
                "tail_risk_latest": (
                    self._tail_risk_history[-1]
                    if self._tail_risk_history
                    else 0.0
                ),
                "param_history_length": len(self._param_history),
            },
        )
