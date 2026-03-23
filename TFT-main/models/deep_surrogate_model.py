"""
Deep Surrogates — Heston stochastic volatility option pricer.

Built-in PyTorch implementation using the Heston semi-closed-form
characteristic function for GPU-accelerated option pricing, IV
computation, and Greeks. No external dependencies required.

Key feature: Build a TAIL RISK INDEX from daily recalibrated Heston
parameters that predicts market crashes by tracking parameter evolution.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm as sp_norm

from models.base import BaseTFTModel, ModelInfo, ModelPrediction

logger = logging.getLogger(__name__)

DEEP_SURROGATE_MODEL_TYPE = os.getenv("DEEP_SURROGATE_MODEL_TYPE", "heston")

# Tail risk index thresholds
TAIL_RISK_VOL_OF_VOL_WEIGHT = 0.35
TAIL_RISK_CORRELATION_WEIGHT = 0.25
TAIL_RISK_SKEW_WEIGHT = 0.25
TAIL_RISK_JUMP_WEIGHT = 0.15


class HestonEngine:
    """
    Heston stochastic volatility option pricer using the semi-closed-form
    characteristic function with numerical integration.

    GPU-accelerated via PyTorch for batched pricing. Uses the 'little Heston
    trap' formulation (Albrecher et al.) for numerical stability.

    Provides:
      - get_price(): Heston call prices (normalized, C/S)
      - get_iv(): Black-Scholes implied volatilities from Heston prices
      - get_iv_delta(): BS deltas at the implied vol
    """

    def __init__(self, device=None, n_points=256, u_max=50.0):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._n_points = n_points
        self._u_max = u_max
        self._du = u_max / n_points
        self._u = torch.linspace(
            self._du / 2, u_max - self._du / 2, n_points,
            dtype=torch.float64, device=self.device,
        )

    def _heston_call(self, moneyness, tau, rate, kappa, theta, sigma_h, rho, v0):
        """
        Vectorized Heston European call prices (normalized: S=1, K=moneyness).

        Uses the 'little Heston trap' formulation for numerical stability.
        All inputs: 1D float64 tensors of length N.
        Returns: 1D float64 tensor of normalized call prices C/S.
        """
        tau = torch.clamp(tau, min=1.0 / 365.0)
        sigma_h = torch.clamp(sigma_h, min=0.01)
        kappa = torch.clamp(kappa, min=0.01)
        theta = torch.clamp(theta, min=1e-4)
        v0 = torch.clamp(v0, min=1e-4)

        K = moneyness
        u = self._u
        du = self._du

        # Broadcast: (N, 1) x (1, Q) -> (N, Q)
        K_ = K.unsqueeze(1)
        r_ = rate.unsqueeze(1)
        T_ = tau.unsqueeze(1)
        kap_ = kappa.unsqueeze(1)
        the_ = theta.unsqueeze(1)
        sig_ = sigma_h.unsqueeze(1)
        rh_ = rho.unsqueeze(1)
        v_ = v0.unsqueeze(1)
        u_ = u.unsqueeze(0)
        a_ = kap_ * the_

        P = []
        for j in [1, 2]:
            uj = 0.5 if j == 1 else -0.5
            bj = (kap_ - rh_ * sig_) if j == 1 else kap_

            iu = 1j * u_
            rsi = rh_ * sig_ * iu

            disc = (rsi - bj) ** 2 + sig_ ** 2 * (2 * uj * iu + u_ ** 2)
            d = torch.sqrt(disc + 0j)

            g_num = bj - rsi - d
            g_den = bj - rsi + d
            g = g_num / (g_den + 1e-30)

            exp_neg_dt = torch.exp(-d * T_)
            denom = 1.0 - g * exp_neg_dt
            denom = denom + (denom.abs() < 1e-30) * 1e-30

            D_val = (g_num / (sig_ ** 2 + 1e-30)) * (
                (1.0 - exp_neg_dt) / denom
            )

            log_arg = denom / (1.0 - g + 1e-30)
            C_val = r_ * iu * T_ + (a_ / (sig_ ** 2 + 1e-30)) * (
                g_num * T_ - 2.0 * torch.log(log_arg + 0j)
            )

            f = torch.exp(C_val + D_val * v_)
            integrand = (torch.exp(-iu * torch.log(K_ + 0j)) * f / (iu + 1e-30)).real

            Pj = 0.5 + (1.0 / np.pi) * torch.sum(integrand * du, dim=1)
            P.append(torch.clamp(Pj, 0.0, 1.0))

        P1, P2 = P
        prices = P1 - K * torch.exp(-rate * tau) * P2
        return torch.clamp(prices, min=0.0)

    @staticmethod
    def _bs_call_np(sigma, moneyness, rate, tau):
        """Black-Scholes call price (S=1, K=moneyness). Numpy."""
        sigma = np.maximum(sigma, 1e-8)
        tau = np.maximum(tau, 1e-8)
        sqrt_t = np.sqrt(tau)
        d1 = (np.log(1.0 / np.maximum(moneyness, 1e-8))
              + (rate + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        return sp_norm.cdf(d1) - moneyness * np.exp(-rate * tau) * sp_norm.cdf(d2)

    @staticmethod
    def _bs_vega_np(sigma, moneyness, rate, tau):
        """Black-Scholes vega (S=1). Numpy."""
        sigma = np.maximum(sigma, 1e-8)
        tau = np.maximum(tau, 1e-8)
        sqrt_t = np.sqrt(tau)
        d1 = (np.log(1.0 / np.maximum(moneyness, 1e-8))
              + (rate + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_t)
        return sqrt_t * sp_norm.pdf(d1)

    @staticmethod
    def _bs_delta_np(sigma, moneyness, rate, tau):
        """Black-Scholes delta (S=1). Numpy."""
        sigma = np.maximum(sigma, 1e-8)
        tau = np.maximum(tau, 1e-8)
        sqrt_t = np.sqrt(tau)
        d1 = (np.log(1.0 / np.maximum(moneyness, 1e-8))
              + (rate + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_t)
        return sp_norm.cdf(d1)

    def _implied_vol(self, price, moneyness, rate, tau, max_iter=50, tol=1e-8):
        """Extract BS implied vol from Heston price via Newton's method."""
        sigma = np.full_like(price, 0.3)
        for _ in range(max_iter):
            bs_price = self._bs_call_np(sigma, moneyness, rate, tau)
            vega = self._bs_vega_np(sigma, moneyness, rate, tau)
            diff = bs_price - price
            mask = np.abs(vega) > 1e-12
            if not np.any(mask):
                break
            sigma[mask] -= diff[mask] / vega[mask]
            sigma = np.clip(sigma, 0.001, 5.0)
            if np.max(np.abs(diff[mask])) < tol:
                break
        return sigma

    def _to_tensor(self, arr):
        return torch.tensor(np.asarray(arr, dtype=np.float64), device=self.device)

    def get_price(self, df, model_type="heston"):
        """
        Compute Heston call prices (normalized: C/S where S=1, K=moneyness).

        Input: DataFrame with [kappa, theta, sigma, rho, v0, rate, tau, moneyness].
        Returns: numpy array of normalized call prices.
        """
        with torch.no_grad():
            prices = self._heston_call(
                self._to_tensor(df["moneyness"].values),
                self._to_tensor(df["tau"].values),
                self._to_tensor(df["rate"].values),
                self._to_tensor(df["kappa"].values),
                self._to_tensor(df["theta"].values),
                self._to_tensor(df["sigma"].values),
                self._to_tensor(df["rho"].values),
                self._to_tensor(df["v0"].values),
            )
        return prices.cpu().numpy()

    def get_iv(self, df, model_type="heston"):
        """Compute implied volatilities (Heston price -> BS IV inversion)."""
        prices = self.get_price(df, model_type)
        return self._implied_vol(
            prices,
            df["moneyness"].values.astype(np.float64),
            df["rate"].values.astype(np.float64),
            df["tau"].values.astype(np.float64),
        )

    def get_iv_delta(self, df, model_type="heston"):
        """Compute Black-Scholes delta at the implied vol."""
        ivs = self.get_iv(df, model_type)
        return self._bs_delta_np(
            ivs,
            df["moneyness"].values.astype(np.float64),
            df["rate"].values.astype(np.float64),
            df["tau"].values.astype(np.float64),
        )


class DeepSurrogateModel(BaseTFTModel):
    """
    Wraps the built-in Heston option pricing engine for fast IV/Greeks
    computation and tail risk monitoring.

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
        symbols = data["symbol"].unique() if "symbol" in data.columns else ["SPY"]

        for symbol in symbols:
            try:
                sym_data = (
                    data[data["symbol"] == symbol] if "symbol" in data.columns else data
                )
                preds = self._predict_symbol(symbol, sym_data)
                predictions.extend(preds)
            except Exception as e:
                logger.error("DeepSurrogate prediction failed for %s: %s", symbol, e)

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
            if len(self._param_history) > 252:
                self._param_history = self._param_history[-252:]

            tail_risk = self._compute_tail_risk_index(heston_params)
            self._tail_risk_history.append(tail_risk)

            risk_signal = -tail_risk
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

        iv_signal = self._compute_iv_signal(iv_surface, symbol)
        if iv_signal is not None:
            results.append(iv_signal)

        return results

    def _compute_iv_surface(self, df: pd.DataFrame) -> Optional[Dict[str, np.ndarray]]:
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
            deltas = self._surrogate.get_iv_delta(input_df, model_type=self._model_type)

            return {
                "iv": np.array(ivs) if not isinstance(ivs, np.ndarray) else ivs,
                "delta": (
                    np.array(deltas) if not isinstance(deltas, np.ndarray) else deltas
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
        param_cols = ["kappa", "theta", "sigma", "rho", "v0"]
        if all(col in df.columns for col in param_cols):
            row = df[param_cols].iloc[-1]
            return {col: float(row[col]) for col in param_cols}

        params = {}
        for col in param_cols:
            if col in df.columns:
                params[col] = float(df[col].median())
        return params if params else None

    def _compute_tail_risk_index(self, params: Dict[str, float]) -> float:
        """
        Build a tail risk index from Heston parameters.

        Captures vol-of-vol, correlation (leverage effect), variance skew,
        and mean reversion speed. Returns score in [0, 1] where 1 = max risk.
        """
        sigma = params.get("sigma", 0.3)
        rho = params.get("rho", -0.7)
        theta = params.get("theta", 0.04)
        v0 = params.get("v0", 0.04)
        kappa = params.get("kappa", 2.0)

        vol_of_vol_score = min(1.0, max(0.0, (sigma - 0.1) / 1.4))
        correlation_score = min(1.0, max(0.0, -rho))

        if theta > 0:
            variance_ratio = v0 / theta
            skew_score = min(1.0, max(0.0, (variance_ratio - 1.0) / 3.0))
        else:
            skew_score = 0.5

        jump_score = min(1.0, max(0.0, 1.0 - kappa / 10.0))

        tail_risk = (
            TAIL_RISK_VOL_OF_VOL_WEIGHT * vol_of_vol_score
            + TAIL_RISK_CORRELATION_WEIGHT * correlation_score
            + TAIL_RISK_SKEW_WEIGHT * skew_score
            + TAIL_RISK_JUMP_WEIGHT * jump_score
        )

        if len(self._tail_risk_history) >= 5:
            recent = np.mean(self._tail_risk_history[-5:])
            older = (
                np.mean(self._tail_risk_history[-21:-5])
                if len(self._tail_risk_history) >= 21
                else recent
            )
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
        iv_signal = -(atm_iv - 0.20) * 2.0
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

        Uses multi-start L-BFGS-B optimization with the built-in Heston
        pricer for fast evaluation. Best result across restarts is returned.
        Validates the Feller condition on the output.

        Args:
            market_data: DataFrame with columns: strike, spot, rate, tau,
                         market_price (observed option prices).
            n_restarts: Number of random starting points to try.

        Returns:
            Dict with calibrated parameters: kappa, theta, sigma, rho, v0,
            plus calibration_error, feller_satisfied, and success flag.
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

        # Normalize market prices by spot for comparison with C/S prices
        spot = df["spot"].values
        market_prices_norm = df["market_price"].values / spot

        if "moneyness" not in df.columns:
            df = df.copy()
            df["moneyness"] = df["strike"] / df["spot"]

        surrogate_cols = [
            "kappa", "theta", "sigma", "rho", "v0", "rate", "tau", "moneyness",
        ]

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
                if len(model_prices) != len(market_prices_norm):
                    return 1e10
                denom = np.maximum(np.abs(market_prices_norm), 0.001)
                return float(np.sum(((model_prices - market_prices_norm) / denom) ** 2))
            except Exception:
                return 1e10

        bounds = [
            (0.01, 20.0),   # kappa
            (0.001, 1.0),   # theta
            (0.01, 2.0),    # sigma (vol of vol)
            (-0.99, 0.0),   # rho
            (0.001, 1.0),   # v0
        ]

        starting_points = [[2.0, 0.04, 0.3, -0.7, 0.04]]
        rng = np.random.default_rng(42)
        for _ in range(n_restarts - 1):
            starting_points.append([
                rng.uniform(0.5, 10.0),
                rng.uniform(0.01, 0.15),
                rng.uniform(0.1, 1.0),
                rng.uniform(-0.95, -0.2),
                rng.uniform(0.01, 0.15),
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
        """Load the built-in Heston pricing engine (GPU-accelerated)."""
        try:
            self._surrogate = HestonEngine()
            self._is_loaded = True
            logger.info(
                "DeepSurrogate loaded (model_type=%s, device=%s)",
                self._model_type, self._surrogate.device,
            )
            return True
        except Exception as e:
            logger.error("Failed to load DeepSurrogate: %s", e)
            return False

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            asset_class=self.asset_class,
            version="2.0",
            symbols=self._symbols,
            model_path="built-in (PyTorch Heston)",
            is_loaded=self._is_loaded,
            metrics={
                "tail_risk_latest": (
                    self._tail_risk_history[-1] if self._tail_risk_history else 0.0
                ),
                "param_history_length": len(self._param_history),
            },
        )
