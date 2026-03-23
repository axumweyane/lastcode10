"""
TDGF — Time Deep Gradient Flow for American option pricing.

Built-in PyTorch implementation of the Deep Galerkin Method (DGM) neural
PDE solver. Handles free-boundary PDEs for American options with early
exercise. No external dependencies required.

Architecture: DGM-style layers with GRU-like gating, softplus output
for no-arbitrage constraint. Supports Black-Scholes (1D) and Heston (2D).

GPU-accelerated training and inference via PyTorch.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.base import BaseTFTModel, ModelInfo, ModelPrediction

logger = logging.getLogger(__name__)

TDGF_HIDDEN_LAYERS = int(os.getenv("TDGF_HIDDEN_LAYERS", "3"))
TDGF_HIDDEN_UNITS = int(os.getenv("TDGF_HIDDEN_UNITS", "50"))
TDGF_LEARNING_RATE = float(os.getenv("TDGF_LEARNING_RATE", "0.001"))
TDGF_MAX_EPOCHS = int(os.getenv("TDGF_MAX_EPOCHS", "5000"))
TDGF_PDE_MODEL = os.getenv("TDGF_PDE_MODEL", "heston")


# ── DGM Neural Network Architecture ─────────────────────────────────


class DGMLayer(nn.Module):
    """
    Single DGM (Deep Galerkin Method) layer with GRU-like gating.

    Implements:
        Z = sigmoid(Uz*x + Wz*s + bz)         (update gate)
        G = sigmoid(Ug*x + Wg*(s*Z) + bg)     (output gate)
        R = sigmoid(Ur*x + Wr*s + br)         (reset gate)
        H = tanh(Uh*x + Wh*(s*R) + bh)        (candidate)
        out = (1 - G) * H + G * s              (GRU-style update)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.Uz = nn.Linear(input_dim, hidden_dim)
        self.Wz = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Ug = nn.Linear(input_dim, hidden_dim)
        self.Wg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Ur = nn.Linear(input_dim, hidden_dim)
        self.Wr = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Uh = nn.Linear(input_dim, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.Uz(x) + self.Wz(s))
        g = torch.sigmoid(self.Ug(x) + self.Wg(s * z))
        r = torch.sigmoid(self.Ur(x) + self.Wr(s))
        h = torch.tanh(self.Uh(x) + self.Wh(s * r))
        return (1.0 - g) * h + g * s


class DGMNet(nn.Module):
    """
    Deep Galerkin Method network for solving PDEs.

    Input: (s, tau) for BS or (s, w, tau) for Heston
    Output: normalized option price v = V/K (non-negative via softplus)
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.input_dim = input_dim
        self.initial = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            DGMLayer(input_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.tanh(self.initial(x))
        for layer in self.layers:
            s = layer(x, s)
        return torch.nn.functional.softplus(self.output(s))


# ── American Option PDE Solver ───────────────────────────────────────


class AmericanOptionSolver:
    """
    Solves American option pricing PDEs using the DGM neural network.

    Supports Black-Scholes (1D) and Heston (2D) models. Uses the penalty
    method to enforce the early exercise constraint.

    The PDE is solved in normalized coordinates (s = S/K, v = V/K) so the
    solution is independent of strike K.
    """

    def __init__(self, hidden_layers: int = 3, hidden_units: int = 50):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.net = None
        self._pde_params: Dict[str, float] = {}
        self._trained_model_type: Optional[str] = None

    def train(self, model_type, params, n_epochs, learning_rate,
              hidden_layers=None, hidden_units=None):
        """
        Train the DGM network to solve the option pricing PDE.

        The network learns v(s, tau) = V(S, tau)/K in normalized coordinates.
        Uses penalty method for American early exercise constraint.
        """
        hl = hidden_layers or self.hidden_layers
        hu = hidden_units or self.hidden_units

        if model_type == "black_scholes":
            input_dim = 2  # (s, tau)
        elif model_type in ("heston", "lifted_heston"):
            input_dim = 3  # (s, w, tau)
        else:
            input_dim = 2

        # Rebuild network if input dimension changed
        if self.net is None or self.net.input_dim != input_dim:
            self.net = DGMNet(input_dim, hu, hl).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, n_epochs // 3), gamma=0.5)

        # Extract scalar PDE parameters
        r = float(np.mean(params["rate"]))
        T = float(np.max(params["tau"])) if np.max(params["tau"]) > 0 else 1.0
        T = max(T, 0.1)

        self._pde_params = {"r": r, "T": T, "model_type": model_type}

        if model_type == "black_scholes":
            sigma = float(np.mean(params["sigma"]))
            self._pde_params["sigma"] = sigma
            result = self._train_bs(r, T, sigma, n_epochs, optimizer, scheduler)
        elif model_type in ("heston", "lifted_heston"):
            kappa = float(np.mean(params["kappa"]))
            theta = float(np.mean(params["theta"]))
            sigma_h = float(np.mean(params["sigma"]))
            rho = float(np.mean(params["rho"]))
            v0 = float(np.mean(params["v0"]))
            self._pde_params.update({
                "kappa": kappa, "theta": theta, "sigma_h": sigma_h,
                "rho": rho, "v0": v0,
            })
            result = self._train_heston(
                r, T, kappa, theta, sigma_h, rho, v0, n_epochs, optimizer, scheduler,
            )
        else:
            logger.warning("Unknown model type %s, defaulting to BS", model_type)
            sigma = float(np.mean(params.get("sigma", [0.2])))
            self._pde_params["sigma"] = sigma
            result = self._train_bs(r, T, sigma, n_epochs, optimizer, scheduler)

        self._trained_model_type = model_type
        return result

    def _train_bs(self, r, T, sigma, n_epochs, optimizer, scheduler,
                  n_interior=500, n_ic=100, n_bc=100):
        """Train on Black-Scholes PDE for American put."""
        s_max = 3.0  # normalized domain [0, 3]

        final_loss = 0.0
        final_pde = 0.0
        final_ic = 0.0

        for epoch in range(n_epochs):
            # Interior points
            s = (torch.rand(n_interior, 1, device=self.device) * s_max).detach().requires_grad_(True)
            tau = (torch.rand(n_interior, 1, device=self.device) * T).detach().requires_grad_(True)

            x = torch.cat([s, tau], dim=1)
            v = self.net(x)

            grads = torch.autograd.grad(v.sum(), [s, tau], create_graph=True)
            dv_ds = grads[0]
            dv_dtau = grads[1]
            d2v_ds2 = torch.autograd.grad(dv_ds.sum(), s, create_graph=True)[0]

            # PDE: dv/dtau = 0.5*sigma^2*s^2*d2v/ds2 + r*s*dv/ds - r*v
            Lv = 0.5 * sigma ** 2 * s ** 2 * d2v_ds2 + r * s * dv_ds - r * v
            pde_res = dv_dtau - Lv
            L_pde = torch.mean(pde_res ** 2)

            # Initial condition at tau=0 (expiry): v = max(1-s, 0)
            s_ic = (torch.rand(n_ic, 1, device=self.device) * s_max).detach()
            tau_ic = torch.zeros(n_ic, 1, device=self.device)
            x_ic = torch.cat([s_ic, tau_ic], dim=1)
            v_ic = self.net(x_ic)
            payoff_ic = torch.clamp(1.0 - s_ic, min=0.0)
            L_ic = torch.mean((v_ic - payoff_ic) ** 2)

            # Boundary: s=0 -> v = exp(-r*tau)
            tau_bc = (torch.rand(n_bc, 1, device=self.device) * T).detach()
            s_bc0 = torch.zeros(n_bc, 1, device=self.device)
            x_bc0 = torch.cat([s_bc0, tau_bc], dim=1)
            v_bc0 = self.net(x_bc0)
            L_bc0 = torch.mean((v_bc0 - torch.exp(-r * tau_bc)) ** 2)

            # Boundary: s=s_max -> v ~ 0
            s_bcmax = torch.full((n_bc, 1), s_max, device=self.device)
            x_bcmax = torch.cat([s_bcmax, tau_bc], dim=1)
            v_bcmax = self.net(x_bcmax)
            L_bcmax = torch.mean(v_bcmax ** 2)

            # American constraint: v >= max(1-s, 0)
            payoff = torch.clamp(1.0 - s, min=0.0)
            L_am = torch.mean(torch.clamp(payoff - v, min=0.0) ** 2)

            loss = L_pde + 10.0 * L_ic + L_bc0 + L_bcmax + 100.0 * L_am

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            final_loss = float(loss.detach())
            final_pde = float(L_pde.detach())
            final_ic = float(L_ic.detach())

        return {
            "loss": final_loss,
            "epochs": n_epochs,
            "pde_residual": final_pde,
            "boundary_error": final_ic,
        }

    def _train_heston(self, r, T, kappa, theta, sigma_h, rho, v0,
                      n_epochs, optimizer, scheduler,
                      n_interior=500, n_ic=100, n_bc=80):
        """Train on Heston PDE for American put."""
        s_max = 3.0
        w_max = 1.0  # variance domain [0.001, 1.0]

        final_loss = 0.0
        final_pde = 0.0
        final_ic = 0.0

        for epoch in range(n_epochs):
            # Interior points: (s, w, tau)
            s = (torch.rand(n_interior, 1, device=self.device) * s_max).detach().requires_grad_(True)
            w = (torch.rand(n_interior, 1, device=self.device) * w_max + 0.001).detach().requires_grad_(True)
            tau = (torch.rand(n_interior, 1, device=self.device) * T).detach().requires_grad_(True)

            x = torch.cat([s, w, tau], dim=1)
            v = self.net(x)

            grads = torch.autograd.grad(v.sum(), [s, w, tau], create_graph=True)
            dv_ds = grads[0]
            dv_dw = grads[1]
            dv_dtau = grads[2]

            d2v_ds2 = torch.autograd.grad(dv_ds.sum(), s, create_graph=True)[0]
            d2v_dw2 = torch.autograd.grad(dv_dw.sum(), w, create_graph=True)[0]
            d2v_dsdw = torch.autograd.grad(dv_ds.sum(), w, create_graph=True)[0]

            # Heston PDE
            Lv = (0.5 * w * s ** 2 * d2v_ds2
                  + rho * sigma_h * w * s * d2v_dsdw
                  + 0.5 * sigma_h ** 2 * w * d2v_dw2
                  + r * s * dv_ds
                  + kappa * (theta - w) * dv_dw
                  - r * v)
            pde_res = dv_dtau - Lv
            L_pde = torch.mean(pde_res ** 2)

            # Initial condition at tau=0
            s_ic = (torch.rand(n_ic, 1, device=self.device) * s_max).detach()
            w_ic = (torch.rand(n_ic, 1, device=self.device) * w_max + 0.001).detach()
            tau_ic = torch.zeros(n_ic, 1, device=self.device)
            x_ic = torch.cat([s_ic, w_ic, tau_ic], dim=1)
            v_ic = self.net(x_ic)
            payoff_ic = torch.clamp(1.0 - s_ic, min=0.0)
            L_ic = torch.mean((v_ic - payoff_ic) ** 2)

            # Boundary conditions
            tau_bc = (torch.rand(n_bc, 1, device=self.device) * T).detach()
            w_bc = (torch.rand(n_bc, 1, device=self.device) * w_max + 0.001).detach()

            # s=0: v = exp(-r*tau)
            s_bc0 = torch.zeros(n_bc, 1, device=self.device)
            x_bc0 = torch.cat([s_bc0, w_bc, tau_bc], dim=1)
            v_bc0 = self.net(x_bc0)
            L_bc0 = torch.mean((v_bc0 - torch.exp(-r * tau_bc)) ** 2)

            # s=s_max: v ~ 0
            s_bcmax = torch.full((n_bc, 1), s_max, device=self.device)
            x_bcmax = torch.cat([s_bcmax, w_bc, tau_bc], dim=1)
            v_bcmax = self.net(x_bcmax)
            L_bcmax = torch.mean(v_bcmax ** 2)

            # American constraint
            payoff = torch.clamp(1.0 - s, min=0.0)
            L_am = torch.mean(torch.clamp(payoff - v, min=0.0) ** 2)

            loss = L_pde + 10.0 * L_ic + L_bc0 + L_bcmax + 100.0 * L_am

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            final_loss = float(loss.detach())
            final_pde = float(L_pde.detach())
            final_ic = float(L_ic.detach())

        return {
            "loss": final_loss,
            "epochs": n_epochs,
            "pde_residual": final_pde,
            "boundary_error": final_ic,
        }

    def price(self, model_type, params):
        """
        Price options using the trained DGM network.

        Evaluates v(s, tau) = V/K in normalized coordinates, then scales
        back: V = K * v(S/K, tau).
        """
        if self.net is None:
            return None

        S = np.asarray(params["spot"], dtype=np.float32).flatten()
        K = np.asarray(params["strike"], dtype=np.float32).flatten()
        tau = np.asarray(params["tau"], dtype=np.float32).flatten()
        s = S / np.maximum(K, 1e-8)

        if model_type in ("heston", "lifted_heston") and "v0" in params:
            w = np.asarray(params["v0"], dtype=np.float32).flatten()
            # Ensure same length (broadcast if scalar)
            if len(w) == 1 and len(s) > 1:
                w = np.full_like(s, w[0])
            x_np = np.column_stack([s, w, tau])
        else:
            x_np = np.column_stack([s, tau])

        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v = self.net(x).cpu().numpy().flatten()

        return K * v

    def greeks(self, model_type, params):
        """Compute option Greeks via automatic differentiation."""
        if self.net is None:
            return None

        S = np.asarray(params["spot"], dtype=np.float32).flatten()
        K = np.asarray(params["strike"], dtype=np.float32).flatten()
        tau_np = np.asarray(params["tau"], dtype=np.float32).flatten()
        s_np = S / np.maximum(K, 1e-8)

        s = torch.tensor(s_np, dtype=torch.float32, device=self.device).requires_grad_(True)
        t = torch.tensor(tau_np, dtype=torch.float32, device=self.device).requires_grad_(True)

        if model_type in ("heston", "lifted_heston") and "v0" in params:
            w_np = np.asarray(params["v0"], dtype=np.float32).flatten()
            if len(w_np) == 1 and len(s_np) > 1:
                w_np = np.full_like(s_np, w_np[0])
            w = torch.tensor(w_np, dtype=torch.float32, device=self.device).requires_grad_(True)
            x = torch.stack([s, w, t], dim=1)
        else:
            w = None
            x = torch.stack([s, t], dim=1)

        v = self.net(x).squeeze(-1)  # (N,)
        K_t = torch.tensor(K, dtype=torch.float32, device=self.device)

        # Delta = dV/dS = dv/ds (since V=K*v, s=S/K, chain rule cancels K)
        v_sum = v.sum()
        inputs = [s, t] if w is None else [s, w, t]
        grads = torch.autograd.grad(v_sum, inputs, create_graph=True)
        dv_ds = grads[0]
        dv_dt = grads[-1]  # last is always tau

        d2v_ds2 = torch.autograd.grad(dv_ds.sum(), s, create_graph=False)[0]

        result = {
            "delta": dv_ds.detach().cpu().numpy(),
            "gamma": (d2v_ds2 / K_t).detach().cpu().numpy(),
            "theta": -(K_t * dv_dt).detach().cpu().numpy(),
        }

        if w is not None:
            dv_dw = grads[1]
            result["vega"] = (K_t * dv_dw).detach().cpu().numpy()

        return result

    def state_dict(self):
        if self.net is None:
            return {}
        return self.net.state_dict()

    def load_state_dict(self, state_dict):
        if self.net is not None and state_dict:
            self.net.load_state_dict(state_dict)


# ── TDGF Model (BaseTFTModel interface) ─────────────────────────────


class TDGFModel(BaseTFTModel):
    """
    Time Deep Gradient Flow for American option pricing.

    Uses the DGM (Deep Galerkin Method) neural network to solve option
    pricing PDEs. Supports Black-Scholes (1D) and Heston (2D).
    Softplus output enforces no-arbitrage (non-negative prices).
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

        Expected columns for Heston: spot, strike, rate, tau, kappa, theta, sigma, rho, v0
        For Black-Scholes: spot, strike, rate, tau, sigma
        """
        df = raw_data.copy()
        df.columns = [c.lower() for c in df.columns]

        if (
            "moneyness" not in df.columns
            and "spot" in df.columns
            and "strike" in df.columns
        ):
            df["moneyness"] = df["spot"] / df["strike"]

        if "tau" not in df.columns and "expiry" in df.columns:
            today = pd.Timestamp.now()
            df["tau"] = (pd.to_datetime(df["expiry"]) - today).dt.days / 365.0
            df["tau"] = df["tau"].clip(lower=1 / 365.0)

        return df

    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Train TDGF network on specific option parameters.

        The solver learns the PDE solution for the given model parameters.
        """
        if not self._is_loaded:
            logger.error("TDGF solver not loaded. Call load() first.")
            return {"error": "not_loaded"}

        epochs = kwargs.get("epochs", TDGF_MAX_EPOCHS)
        lr = kwargs.get("learning_rate", TDGF_LEARNING_RATE)

        df = self.prepare_features(data)

        try:
            params = self._extract_pde_params(df)
            if params is None:
                return {"error": "invalid_params"}

            logger.info(
                "Training TDGF (%s model, %d epochs, lr=%.6f)",
                self._pde_model, epochs, lr,
            )

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
        """Price American options and compute Greeks using the trained DGM network."""
        if not self._is_loaded:
            logger.debug("TDGF not loaded, returning empty predictions")
            return []

        if not self._is_trained:
            logger.debug("TDGF not trained, returning empty predictions")
            return []

        predictions = []
        symbols = data["symbol"].unique() if "symbol" in data.columns else ["OPT"]

        for symbol in symbols:
            try:
                sym_data = (
                    data[data["symbol"] == symbol] if "symbol" in data.columns else data
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
            prices = self._solver.price(
                model_type=self._pde_model, params=params,
            )

            greeks = None
            if hasattr(self._solver, "greeks"):
                greeks = self._solver.greeks(
                    model_type=self._pde_model, params=params,
                )

            if prices is None:
                return results

            if not isinstance(prices, np.ndarray):
                prices = np.array(prices).flatten()

            for i, row in df.iterrows():
                if i >= len(prices):
                    break

                tdgf_price = float(prices[i]) if i < len(prices) else 0.0
                market_price = float(row.get("market_price", tdgf_price))

                if market_price > 0 and tdgf_price > 0:
                    mispricing = (tdgf_price - market_price) / market_price
                else:
                    mispricing = 0.0

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

                if greeks is not None and isinstance(greeks, dict):
                    for greek_name, greek_vals in greeks.items():
                        if isinstance(greek_vals, (list, np.ndarray)):
                            if i < len(greek_vals):
                                metadata[f"greek_{greek_name}"] = float(greek_vals[i])
                        else:
                            metadata[f"greek_{greek_name}"] = float(greek_vals)

                results.append(
                    ModelPrediction(
                        symbol=symbol,
                        predicted_value=mispricing,
                        lower_bound=mispricing - 0.05,
                        upper_bound=mispricing + 0.05,
                        confidence=confidence,
                        horizon_days=max(1, int(row.get("tau", 0.1) * 365)),
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
                "spot", "strike", "rate", "tau",
                "kappa", "theta", "sigma", "rho", "v0",
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

        params["option_type"] = df.get("option_type", pd.Series(["put"])).iloc[0]
        params["exercise_type"] = "american"

        return params

    def save(self, path: str) -> None:
        """Save trained TDGF network."""
        if not self._is_trained:
            logger.info("TDGF not trained yet; nothing to save")
            return

        try:
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
        """
        Load the built-in DGM PDE solver.

        Creates the AmericanOptionSolver and performs a quick self-training
        on a default American put problem for out-of-box operation. If a
        checkpoint file exists at `path`, loads pre-trained weights instead.
        """
        try:
            self._solver = AmericanOptionSolver(
                hidden_layers=TDGF_HIDDEN_LAYERS,
                hidden_units=TDGF_HIDDEN_UNITS,
            )
            self._is_loaded = True

            # Try to load pre-trained weights
            if path and Path(path).exists():
                try:
                    checkpoint = torch.load(
                        path, map_location=self._solver.device, weights_only=False,
                    )
                    if "state_dict" in checkpoint:
                        self._solver.load_state_dict(checkpoint["state_dict"])
                    self._pde_model = checkpoint.get("pde_model", self._pde_model)
                    self._trained_at = checkpoint.get("trained_at")
                    self._training_metrics = checkpoint.get("training_metrics", {})
                    self._is_trained = True
                    logger.info("TDGF loaded pre-trained weights from %s", path)
                except Exception as e:
                    logger.info("Could not load TDGF weights from %s: %s", path, e)
                    self._quick_train()
            else:
                self._quick_train()

            logger.info(
                "TDGF solver loaded (model=%s, device=%s)",
                self._pde_model, self._solver.device,
            )
            return True
        except Exception as e:
            logger.error("Failed to load TDGF: %s", e)
            return False

    def _quick_train(self):
        """Quick self-training on default American put for out-of-box operation."""
        logger.info("TDGF: quick self-training on default American put (BS, 1000 epochs)...")
        default_params = {
            "spot": np.array([1.0]),
            "strike": np.array([1.0]),
            "rate": np.array([0.05]),
            "tau": np.array([2.0]),
            "sigma": np.array([0.2]),
        }
        metrics = self._solver.train(
            model_type="black_scholes",
            params=default_params,
            n_epochs=1000,
            learning_rate=1e-3,
            hidden_layers=TDGF_HIDDEN_LAYERS,
            hidden_units=TDGF_HIDDEN_UNITS,
        )
        self._is_trained = True
        self._trained_at = datetime.now()
        self._training_metrics = metrics
        logger.info("TDGF quick training complete: loss=%.6f", metrics.get("loss", 0))

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            asset_class=self.asset_class,
            version="2.0",
            trained_at=self._trained_at,
            symbols=self._symbols,
            model_path="built-in (DGM PDE solver)",
            is_loaded=self._is_loaded,
            metrics=self._training_metrics,
        )
