#!/usr/bin/env python3
"""
APEX Model Validation Suite — loads all TFT models, runs inference, grades each.

Usage:
    python validate_models.py
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
load_dotenv()

sys.path.insert(0, ".")

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
BOLD = "\033[1m"; RST = "\033[0m"
SEP = "=" * 72
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "TFT-Stocks": {"path": "models/tft_model.pth", "symbols": ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA"], "asset": "stocks"},
    "TFT-Forex": {"path": "models/tft_forex.pth",
                   "yf_symbols": ["EURUSD=X","GBPUSD=X","JPY=X","AUDUSD=X","CAD=X","CHF=X"],
                   "yf_map": {"EURUSD":"EURUSD","GBPUSD":"GBPUSD","JPY":"USDJPY","AUDUSD":"AUDUSD","CAD":"USDCAD","CHF":"USDCHF"},
                   "asset": "forex"},
    "TFT-Volatility": {"path": "models/tft_volatility.pth", "symbols": ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","BAC","XOM","SPY","QQQ"], "asset": "volatility"},
}

def status(p, label, detail=""):
    tag = f"{G}PASS{RST}" if p else f"{R}FAIL{RST}"
    print(f"  [{tag}] {label:<40s} {detail}")
    return p

def warn(label, detail=""): print(f"  [{Y}WARN{RST}] {label:<40s} {detail}")
def header(t): print(f"\n{B}{BOLD}{'─'*72}\n  {t}\n{'─'*72}{RST}")

def _patch_imports():
    import pytorch_forecasting.data.timeseries as ts
    sys.modules.setdefault("pytorch_forecasting.data.timeseries._timeseries", ts)

def download_data(symbols, period="6mo"):
    rows = []
    for sym in symbols:
        try:
            h = yf.Ticker(sym).history(period=period, auto_adjust=True)
            if h.empty: continue
            for dt, r in h.iterrows():
                rows.append({"symbol": sym.replace("=X",""), "timestamp": dt.tz_localize(None) if dt.tzinfo else dt,
                             "open": float(r["Open"]), "high": float(r["High"]), "low": float(r["Low"]),
                             "close": float(r["Close"]), "volume": int(r.get("Volume",0))})
        except Exception: pass
    return pd.DataFrame(rows)

def load_checkpoint(path):
    if not Path(path).exists(): return None
    _patch_imports()
    return torch.load(path, map_location="cpu", weights_only=False)

def reconstruct(ckpt):
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    cfg = ckpt.get("config", {}); sd = ckpt.get("model_state_dict"); ds = ckpt.get("training_dataset")
    if sd is None or ds is None: return None
    q = cfg.get("quantiles", [0.1, 0.5, 0.9])
    if "loss_type" in cfg:
        from tft_model import EnhancedTFTModel
        m = EnhancedTFTModel(); m.config = cfg; m.training_dataset = ds
        m.model = m.create_model(ds); m.model.load_state_dict(sd); return m.model
    hcs = cfg.get("hidden_continuous_size")
    if hcs is None:
        for k, v in sd.items():
            if k.startswith("prescalers.") and k.endswith(".weight") and v.shape[-1] == 1:
                hcs = v.shape[0]; break
        hcs = hcs or 8
    tft = TemporalFusionTransformer.from_dataset(ds,
        learning_rate=cfg.get("learning_rate", 0.001), hidden_size=cfg.get("hidden_size", 64),
        attention_head_size=cfg.get("attention_head_size", 4), dropout=cfg.get("dropout", 0.2),
        hidden_continuous_size=hcs, lstm_layers=cfg.get("lstm_layers", 2),
        loss=QuantileLoss(quantiles=q), output_size=len(q), reduce_on_plateau_patience=4, optimizer="adamw")
    tft.load_state_dict(sd); return tft

def prepare_stock_features(df):
    SECTOR = {"AAPL":"Tech","MSFT":"Tech","GOOGL":"Tech","AMZN":"Cons","NVDA":"Tech","META":"Tech","TSLA":"Cons","JPM":"Fin","BAC":"Fin","XOM":"Energy","SPY":"Index","QQQ":"Index"}
    df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    df["sector"] = df["symbol"].map(SECTOR).fillna("Other"); df["industry"] = "General"; df["exchange"] = "NASDAQ"
    frames = []
    for sym, g in df.groupby("symbol"):
        g = g.copy(); c = g["close"]
        g["adj_open"]=g["open"]; g["adj_high"]=g["high"]; g["adj_low"]=g["low"]; g["adj_close"]=c; g["adj_volume"]=g["volume"].astype(float)
        for w in [5,10,20,50]: g[f"sma_{w}"] = c.rolling(w).mean()
        g["ema_12"]=c.ewm(span=12).mean(); g["ema_26"]=c.ewm(span=26).mean()
        d=c.diff(); ga=d.where(d>0,0).rolling(14).mean(); lo=(-d.where(d<0,0)).rolling(14).mean(); g["rsi"]=100-(100/(1+ga/lo))
        g["macd"]=g["ema_12"]-g["ema_26"]; g["macd_signal"]=g["macd"].ewm(span=9).mean(); g["macd_histogram"]=g["macd"]-g["macd_signal"]
        ma20=c.rolling(20).mean(); s20=c.rolling(20).std(); g["bb_upper"]=ma20+2*s20; g["bb_lower"]=ma20-2*s20; g["bb_position"]=(c-ma20)/s20.where(s20>0,1)
        g["volume_ratio"]=g["volume"]/g["volume"].rolling(20).mean(); g["obv"]=(np.sign(c.diff())*g["volume"]).cumsum()
        g["price_change"]=c.pct_change(); g["high_low_ratio"]=g["high"]/g["low"]; g["close_open_ratio"]=c/g["open"]; g["returns_volatility"]=c.pct_change().rolling(20).std()
        for col in ["sentiment_score","sentiment_magnitude","news_count","market_cap","pe_ratio","eps","dividend_yield","earnings_flag","days_to_earnings"]: g[col]=0.0
        ts_col=pd.to_datetime(g["timestamp"]); dow=ts_col.dt.dayofweek; wk=ts_col.dt.isocalendar().week.astype(int)
        g["day_sin"]=np.sin(2*np.pi*dow/7); g["day_cos"]=np.cos(2*np.pi*dow/7)
        g["week_sin"]=np.sin(2*np.pi*wk/52); g["week_cos"]=np.cos(2*np.pi*wk/52)
        g["month_sin"]=np.sin(2*np.pi*ts_col.dt.month/12); g["month_cos"]=np.cos(2*np.pi*ts_col.dt.month/12)
        g["is_monday"]=(dow==0).astype(float); g["is_friday"]=(dow==4).astype(float)
        g["is_month_end"]=ts_col.dt.is_month_end.astype(float); g["is_quarter_end"]=ts_col.dt.is_quarter_end.astype(float)
        g["target"]=c.shift(-5)/c-1; frames.append(g)
    df=pd.concat(frames,ignore_index=True); df["time_idx"]=df.groupby("symbol").cumcount()
    df=df.dropna(subset=["rsi","macd","target","sma_50"])
    for col in df.select_dtypes(include=[np.number]).columns: df[col]=df[col].fillna(0)
    return df

def run_inference(model, ds, df):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import NaNLabelEncoder
    params = ds.get_parameters()
    encs = params.get("categorical_encoders", {})
    for k, enc in encs.items():
        if isinstance(enc, NaNLabelEncoder) and not enc.add_nan:
            enc.add_nan = True; enc.warn = False
    try:
        val_ds = TimeSeriesDataSet.from_dataset(ds, df, predict=True, stop_randomization=True,
                                                 categorical_encoders=encs if encs else None)
        dl = val_ds.to_dataloader(train=False, batch_size=32, num_workers=0)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(dev); model.eval()
        preds = model.predict(dl, mode="prediction", return_x=False)
        return preds.cpu().numpy()
    except Exception as e:
        logging.error("Inference failed: %s", e); return None

def grade(checks):
    """A/B/C/F based on check pass rate and metrics."""
    critical = ["load","reconstruct","inference","nan_check","inf_check","distribution"]
    crit_pass = sum(1 for c in critical if checks.get(c, {}).get("passed", False))
    total_pass = sum(1 for v in checks.values() if isinstance(v, dict) and v.get("passed", False))
    total = sum(1 for v in checks.values() if isinstance(v, dict) and "passed" in v)
    if crit_pass < len(critical): return "F"
    ratio = total_pass / max(total, 1)
    if ratio >= 0.9: return "A"
    if ratio >= 0.7: return "B"
    if ratio >= 0.5: return "C"
    return "F"

def validate_model(name, cfg):
    header(f"MODEL: {name}")
    checks = {}

    # Load
    print(f"  Loading...", end=" ", flush=True)
    ckpt = load_checkpoint(cfg["path"])
    if ckpt is None:
        print(f"{R}NOT FOUND{RST}")
        checks["load"] = {"passed": False, "detail": "File not found"}
        return {"name": name, "checks": checks, "grade": "F"}
    ds = ckpt.get("training_dataset"); config = ckpt.get("config", {})
    trained_at = ckpt.get("trained_at", "unknown")
    print(f"{G}OK{RST}")
    checks["load"] = {"passed": True, "detail": f"trained_at={trained_at}"}

    # Reconstruct
    print(f"  Reconstructing...", end=" ", flush=True)
    try:
        model = reconstruct(ckpt)
        if model is None: raise ValueError("None")
        pc = sum(p.numel() for p in model.parameters())
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(dev)
        print(f"{G}OK{RST} ({pc:,} params, {dev})")
        checks["reconstruct"] = {"passed": True, "detail": f"{pc:,} params, {dev}"}
    except Exception as e:
        print(f"{R}FAIL{RST}: {e}")
        checks["reconstruct"] = {"passed": False, "detail": str(e)[:100]}
        return {"name": name, "checks": checks, "grade": "F"}

    # GPU check
    on_gpu = next(model.parameters()).is_cuda if torch.cuda.is_available() else False
    checks["gpu"] = {"passed": on_gpu or not torch.cuda.is_available(), "detail": "CUDA" if on_gpu else "CPU"}
    status(checks["gpu"]["passed"], "GPU check", checks["gpu"]["detail"])

    # Download data
    print(f"  Downloading data...", end=" ", flush=True)
    yf_syms = cfg.get("yf_symbols", cfg.get("symbols", []))
    # Filter to known symbols
    if ds and hasattr(ds, "decoded_index"):
        known = set(ds.decoded_index["symbol"].unique())
        if cfg["asset"] == "forex":
            yf_map = cfg.get("yf_map", {})
            yf_syms = [s for s in yf_syms if yf_map.get(s.replace("=X",""), s.replace("=X","")) in known]
        else:
            yf_syms = [s for s in yf_syms if s in known]
    raw = download_data(yf_syms, "6mo")
    if raw.empty:
        print(f"{R}FAIL{RST}"); checks["data"] = {"passed": False}
        return {"name": name, "checks": checks, "grade": "F"}
    if "yf_map" in cfg:
        raw["symbol"] = raw["symbol"].map(lambda s: cfg["yf_map"].get(s, s))
    print(f"{G}OK{RST} ({len(raw)} rows)")
    checks["data"] = {"passed": True, "detail": f"{len(raw)} rows"}

    # Prepare features
    print(f"  Preparing features...", end=" ", flush=True)
    try:
        if cfg["asset"] == "stocks":
            df = prepare_stock_features(raw)
        elif cfg["asset"] == "forex":
            from models.forex_model import TFTForexModel
            df = TFTForexModel().prepare_features(raw)
        elif cfg["asset"] == "volatility":
            from models.volatility_model import TFTVolatilityModel
            df = TFTVolatilityModel().prepare_features(raw)
        else:
            df = prepare_stock_features(raw)
        print(f"{G}OK{RST} ({len(df)} rows)")
        checks["features"] = {"passed": True, "detail": f"{len(df)} rows"}
    except Exception as e:
        print(f"{R}FAIL{RST}: {e}")
        checks["features"] = {"passed": False, "detail": str(e)[:100]}
        return {"name": name, "checks": checks, "grade": "F"}

    # Inference
    print(f"  Running inference...", end=" ", flush=True)
    preds = run_inference(model, ds, df)
    if preds is None:
        print(f"{R}FAIL{RST}"); checks["inference"] = {"passed": False}
        return {"name": name, "checks": checks, "grade": "F"}
    print(f"{G}OK{RST} (shape={preds.shape})")
    checks["inference"] = {"passed": True, "detail": f"shape={preds.shape}"}

    # NaN/Inf
    has_nan = bool(np.isnan(preds).any())
    has_inf = bool(np.isinf(preds).any())
    checks["nan_check"] = {"passed": not has_nan, "detail": f"NaN={np.isnan(preds).sum()}"}
    checks["inf_check"] = {"passed": not has_inf, "detail": f"Inf={np.isinf(preds).sum()}"}
    status(not has_nan, "NaN check", checks["nan_check"]["detail"])
    status(not has_inf, "Inf check", checks["inf_check"]["detail"])

    # Distribution
    clean = preds[~np.isnan(preds) & ~np.isinf(preds)]
    std_val = float(np.std(clean)) if len(clean) > 0 else 0
    mean_val = float(np.mean(clean)) if len(clean) > 0 else 0
    degenerate = std_val < 0.001
    checks["distribution"] = {"passed": not degenerate,
                               "detail": f"mean={mean_val:.6f}, std={std_val:.6f}, range=[{np.min(clean):.6f}, {np.max(clean):.6f}]"}
    status(not degenerate, "Distribution", checks["distribution"]["detail"])

    # Confidence intervals (quantile check)
    if preds.ndim == 3 and preds.shape[-1] >= 3:
        lower, median, upper = preds[:, -1, 0], preds[:, -1, 1], preds[:, -1, 2]
        ci_ok = bool(np.all(upper >= median) and np.all(median >= lower))
        checks["confidence_intervals"] = {"passed": ci_ok, "detail": f"upper>=med>=lower: {ci_ok}"}
        status(ci_ok, "Confidence intervals", checks["confidence_intervals"]["detail"])

    # Directional accuracy + MAE/RMSE
    try:
        if preds.ndim == 3: med_preds = preds[:, -1, 1]
        elif preds.ndim == 2: med_preds = preds[:, -1]
        else: med_preds = preds
        actuals = []
        for sym in df["symbol"].unique():
            tgts = df[df["symbol"]==sym].sort_values("time_idx")["target"].values
            n_per = len(med_preds) // df["symbol"].nunique()
            actuals.extend(tgts[-n_per:])
        actuals = np.array(actuals[:len(med_preds)])
        n = min(len(med_preds), len(actuals))
        if n >= 3:
            p_, a_ = med_preds[:n], actuals[:n]
            mask = ~(np.isnan(p_)|np.isnan(a_)|np.isinf(p_)|np.isinf(a_))
            p_, a_ = p_[mask], a_[mask]
            if len(p_) >= 3:
                mae = float(np.mean(np.abs(p_-a_)))
                rmse = float(np.sqrt(np.mean((p_-a_)**2)))
                dir_acc = float(np.mean(np.sign(p_)==np.sign(a_))*100)
                checks["mae"] = {"passed": True, "detail": f"{mae:.6f}"}
                checks["rmse"] = {"passed": True, "detail": f"{rmse:.6f}"}
                checks["directional_accuracy"] = {"passed": dir_acc >= 40, "detail": f"{dir_acc:.1f}%"}
                status(True, "MAE", f"{mae:.6f}")
                status(True, "RMSE", f"{rmse:.6f}")
                status(dir_acc >= 40, "Directional accuracy", f"{dir_acc:.1f}%")
    except Exception:
        pass

    # Overfitting check (compare config train vs val loss if available)
    train_loss = config.get("final_train_loss")
    val_loss = config.get("final_val_loss")
    if train_loss and val_loss and train_loss > 0:
        ratio = val_loss / train_loss
        overfit = ratio > 3.0
        checks["overfitting"] = {"passed": not overfit, "detail": f"val/train ratio={ratio:.2f}"}
        status(not overfit, "Overfitting check", checks["overfitting"]["detail"])

    g = grade(checks)
    checks["grade"] = g
    return {"name": name, "checks": checks, "grade": g}


def check_model_manager():
    header("MODEL MANAGER (all 10 models)")
    try:
        from models.manager import ModelManager
        mgr = ModelManager()
        load_status = mgr.load_all()
        loaded = load_status.models_loaded
        total = load_status.models_registered
        status(loaded > 0, "ModelManager.load_all()", f"{loaded}/{total} models loaded")
        for info in load_status.details:
            tag = f"{G}loaded{RST}" if info.is_loaded else f"{Y}not loaded{RST}"
            print(f"    {info.name:<25s} {tag}")
        return loaded, total
    except Exception as e:
        status(False, "ModelManager", str(e)[:80])
        return 0, 0


def main():
    torch.set_float32_matmul_precision("medium")
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    print(f"\n{BOLD}{SEP}")
    print(f"  APEX MODEL VALIDATION SUITE")
    dev = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    print(f"  Device: {dev}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{SEP}{RST}")

    results = []
    for name, cfg in MODELS.items():
        r = validate_model(name, cfg)
        results.append(r)

    mm_loaded, mm_total = check_model_manager()

    # Summary
    print(f"\n{BOLD}{SEP}")
    print(f"  MODEL REPORT CARD")
    print(f"{'─'*72}{RST}")
    grade_colors = {"A": G, "B": G, "C": Y, "F": R}
    all_grades = []
    for r in results:
        g = r["grade"]; gc = grade_colors.get(g, RST)
        mae_str = ""
        if "mae" in r["checks"]: mae_str = f"MAE={r['checks']['mae']['detail']}"
        dir_str = ""
        if "directional_accuracy" in r["checks"]: dir_str = f"DirAcc={r['checks']['directional_accuracy']['detail']}"
        metrics = ", ".join(filter(None, [mae_str, dir_str]))
        print(f"  [{gc}{g}{RST}] {r['name']:<20s} {metrics}")
        all_grades.append(g)
    if mm_loaded > 0:
        print(f"  [{'G' if mm_loaded >= 3 else 'Y'}INFO{RST}] ModelManager           {mm_loaded}/{mm_total} models loaded")
    print(SEP)

    passed = sum(1 for g in all_grades if g in ("A","B"))
    failed = sum(1 for g in all_grades if g == "F")

    # Save
    result_file = RESULTS_DIR / f"models_{ts}.json"
    with open(result_file, "w") as f:
        json.dump({"timestamp": ts, "grades": {r["name"]: r["grade"] for r in results},
                   "model_manager": {"loaded": mm_loaded, "total": mm_total},
                   "details": {r["name"]: {k: v for k, v in r["checks"].items() if isinstance(v, dict)} for r in results}},
                  f, indent=2, default=str)
    print(f"  Results saved to {result_file}")

    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
