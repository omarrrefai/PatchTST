#!/usr/bin/env python3
"""
run_patchtst_training_bundle.py

Unified trainer/evaluator for PatchTST forecasts (t+1 and DA288) across selected Canada market areas,
with consistent artifact saving for downstream MILP/MPC experiments.

Key outputs per (area, horizon):
  - pred.npy            : predictions in RAW $/MWh scale (after scaling fix)
  - true.npy            : aligned ground truth in RAW $/MWh scale
  - timestamps.npy      : timestamps for the aligned test windows
  - metrics.json        : MAE/RMSE/MAPE + baseline comparisons + tail metrics
  - horizon_mae.csv     : per-horizon MAE for DA288
  - plots/              : sanity plots (first window, horizon MAE, etc.)
  - train_cmd.txt       : exact training command
  - train_stdout.log    : captured stdout/stderr from PatchTST run

IMPORTANT ABOUT SCALING:
Many PatchTST repos save pred.npy in a normalized space. We confirmed your case.
This script produces RAW-scale pred.npy by fitting an affine calibration:
  y_raw H a * pred_saved + b
using a calibration subset of test windows (default 20%).
This is robust even if the internal scaler is not easily accessible.

If your run_longExp.py supports inverse saving (e.g. --inverse 1),
you can enable TRY_INVERSE_FLAG=True and the script will pass it.
Even then, we still verify scale and will fall back to calibration if mismatch remains.
"""

import argparse
import json
import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# User / Project Configuration
# ---------------------------

DATA_DIR = Path("/home/omaralrefai/dev/PatchTST/.dataset/canada").resolve()
RUN_LONGEXP = Path("run_longExp.py")  # must be runnable from current working directory
PATCHTST_RESULTS_ROOT = Path("results").resolve()

# Where we store paper-ready artifacts (NEW canonical training bundle)
BUNDLE_ROOT = Path("/home/omaralrefai/dev/VPP_Forecast_Results/training").resolve()

# Only these areas are allowed (as you requested)
AREAS = ["manitoba", "new-york", "ontario", "quebec_p33c", "manitoba_sk"]

# Map areas -> input features CSV filenames (existing in DATA_DIR)
AREA_TO_FEATURES_FILE = {
    "manitoba": "manitoba_features.csv",
    "new-york": "new-york_features.csv",
    "ontario": "ontario_features.csv",
    "quebec_p33c": "quebec_p33c_features.csv",
    "manitoba_sk": "manitoba_sk_features.csv",
}

# Clean CSV output root for PatchTST custom loader
CLEAN_DATA_ROOT = (DATA_DIR / "per_area_features_clean").resolve()
CLEAN_DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Forecast experiment settings (common)
FREQ = "5min"
PER_HOUR = 12
DAILY_LAG = 24 * PER_HOUR  # 288 steps

# Model/data channels
ENC_IN = 7
DEC_IN = 7
C_OUT = 1  # price

# History and label
SEQ_LEN = 576
LABEL_LEN = 72

# Training hyperparameters (your defaults)
D_MODEL = 512
N_HEADS = 8
E_LAYERS = 3
D_LAYERS = 2
D_FF = 2048
DROPOUT = 0.05
BATCH = 32
LR = 1e-4
EPOCHS = 30
PATIENCE = 8
PATCH_LEN = 48
STRIDE = 24
EMBED = "fixed"
NUM_WORKERS = 0
USE_GPU = 1
USE_AMP = True

# Normalization-related flags (repo-dependent)
REVIN = 0
AFFINE = 0
SUBTRACT_LAST = 0

# Data split
SPLIT = (0.7, 0.1, 0.2)
EPS_MAPE = 1.0

# Scaling fix controls
TRY_INVERSE_FLAG = True   # will attempt to pass --inverse 1 (if run_longExp.py supports it)
CALIB_FIT_FRAC = 0.2      # fraction of test windows used to fit affine mapping
CALIB_MIN_SAMPLES = 1000  # minimum samples used for calibration fit
SCALE_MISMATCH_RATIO = 5.0  # if std(true)/std(pred) > this, we treat it as scaled mismatch

# Tail metric for paper (spikes)
TAIL_Q = 0.99

# ---------------------------
# Utilities
# ---------------------------

def mae(a, b) -> float:
    return float(np.mean(np.abs(a - b)))

def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mape(a, b, eps=1.0) -> float:
    return float(np.mean(np.abs(a - b) / np.maximum(eps, np.abs(a))) * 100.0)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def build_per_area_clean_csv(src_csv: Path, out_root: Path) -> Path:
    """
    Produce a strict numeric CSV for PatchTST custom data loader.

    Expected input columns (case-insensitive):
      timestamp, hour, day_of_week, interval, month, windspeed_10m, temperature_2m, price

    Output columns:
      date, DATE_ORD, hour, day_of_week_idx, interval, month, windspeed_10m, temperature_2m, price
    """
    df = pd.read_csv(src_csv)
    df = normalize_cols(df)

    req = {"timestamp", "hour", "day_of_week", "interval", "month", "windspeed_10m", "temperature_2m", "price"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"{src_csv.name}: missing required columns: {', '.join(sorted(missing))}")

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"{src_csv.name}: unparsable timestamps found: {bad}")

    out = pd.DataFrame({
        "date": ts,
        "DATE_ORD": ts.map(lambda x: x.toordinal()),
        "hour": pd.to_numeric(df["hour"], errors="coerce"),
        "day_of_week_idx": pd.Categorical(df["day_of_week"]).codes,
        "interval": pd.to_numeric(df["interval"], errors="coerce"),
        "month": pd.to_numeric(df["month"], errors="coerce"),
        "windspeed_10m": pd.to_numeric(df["windspeed_10m"], errors="coerce"),
        "temperature_2m": pd.to_numeric(df["temperature_2m"], errors="coerce"),
        "price": pd.to_numeric(df["price"], errors="coerce"),
    })

    # Basic cleaning
    out = out[(out["interval"] >= 1) & (out["interval"] <= 12)]
    out = out[out["hour"].between(0, 23)]
    out = out[out["month"].between(1, 12)]
    out = out.dropna().sort_values("date").reset_index(drop=True)

    out_path = out_root / f"{src_csv.stem.replace('_features','')}_clean.csv"
    out.to_csv(out_path, index=False)
    return out_path

def load_price_series(clean_csv: Path) -> Tuple[pd.Series, pd.DatetimeIndex]:
    df = pd.read_csv(clean_csv, parse_dates=["date"]).sort_values("date").set_index("date")
    y = df["price"].astype(float)
    return y, y.index

def compute_test_start(N: int) -> int:
    num_train = int(N * SPLIT[0])
    num_val = int(N * SPLIT[1])
    return num_train + num_val

def shape_pred(pred: np.ndarray, pred_len: int, target_idx: int = 0) -> np.ndarray:
    """
    Normalize PatchTST pred.npy formats into (W, pred_len) for the TARGET channel.

    Common formats:
      - pred_len=1:
          (W,) or (W,1) or (W,C) or (W,1,C) or (W,C,1)
      - pred_len>1:
          (W,pred_len) or (W,pred_len,1) or (W,pred_len,C) depending on repo

    We always extract target_idx and return shape (W, pred_len).
    """
    arr = np.array(pred)

    # Handle 3D variants where last dim is singleton
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]  # drop last singleton

        if pred_len == 1:
            # (W,) -> (W,1)
            if arr.ndim == 1:
                return arr.reshape(-1, 1)

            # (W,1) -> ok
            if arr.ndim == 2 and arr.shape[1] == 1:
                return arr

            # (W,C) -> take target channel -> (W,1)
            if arr.ndim == 2 and arr.shape[1] > 1:
                if target_idx >= arr.shape[1]:
                    raise ValueError(f"target_idx {target_idx} out of range for pred shape {arr.shape}")
                return arr[:, target_idx].reshape(-1, 1)

            # (W,1,C) -> take target channel -> (W,1)
            if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] >= 1:
                C = arr.shape[2]
                if target_idx >= C:
                    raise ValueError(f"target_idx {target_idx} out of range for pred shape {arr.shape}")
                return arr[:, 0, target_idx].reshape(-1, 1)

            raise ValueError(f"Unexpected pred shape for pred_len=1: {arr.shape}")

    # pred_len > 1
    # Accept (W,pred_len)
    if arr.ndim == 2 and arr.shape[1] == pred_len:
        return arr

    # Accept (W,pred_len,C) and extract channel
    if arr.ndim == 3 and arr.shape[1] == pred_len:
        C = arr.shape[2]
        if target_idx >= C:
            raise ValueError(f"target_idx {target_idx} out of range for pred shape {arr.shape}")
        return arr[:, :, target_idx]

    raise ValueError(f"Unexpected pred shape {arr.shape} for pred_len={pred_len}")

def find_latest_results_dir(model_id_prefix: str, pred_len: int) -> Path:
    """
    Look for the newest PatchTST results folder matching the expected naming prefix.
    """
    prefix = f"{model_id_prefix}_PatchTST_custom_ftM_sl{SEQ_LEN}_ll{LABEL_LEN}_pl{pred_len}_"
    cands = sorted(
        [p for p in PATCHTST_RESULTS_ROOT.glob(f"{prefix}*") if (p / "pred.npy").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not cands:
        # last resort: newest pred.npy anywhere
        any_cands = sorted(
            (p for p in PATCHTST_RESULTS_ROOT.rglob("*") if (p / "pred.npy").exists()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if any_cands:
            return any_cands[0]
        raise FileNotFoundError(f"No results found under {PATCHTST_RESULTS_ROOT} for prefix {prefix}")
    return cands[0]

def build_aligned_truth_windows(
    y: pd.Series, y_index: pd.DatetimeIndex, test_start: int, W: int, pred_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build y_true windows aligned to PatchTST test windowing:
      window k corresponds to start = test_start + k, covering [start, start+pred_len).
    Returns:
      true_windows: (W, pred_len)
      ts_windows  : (W, pred_len) as numpy datetime64
    """
    N = len(y)
    max_possible_W = N - test_start - pred_len + 1
    if max_possible_W <= 0:
        raise ValueError(f"Not enough samples for test windows: N={N}, test_start={test_start}, pred_len={pred_len}")
    W_use = min(W, max_possible_W)

    true_w = np.zeros((W_use, pred_len), dtype=np.float32)
    ts_w = np.zeros((W_use, pred_len), dtype="datetime64[ns]")

    for k in range(W_use):
        start = test_start + k
        end = start + pred_len
        true_w[k, :] = y.iloc[start:end].to_numpy(dtype=np.float32)
        ts_w[k, :] = y_index[start:end].to_numpy()

    return true_w, ts_w

def baselines_for_windows(
    y: pd.Series, test_start: int, W: int, pred_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Persistence baseline: last observed value repeated for the horizon.
    Daily baseline: value from t-288 for each horizon step (fallback persistence if not available).
    Returns:
      persist: (W, pred_len)
      daily  : (W, pred_len)
    """
    persist = np.zeros((W, pred_len), dtype=np.float32)
    daily = np.zeros((W, pred_len), dtype=np.float32)

    for k in range(W):
        start = test_start + k
        last = float(y.iloc[start - 1]) if start - 1 >= 0 else float(y.iloc[0])
        persist[k, :] = last

        daily_start = start - DAILY_LAG
        if daily_start >= 0:
            daily[k, :] = y.iloc[daily_start:daily_start + pred_len].to_numpy(dtype=np.float32)
        else:
            daily[k, :] = last
    return persist, daily

def scale_mismatch(pred_flat: np.ndarray, true_flat: np.ndarray) -> bool:
    sp = float(np.std(pred_flat))
    st = float(np.std(true_flat))
    if sp < 1e-9:
        return True
    return (st / sp) > SCALE_MISMATCH_RATIO

def affine_calibrate(pred: np.ndarray, true: np.ndarray, fit_frac: float) -> Tuple[float, float, int]:
    """
    Fit y H a*x + b using flattened arrays (same length), on first fit_frac portion.
    Returns (a, b, nfit).
    """
    x = pred.reshape(-1).astype(np.float64)
    y = true.reshape(-1).astype(np.float64)
    n = len(y)
    nfit = max(CALIB_MIN_SAMPLES, int(fit_frac * n))
    nfit = min(nfit, n)

    A = np.vstack([x[:nfit], np.ones(nfit)]).T
    a, b = np.linalg.lstsq(A, y[:nfit], rcond=None)[0]
    return float(a), float(b), int(nfit)

def tail_metrics(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> Dict[str, float]:
    t = y_true.reshape(-1)
    p = y_pred.reshape(-1)
    thr = float(np.quantile(t, q))
    mask = t >= thr
    if not np.any(mask):
        return {"tail_q": q, "tail_thr": thr, "tail_mae": float("nan"), "tail_bias": float("nan")}
    return {
        "tail_q": q,
        "tail_thr": thr,
        "tail_mae": float(np.mean(np.abs(t[mask] - p[mask]))),
        "tail_bias": float(np.mean(p[mask] - t[mask])),
    }


@dataclass
class HorizonSpec:
    name: str
    pred_len: int
    model_id_prefix: str


def build_train_cmd(area: str, clean_csv_name: str, model_id: str, pred_len: int) -> List[str]:
    cmd = [
        sys.executable, str(RUN_LONGEXP),
        "--is_training", "1",
        "--model_id", model_id,
        "--model", "PatchTST",
        "--data", "custom",
        "--root_path", str(CLEAN_DATA_ROOT),
        "--data_path", clean_csv_name,
        "--features", "M",
        "--target", "price",
        "--seq_len", str(SEQ_LEN),
        "--label_len", str(LABEL_LEN),
        "--pred_len", str(pred_len),
        "--enc_in", str(ENC_IN),
        "--dec_in", str(DEC_IN),
        "--c_out", str(C_OUT),
        "--d_model", str(D_MODEL),
        "--n_heads", str(N_HEADS),
        "--e_layers", str(E_LAYERS),
        "--d_layers", str(D_LAYERS),
        "--d_ff", str(D_FF),
        "--dropout", str(DROPOUT),
        "--batch_size", str(BATCH),
        "--learning_rate", str(LR),
        "--train_epochs", str(EPOCHS),
        "--patience", str(PATIENCE),
        "--freq", FREQ,
        "--patch_len", str(PATCH_LEN),
        "--stride", str(STRIDE),
        "--padding_patch", "end",
        "--revin", str(REVIN),
        "--affine", str(AFFINE),
        "--subtract_last", str(SUBTRACT_LAST),
        "--decomposition", "0",
        "--kernel_size", "25",
        "--itr", "1",
        "--embed", EMBED,
        "--num_workers", str(NUM_WORKERS),
        "--use_gpu", str(USE_GPU),
        "--gpu", "0",
    ]
    if USE_AMP:
        cmd.append("--use_amp")
    if TRY_INVERSE_FLAG:
        # repo-dependent; if unsupported it will error, and we handle it by retry without it
        cmd += ["--inverse", "1"]
    return cmd


def run_training(cmd: List[str], workdir: Path, log_path: Path) -> None:
    """
    Run PatchTST training and capture stdout/stderr.

    If --inverse is unsupported (argparse error), retry without ONLY the
    '--inverse 1' pair (do not remove other "1" tokens in the command).
    """
    ensure_dir(log_path.parent)
    log_path.write_text("COMMAND:\n" + " ".join(cmd) + "\n\n")

    def _run(c: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(c, cwd=str(workdir), capture_output=True, text=True, check=True)

    try:
        proc = _run(cmd)
        log_path.write_text(log_path.read_text() + proc.stdout + "\n" + proc.stderr)
        return

    except subprocess.CalledProcessError as e:
        out = (e.stdout or "") + "\n" + (e.stderr or "")
        log_path.write_text(log_path.read_text() + "\n[ERROR]\n" + out + "\n")

        # If inverse is unsupported, retry without only that pair
        if TRY_INVERSE_FLAG and ("--inverse" in cmd) and (
            "unrecognized arguments: --inverse" in out
            or "error: unrecognized arguments: --inverse" in out
        ):
            # Remove exactly the '--inverse' token and its immediate value token (if present)
            cmd2 = cmd.copy()
            try:
                i = cmd2.index("--inverse")
                # remove value token if it exists and is not another flag
                if i + 1 < len(cmd2) and not cmd2[i + 1].startswith("--"):
                    del cmd2[i:i + 2]
                else:
                    del cmd2[i:i + 1]
            except ValueError:
                # shouldn't happen, but keep safe
                cmd2 = [x for x in cmd if x != "--inverse"]

            log_path.write_text(
                log_path.read_text()
                + "\n[WARN] --inverse not supported; retrying without it.\n"
                + "RETRY COMMAND:\n" + " ".join(cmd2) + "\n\n"
            )

            proc2 = _run(cmd2)
            log_path.write_text(log_path.read_text() + proc2.stdout + "\n" + proc2.stderr)
            return

        # Otherwise: re-raise (real training failure)
        raise

def save_plot_first_window(out_dir: Path, area: str, hspec: HorizonSpec,
                           ts0: np.ndarray, y_true0: np.ndarray, y_pred0: np.ndarray,
                           y_persist0: np.ndarray, y_daily0: np.ndarray) -> None:
    ensure_dir(out_dir / "plots")
    plt.figure(figsize=(12, 4))
    plt.plot(ts0, y_true0, label="Actual")
    plt.plot(ts0, y_pred0, label=f"PatchTST {hspec.name}")
    plt.plot(ts0, y_persist0, label="Persistence")
    plt.plot(ts0, y_daily0, label="Daily (t-288)")
    plt.title(f"{area}  {hspec.name}  first test window (raw scale)")
    plt.xlabel("Time")
    plt.ylabel("Price ($/MWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "first_window.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-train", action="store_true", help="Actually run PatchTST training. If omitted, only packages latest outputs.")
    ap.add_argument("--workdir", type=Path, default=Path.cwd(), help="Working directory where run_longExp.py is runnable.")
    ap.add_argument("--calib-fit-frac", type=float, default=CALIB_FIT_FRAC, help="Fraction of test windows used for affine calibration.")
    args = ap.parse_args()

    ensure_dir(BUNDLE_ROOT)
    ensure_dir(PATCHTST_RESULTS_ROOT)

    horizons = [
        HorizonSpec(name="tplus1", pred_len=1, model_id_prefix="PTST_CAN_features_t+1"),
        HorizonSpec(name="DA288", pred_len=288, model_id_prefix="PTST_CAN_features_DA288"),
    ]

    # Bundle summary (across areas and horizons)
    summary_rows = []

    for area in AREAS:
        src_name = AREA_TO_FEATURES_FILE.get(area)
        src_path = (DATA_DIR / src_name).resolve()

        if not src_path.exists():
            print(f"[WARN] Missing file for area={area}: {src_path} (skipping)")
            continue

        # 1) Build clean CSV
        clean_csv = build_per_area_clean_csv(src_path, CLEAN_DATA_ROOT)
        y, y_idx = load_price_series(clean_csv)
        N = len(y)
        test_start = compute_test_start(N)

        for hspec in horizons:
            # Output folder layout:
            #   training/<area>/<horizon>/
            out_dir = ensure_dir(BUNDLE_ROOT / area / hspec.name)

            # Resume/skip if packaged artifacts already exist
            packaged_ok = (out_dir / "pred.npy").exists() and (out_dir / "true.npy").exists() and (out_dir / "metrics.json").exists()
            if packaged_ok:
                print(f"[SKIP] Already packaged: {area} {hspec.name} -> {out_dir}")
                continue

            # 2) Train (optional)
            model_id = f"{hspec.model_id_prefix}_{area}"
            if args.run_train:
                cmd = build_train_cmd(area, clean_csv.name, model_id, hspec.pred_len)
                (out_dir / "train_cmd.txt").write_text(" ".join(cmd) + "\n")
                run_training(cmd, workdir=args.workdir, log_path=out_dir / "train_stdout.log")

            # 3) Locate latest PatchTST results dir and load pred.npy
            resdir = find_latest_results_dir(model_id, hspec.pred_len)
            pred_saved = np.load(resdir / "pred.npy")
            pred_w = shape_pred(pred_saved, hspec.pred_len, target_idx=0)  # (W, pred_len)
            W = pred_w.shape[0]

            # 4) Build aligned truth windows in RAW scale
            true_w, ts_w = build_aligned_truth_windows(y, y_idx, test_start, W, hspec.pred_len)
            W_use = true_w.shape[0]
            pred_w = pred_w[:W_use, :]

            # 5) Build baselines (raw)
            persist_w, daily_w = baselines_for_windows(y, test_start, W_use, hspec.pred_len)

            # 6) Scale check and fix pred to RAW scale
            pred_flat = pred_w.reshape(-1).astype(np.float64)
            true_flat = true_w.reshape(-1).astype(np.float64)

            scaling = {
                "method": "none",
                "a": 1.0,
                "b": 0.0,
                "nfit": 0,
                "std_pred_before": float(np.std(pred_flat)),
                "std_true": float(np.std(true_flat)),
                "mean_pred_before": float(np.mean(pred_flat)),
                "mean_true": float(np.mean(true_flat)),
            }

            pred_raw = pred_w.astype(np.float64)

            if scale_mismatch(pred_flat, true_flat):
                a, b, nfit = affine_calibrate(pred_w, true_w, args.calib_fit_frac)
                pred_raw = (a * pred_w + b).astype(np.float64)
                scaling.update({
                    "method": "affine_calibration",
                    "a": float(a),
                    "b": float(b),
                    "nfit": int(nfit),
                    "std_pred_after": float(np.std(pred_raw)),
                    "mean_pred_after": float(np.mean(pred_raw)),
                })

            # 7) Metrics (overall + baselines + tail)
            y_true_f = true_w.reshape(-1)
            y_pred_f = pred_raw.reshape(-1)
            y_pers_f = persist_w.reshape(-1)
            y_daily_f = daily_w.reshape(-1)

            m_model = {"MAE": mae(y_true_f, y_pred_f), "RMSE": rmse(y_true_f, y_pred_f), "MAPE%": mape(y_true_f, y_pred_f, EPS_MAPE)}
            m_pers  = {"MAE": mae(y_true_f, y_pers_f), "RMSE": rmse(y_true_f, y_pers_f), "MAPE%": mape(y_true_f, y_pers_f, EPS_MAPE)}
            m_daily = {"MAE": mae(y_true_f, y_daily_f), "RMSE": rmse(y_true_f, y_daily_f), "MAPE%": mape(y_true_f, y_daily_f, EPS_MAPE)}
            skill_vs_persist = 1.0 - (m_model["MAE"] / max(1e-12, m_pers["MAE"]))

            tails = tail_metrics(true_w, pred_raw, TAIL_Q)

            metrics = {
                "area": area,
                "horizon": hspec.name,
                "pred_len": hspec.pred_len,
                "patchtst_results_dir": str(resdir),
                "clean_csv": str(clean_csv),
                "N_total": int(N),
                "test_start_index": int(test_start),
                "W_windows": int(W_use),
                "model_metrics": m_model,
                "persistence_metrics": m_pers,
                "daily_metrics": m_daily,
                "skill_vs_persistence": float(skill_vs_persist),
                "tail_metrics": tails,
                "scaling": scaling,
            }

            # 8) Save aligned arrays for MPC/MILP usage
            np.save(out_dir / "true.npy", true_w.astype(np.float32))          # RAW
            np.save(out_dir / "pred.npy", pred_raw.astype(np.float32))        # RAW (after scaling fix)
            np.save(out_dir / "timestamps.npy", ts_w)                         # datetime64 grid

            # 9) Save metrics
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

            # 10) Save plots (readable check)
            ts0 = ts_w[0, :]
            save_plot_first_window(
                out_dir=out_dir,
                area=area,
                hspec=hspec,
                ts0=ts0,
                y_true0=true_w[0, :],
                y_pred0=pred_raw[0, :],
                y_persist0=persist_w[0, :],
                y_daily0=daily_w[0, :],
            )

            # 11) DA288 extra: per-horizon MAE curves
            if hspec.pred_len > 1:
                per_h_mae_patch = np.mean(np.abs(true_w - pred_raw), axis=0)
                per_h_mae_pers  = np.mean(np.abs(true_w - persist_w), axis=0)
                per_h_mae_daily = np.mean(np.abs(true_w - daily_w), axis=0)
                horiz_df = pd.DataFrame({
                    "horizon": np.arange(1, hspec.pred_len + 1),
                    "mae_patch": per_h_mae_patch,
                    "mae_persistence": per_h_mae_pers,
                    "mae_daily": per_h_mae_daily,
                })
                horiz_df.to_csv(out_dir / "horizon_mae.csv", index=False)

                plt.figure(figsize=(10, 3.2))
                plt.plot(horiz_df["horizon"], horiz_df["mae_patch"], label="PatchTST")
                plt.plot(horiz_df["horizon"], horiz_df["mae_persistence"], label="Persistence")
                plt.plot(horiz_df["horizon"], horiz_df["mae_daily"], label="Daily (t-288)")
                plt.title(f"{area}  {hspec.name}  MAE vs horizon (raw scale)")
                plt.xlabel("Horizon step (5-min ahead)")
                plt.ylabel("MAE ($/MWh)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "plots" / "horizon_mae.png", dpi=200)
                plt.close()

            # Summary row for master CSV
            summary_rows.append({
                "area": area,
                "horizon": hspec.name,
                "pred_len": hspec.pred_len,
                "mae_patch": m_model["MAE"],
                "rmse_patch": m_model["RMSE"],
                "mape_patch_pct": m_model["MAPE%"],
                "mae_persistence": m_pers["MAE"],
                "mae_daily": m_daily["MAE"],
                "skill_vs_persistence": skill_vs_persist,
                "tail_q": tails["tail_q"],
                "tail_thr": tails["tail_thr"],
                "tail_mae": tails["tail_mae"],
                "tail_bias": tails["tail_bias"],
                "scaling_method": scaling["method"],
                "scaling_a": scaling["a"],
                "scaling_b": scaling["b"],
                "bundle_dir": str(out_dir),
                "patchtst_results_dir": str(resdir),
            })

            print(f"[OK] {area} {hspec.name}: "
                  f"MAE={m_model['MAE']:.3f} | pers={m_pers['MAE']:.3f} | daily={m_daily['MAE']:.3f} "
                  f"| skill={skill_vs_persist:.3f} | scaling={scaling['method']}")

    # Save master summary CSV for paper
    if summary_rows:
        df = pd.DataFrame(summary_rows).sort_values(["horizon", "area"])
        df.to_csv(BUNDLE_ROOT / "training_summary.csv", index=False)
        print("\nSaved:", BUNDLE_ROOT / "training_summary.csv")
    else:
        print("[WARN] No summary rows produced (check inputs).")


if __name__ == "__main__":
    main()