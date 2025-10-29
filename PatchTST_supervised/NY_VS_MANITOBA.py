#!/usr/bin/env python3
# ny_manitoba_focus_study.py
# Focused, readable analysis comparing New-York (best) vs Manitoba (worst).

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= USER CONFIG =================
MASTER_CSV = Path("/home/omaralrefai/dev/PatchTST/.dataset/canada/canada_realtime_ENGY_2010_2025.csv")
DATA_ROOT  = MASTER_CSV.parent / "per_target_timeonly_clean"
RESULTS_ROOT = Path("results")

BEST_TARGET  = "New-York"
WORST_TARGET = "Manitoba"

# Choose your month (inclusive) — change these two lines as you like
MONTH_START = pd.Timestamp("2025-01-01")
MONTH_END   = pd.Timestamp("2025-01-31 23:59:59")

# Must match how you trained
SEQ_LEN, LABEL_LEN, PRED_LEN = 576, 72, 1
SPLIT = (0.7, 0.1, 0.2)
CAL_SLICE = 500   # points to decide best denorm mapping
PER_HOUR = 12     # 5-min data
ROLL_STD_WIN = 12 # 1 hour rolling std for volatility plot

OUT_DIR   = RESULTS_ROOT / "ny_vs_manitoba"
FIG_DIR   = OUT_DIR / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
# ==============================================

def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mape(a, b, eps=1.0): return float(np.mean(np.abs(a - b) / np.maximum(eps, np.abs(a))) * 100.0)

def latest_results_dir_for(target: str) -> Path:
    prefix = f"PTST_CAN_timeonly_t+1_{target.replace(' ', '_')}_PatchTST_custom_ftM_sl{SEQ_LEN}_ll{LABEL_LEN}_pl{PRED_LEN}_"
    cands = sorted([p for p in RESULTS_ROOT.glob(f"{prefix}*") if (p / "pred.npy").exists()],
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if cands: return cands[0]
    fb = sorted([p for p in RESULTS_ROOT.rglob("*") if (p / "pred.npy").exists() and target.replace(' ', '_') in p.name],
                key=lambda p: p.stat().st_mtime, reverse=True)
    if fb: return fb[0]
    raise FileNotFoundError(f"No results with pred.npy for {target}")

def load_series(per_target_csv: Path, target: str) -> pd.Series:
    df = pd.read_csv(per_target_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True).set_index("date")
    return df[target].astype(float)

def to_time_channels(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None: return None
    s = np.squeeze(arr)
    if s.ndim == 0: return s.reshape(-1, 1)
    if s.ndim == 1: return s.reshape(-1, 1)
    if s.ndim == 2: return s if s.shape[0] >= s.shape[1] else s.T
    if s.ndim == 3:
        axes = np.array(s.shape)
        t_ax = int(np.argmax(axes))
        s2 = np.moveaxis(s, t_ax, 0)
        W = s2.shape[0]; C = int(np.prod(s2.shape[1:]))
        return s2.reshape(W, C)
    raise ValueError(f"Unsupported ndim={s.ndim}")

def choose_col_by_corr(pred_WC: np.ndarray, y_true_raw: np.ndarray):
    W, C = pred_WC.shape
    L = min(W, y_true_raw.size)
    ywin = y_true_raw[:L]
    best_j, best_corr = 0, -np.inf
    for j in range(C):
        pj = pred_WC[:L, j].astype(float)
        if np.std(pj) < 1e-12 or np.std(ywin) < 1e-12:
            corr = -np.inf
        else:
            c = np.corrcoef(pj, ywin)[0, 1]
            corr = float(c) if np.isfinite(c) else -np.inf
        if corr > best_corr:
            best_corr, best_j = corr, j
    return pred_WC[:L, best_j].astype(float), best_j, best_corr

def affine_from_true(model_true: np.ndarray, actual_true: np.ndarray):
    L = min(model_true.size, actual_true.size)
    x = model_true[:L].reshape(-1, 1)
    y = actual_true[:L].reshape(-1, 1)
    X = np.hstack([x, np.ones_like(x)])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a = float(beta[0, 0]); b = float(beta[1, 0])
        if not np.isfinite(a) or not np.isfinite(b): a, b = 1.0, 0.0
    except Exception:
        a, b = 1.0, 0.0
    return a, b

def pick_inverse(y_true_cal, y_pred_model_cal, train_mean, train_std, model_true_cal=None):
    best = ("identity", 1.0, 0.0, mae(y_true_cal, y_pred_model_cal))
    if np.isfinite(train_std) and train_std > 1e-12:
        y_std = y_pred_model_cal * train_std + train_mean
        m = mae(y_true_cal, y_std)
        if m < best[3]: best = ("std_inv", train_std, train_mean, m)
    if model_true_cal is not None and model_true_cal.size >= 10:
        a, b = affine_from_true(model_true_cal, y_true_cal)
        y_aff = y_pred_model_cal * a + b
        m = mae(y_true_cal, y_aff)
        if m < best[3]: best = ("affine", a, b, m)
    return best[0], best[1], best[2]

def build_persistence(y: pd.Series, idx: np.ndarray):
    out = np.empty(len(idx), dtype=float)
    for k, t in enumerate(idx):
        out[k] = float(y.iloc[t-1]) if t-1 >= 0 else float(y.iloc[0])
    return out

def compute_fluctuation_metrics(series: np.ndarray):
    # returns-based metrics on 5-min grid
    diff = np.diff(series)
    madiff = np.abs(diff)
    mean_abs_change = float(madiff.mean())
    p95_abs_change  = float(np.percentile(madiff, 95))
    med_abs_change  = float(np.median(madiff))
    large_move_thr  = 2.0 * med_abs_change if med_abs_change > 0 else 0.0
    frac_large_moves = float((madiff > large_move_thr).mean()) if large_move_thr > 0 else 0.0
    # peak count (local maxima above threshold)
    # threshold = median abs change; a point is a peak if it's greater than neighbors by >= thr
    thr = med_abs_change
    x = series
    peaks = 0
    for i in range(1, len(x)-1):
        if (x[i] - x[i-1] >= thr) and (x[i] - x[i+1] >= thr):
            peaks += 1
    peak_to_peak = float(np.max(x) - np.min(x)) if len(x) else 0.0
    return {
        "mean_abs_change": mean_abs_change,
        "p95_abs_change": p95_abs_change,
        "frac_large_moves": frac_large_moves,
        "peak_count": int(peaks),
        "peak_to_peak_range": peak_to_peak,
    }

def run_target(target: str):
    # 1) Load series & splits
    per_csv = DATA_ROOT / f"{target.replace(' ', '_')}_timeonly_clean.csv"
    y = load_series(per_csv, target)
    N = len(y)
    num_train = int(N * SPLIT[0]); num_val = int(N * SPLIT[1])
    train_mean = float(y.iloc[:num_train].mean())
    train_std  = float(y.iloc[:num_train].std(ddof=0)) if num_train > 1 else float(y.std(ddof=0))
    if not np.isfinite(train_std) or train_std < 1e-12: train_std = 1.0

    # 2) Predictions
    resdir = latest_results_dir_for(target)
    pred = np.load(resdir / "pred.npy")
    true_path = resdir / "true.npy"
    model_true = np.load(true_path) if true_path.exists() else None

    pred_WC = to_time_channels(pred)
    true_WC = to_time_channels(model_true) if model_true is not None else None
    W = pred_WC.shape[0]

    # 3) Align test window
    test_start = num_train + num_val
    if test_start + W + PRED_LEN > N:
        test_start = max(SEQ_LEN + LABEL_LEN, N - (W + PRED_LEN))
    idx = np.arange(test_start, min(test_start + W, N))
    ts  = y.index[idx]
    y_true_full = y.iloc[idx].to_numpy(dtype=float)

    # 4) Pick pred column by correlation; take same column from model_true if available
    y_pred_model_full, picked_j, picked_corr = choose_col_by_corr(pred_WC[:len(idx), :], y_true_full)
    model_true_col_full = None
    if true_WC is not None and picked_j < true_WC.shape[1]:
        model_true_col_full = true_WC[:len(idx), picked_j].astype(float)

    # 5) Choose inverse mapping on a small calibration slice
    Lcal = min(CAL_SLICE, y_true_full.size)
    inv_name, a_inv, b_inv = pick_inverse(
        y_true_full[:Lcal], y_pred_model_full[:Lcal], train_mean, train_std,
        model_true_col_full[:Lcal] if model_true_col_full is not None else None
    )

    # 6) Denormalize preds; build persistence
    y_pred_full = y_pred_model_full * a_inv + b_inv
    persist_full = build_persistence(y, idx)

    # 7) Slice the requested month
    df = pd.DataFrame({
        "ts": pd.to_datetime(ts),
        "y_true": y_true_full,
        "y_pred": y_pred_full,
        "persist": persist_full
    })
    month_df = df[(df["ts"] >= MONTH_START) & (df["ts"] <= MONTH_END)].reset_index(drop=True)
    if month_df.empty:
        raise RuntimeError(f"No data for {target} between {MONTH_START} and {MONTH_END}.")

    # 8) Metrics for the month
    m_mae  = mae(month_df.y_true, month_df.y_pred)
    m_rmse = rmse(month_df.y_true, month_df.y_pred)
    m_mape = mape(month_df.y_true, month_df.y_pred, 1.0)
    p_mae  = mae(month_df.y_true, month_df.persist)
    skill  = 1.0 - (m_mae / max(p_mae, 1e-12))
    corr   = float(np.corrcoef(month_df.y_true, month_df.y_pred)[0,1]) if month_df.y_true.std() > 1e-12 else 0.0

    # 9) Daily MAE bars (readable)
    month_df["date"] = month_df["ts"].dt.normalize()
    g = month_df.groupby("date", sort=True)
    daily = pd.DataFrame({
        "date": g["ts"].min().values,
        "mae_model": g.apply(lambda d: mae(d["y_true"].values, d["y_pred"].values)).values,
        "mae_persist": g.apply(lambda d: mae(d["y_true"].values, d["persist"].values)).values,
    }).sort_values("date")

    # 10) Fluctuation metrics
    flucs = compute_fluctuation_metrics(month_df["y_true"].values)

    # 11) Plots (clean & readable)

    # A) Time series (downsample if long)
    step = max(1, len(month_df)//4000)  # keep to ~4k points
    plt.figure(figsize=(12,4))
    plt.plot(month_df["ts"][::step], month_df["y_true"][::step], label="Actual")
    plt.plot(month_df["ts"][::step], month_df["y_pred"][::step], label="PatchTST")
    plt.plot(month_df["ts"][::step], month_df["persist"][::step], label="Persistence (t-1)")
    plt.title(f"{target} — {MONTH_START.date()} to {MONTH_END.date()}")
    plt.xlabel("Time"); plt.ylabel("ENGY ($/MWh)")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / f"{target.replace(' ', '_')}_month_timeseries.png", dpi=160)
    plt.close()

    # B) Daily MAE bars
    x = np.arange(len(daily))
    width = 0.42
    plt.figure(figsize=(12,4))
    plt.bar(x - width/2, daily["mae_model"], width, label="PatchTST")
    plt.bar(x + width/2, daily["mae_persist"], width, label="Persistence")
    plt.xticks(x, [d.strftime("%m-%d") for d in pd.to_datetime(daily["date"])], rotation=60, ha="right")
    plt.ylabel("MAE ($/MWh)")
    plt.title(f"{target} — Daily MAE in Month")
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / f"{target.replace(' ', '_')}_month_daily_mae.png", dpi=160)
    plt.close()

    # C) Residual histogram
    resid = month_df["y_true"].values - month_df["y_pred"].values
    plt.figure(figsize=(8,4))
    plt.hist(resid, bins=60, alpha=0.95)
    plt.axvline(0, color="k", linewidth=1)
    plt.title(f"{target} — Residuals (y_true − y_pred), Month")
    plt.xlabel("Residual ($/MWh)"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{target.replace(' ', '_')}_month_residual_hist.png", dpi=160)
    plt.close()

    # D) Volatility over time (rolling 1h std of returns)
    month_df["ret"] = month_df["y_true"].diff()
    month_df["vol_1h"] = month_df["ret"].rolling(ROLL_STD_WIN, min_periods=ROLL_STD_WIN//2).std()
    plt.figure(figsize=(12,3.6))
    plt.plot(month_df["ts"], month_df["vol_1h"])
    plt.title(f"{target} — Rolling 1h Std of Returns (volatility)"); plt.xlabel("Time"); plt.ylabel("Std($/MWh)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{target.replace(' ', '_')}_month_volatility.png", dpi=160)
    plt.close()

    # 12) Save month stream and daily table
    month_df.to_csv(OUT_DIR / f"{target.replace(' ', '_')}_month_stream.csv", index=False)
    daily.to_csv(OUT_DIR / f"{target.replace(' ', '_')}_month_daily_mae.csv", index=False)

    # 13) Return concise summary row
    return {
        "target": target,
        "picked_col": picked_j,
        "corr_pred_vs_y": picked_corr,
        "inverse_mapping": inv_name,
        "inv_a": a_inv, "inv_b": b_inv,
        "MAE": m_mae, "RMSE": m_rmse, "MAPE%": m_mape,
        "MAE_persist": p_mae, "Skill_vs_persist": skill, "Corr": corr,
        **{f"fluc_{k}": v for k, v in flucs.items()},
    }

def main():
    rows = []
    for tgt in [BEST_TARGET, WORST_TARGET]:
        try:
            row = run_target(tgt)
            rows.append(row)
            print(f"[OK] {tgt}: MAE={row['MAE']:.3f} | Pers={row['MAE_persist']:.3f} | Skill={row['Skill_vs_persist']:.3f} | "
                  f"Corr={row['Corr']:.3f} | inv={row['inverse_mapping']} a={row['inv_a']:.3g} b={row['inv_b']:.3g} | col={row['picked_col']}")
        except Exception as e:
            print(f"[WARN] Failed for {tgt}: {e}")

    if not rows:
        print("Nothing to summarize."); return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "NY_vs_Manitoba_summary.csv", index=False)

    # Print a compact comparison
    keep = ["target","MAE","MAE_persist","Skill_vs_persist","RMSE","MAPE%","Corr",
            "fluc_mean_abs_change","fluc_p95_abs_change","fluc_frac_large_moves",
            "fluc_peak_count","fluc_peak_to_peak_range"]
    cols = [c for c in keep if c in df.columns]
    print("\n=== NY vs Manitoba — Month Summary ===")
    print(df[cols].to_string(index=False))

    print("\nSaved:")
    print("  Per-target month streams/dailies:", OUT_DIR)
    print("  Figures:", FIG_DIR)
    print("  Summary CSV:", OUT_DIR / "NY_vs_Manitoba_summary.csv")

if __name__ == "__main__":
    main()
