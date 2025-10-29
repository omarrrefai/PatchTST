#!/usr/bin/env python3
# ny_manitoba_day_readable.py
# One-day (2025-01-24) readable comparison for New-York vs Manitoba.
# - Auto-denormalizes predictions (identity / std-inverse / affine-from-true)
# - Builds persistence (t-1)
# - Produces readable plots: broken y-axis + spike-window zoom, scatter, residuals
# - Prints concise metrics table and saves per-target CSV streams

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec

# -------- USER SETTINGS --------
MASTER_CSV   = Path("/home/omaralrefai/dev/PatchTST/.dataset/canada/canada_realtime_ENGY_2010_2025.csv")
DATA_ROOT    = MASTER_CSV.parent / "per_target_timeonly_clean"
RESULTS_ROOT = Path("results")

BEST_TARGET  = "New-York"
WORST_TARGET = "Manitoba"

DAY = pd.Timestamp("2025-01-24")     # change if needed
DAY_START = DAY
DAY_END   = DAY + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

# Must match training
SEQ_LEN, LABEL_LEN, PRED_LEN = 576, 72, 1
SPLIT = (0.7, 0.1, 0.2)
CAL_SLICE = 500

OUT_DIR = RESULTS_ROOT / "ny_vs_manitoba_day_readable"
FIG_DIR = OUT_DIR / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Global styling for readability
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.6,
})
# --------------------------------

def mae(a,b): return float(np.mean(np.abs(a-b)))
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def mape(a,b,eps=1.0): return float(np.mean(np.abs(a-b)/np.maximum(eps,np.abs(a))) * 100.0)

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
    if s.ndim == 0: return s.reshape(-1,1)
    if s.ndim == 1: return s.reshape(-1,1)
    if s.ndim == 2: return s if s.shape[0] >= s.shape[1] else s.T
    if s.ndim == 3:
        t_ax = int(np.argmax(s.shape))
        s2 = np.moveaxis(s, t_ax, 0)
        return s2.reshape(s2.shape[0], -1)
    raise ValueError(f"Unsupported pred.npy ndim={s.ndim}")

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
            corr = float(np.corrcoef(pj, ywin)[0,1])
        if np.isfinite(corr) and corr > best_corr:
            best_corr, best_j = corr, j
    return pred_WC[:L, best_j].astype(float), best_j, best_corr

def affine_from_true(model_true: np.ndarray, actual_true: np.ndarray):
    L = min(model_true.size, actual_true.size)
    x = model_true[:L].reshape(-1,1); y = actual_true[:L].reshape(-1,1)
    X = np.hstack([x, np.ones_like(x)])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a = float(beta[0,0]); b = float(beta[1,0])
        if not np.isfinite(a) or not np.isfinite(b): a,b = 1.0,0.0
    except Exception:
        a,b = 1.0,0.0
    return a,b

def pick_inverse(y_true_cal, y_pred_model_cal, train_mean, train_std, model_true_cal=None):
    best = ("identity", 1.0, 0.0, mae(y_true_cal, y_pred_model_cal))
    if np.isfinite(train_std) and train_std > 1e-12:
        y_std = y_pred_model_cal * train_std + train_mean
        m = mae(y_true_cal, y_std)
        if m < best[3]: best = ("std_inv", train_std, train_mean, m)
    if model_true_cal is not None and model_true_cal.size >= 10:
        a,b = affine_from_true(model_true_cal, y_true_cal)
        y_aff = y_pred_model_cal * a + b
        m = mae(y_true_cal, y_aff)
        if m < best[3]: best = ("affine", a, b, m)
    return best[0], best[1], best[2]

def build_persistence(y: pd.Series, idx: np.ndarray):
    out = np.empty(len(idx), dtype=float)
    for k,t in enumerate(idx):
        out[k] = float(y.iloc[t-1]) if t-1 >= 0 else float(y.iloc[0])
    return out

# --------- READABLE PLOTS ----------
def plot_day_readable(df_day: pd.DataFrame, target: str, savepath: Path):
    """
    df_day columns: ts, y_true, y_pred, persist (one day).
    Produces a broken y-axis plot so spikes don't flatten the rest.
    """
    ts   = pd.to_datetime(df_day["ts"])
    y    = df_day["y_true"].to_numpy()
    yhat = df_day["y_pred"].to_numpy()
    per  = df_day["persist"].to_numpy()

    y95  = np.percentile(y, 95)
    ymax = float(np.max(y))

    # No big spike: single clean plot
    if ymax <= y95 * 1.15:
        fig, ax = plt.subplots(figsize=(12, 3.8))
        ax.plot(ts, y,    label="Actual",      lw=1.8)
        ax.plot(ts, yhat, label="PatchTST",    lw=1.6)
        ax.plot(ts, per,  label="Persistence (t-1)", lw=1.2)
        ax.set_title(f"{target} — {ts.iloc[0].date()}")
        ax.set_xlabel("Time"); ax.set_ylabel("ENGY ($/MWh)")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.grid(alpha=0.25); ax.legend(loc="upper right", frameon=False)
        fig.tight_layout(); fig.savefig(savepath, dpi=180); plt.close(fig)
        return

    # Broken y-axis with spike zoom
    fig = plt.figure(figsize=(12, 5.2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1.0], hspace=0.05)
    ax_top   = fig.add_subplot(gs[0])
    ax_spike = fig.add_subplot(gs[1], sharex=ax_top)

    # Top: bulk range
    ax_top.plot(ts, y,    label="Actual",      lw=1.8)
    ax_top.plot(ts, yhat, label="PatchTST",    lw=1.6)
    ax_top.plot(ts, per,  label="Persistence (t-1)", lw=1.2)
    ax_top.set_ylim(0, y95 * 1.05)
    ax_top.set_ylabel("ENGY ($/MWh)")
    ax_top.set_title(f"{target} — {ts.iloc[0].date()} (broken y-axis)")
    ax_top.grid(alpha=0.25)
    ax_top.legend(loc="upper left", ncol=3, frameon=False)

    # Bottom: spike zoom
    low = max(y95 * 0.95, y95 - 5)
    ax_spike.plot(ts, y,    lw=1.8)
    ax_spike.plot(ts, yhat, lw=1.6)
    ax_spike.plot(ts, per,  lw=1.2)
    ax_spike.set_ylim(low, ymax * 1.05)
    ax_spike.set_xlabel("Time"); ax_spike.set_ylabel("Spike zoom")
    ax_spike.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_spike.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax_spike.grid(alpha=0.25)

    # Visual break marks
    ax_top.spines['bottom'].set_visible(False)
    ax_spike.spines['top'].set_visible(False)
    ax_top.tick_params(labelbottom=False)
    d = .5
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8, linestyle='none', color='k', mec='k', mew=1)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_spike.plot([0, 1], [1, 1], transform=ax_spike.transAxes, **kwargs)

    fig.tight_layout(); fig.savefig(savepath, dpi=180); plt.close(fig)

def add_spike_window(df_day: pd.DataFrame, target: str, savepath: Path):
    ts = pd.to_datetime(df_day["ts"])
    y  = df_day["y_true"].to_numpy()
    peak_idx = int(np.argmax(y))
    t0 = ts.iloc[max(0, peak_idx - 24)]   # ~2 hours before
    t1 = ts.iloc[min(len(ts)-1, peak_idx + 24)]
    m = (ts >= t0) & (ts <= t1)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(ts[m], df_day["y_true"][m], label="Actual", lw=1.8)
    ax.plot(ts[m], df_day["y_pred"][m], label="PatchTST", lw=1.6)
    ax.plot(ts[m], df_day["persist"][m], label="Persistence (t-1)", lw=1.2)
    ax.set_title(f"{target} — spike window around peak")
    ax.set_xlabel("Time"); ax.set_ylabel("ENGY ($/MWh)")
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(alpha=0.25); ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(savepath, dpi=180); plt.close(fig)
# --------------------------------

def run_for_day(target: str):
    # 1) Actuals
    per_csv = DATA_ROOT / f"{target.replace(' ', '_')}_timeonly_clean.csv"
    y = load_series(per_csv, target)
    N = len(y)
    num_train = int(N*SPLIT[0]); num_val = int(N*SPLIT[1])
    train_mean = float(y.iloc[:num_train].mean())
    train_std  = float(y.iloc[:num_train].std(ddof=0)) if num_train>1 else float(y.std(ddof=0))
    if not np.isfinite(train_std) or train_std < 1e-12: train_std = 1.0

    # 2) Predictions
    rdir = latest_results_dir_for(target)
    pred = np.load(rdir / "pred.npy")
    true_path = rdir / "true.npy"
    model_true = np.load(true_path) if true_path.exists() else None

    pred_WC = to_time_channels(pred)
    true_WC = to_time_channels(model_true) if model_true is not None else None
    W = pred_WC.shape[0]

    # 3) Align to test window
    test_start = num_train + num_val
    if test_start + W + PRED_LEN > N:
        test_start = max(SEQ_LEN + LABEL_LEN, N - (W + PRED_LEN))
    idx = np.arange(test_start, min(test_start + W, N))
    ts  = pd.to_datetime(y.index[idx])
    y_true_full = y.iloc[idx].to_numpy(dtype=float)

    # 4) Pick pred column; same column from model_true if present
    y_pred_model_full, picked_j, picked_corr = choose_col_by_corr(pred_WC[:len(idx), :], y_true_full)
    mtrue_col = None
    if true_WC is not None and picked_j < true_WC.shape[1]:
        mtrue_col = true_WC[:len(idx), picked_j].astype(float)

    # 5) Choose inverse on small calibration slice
    Lcal = min(CAL_SLICE, y_true_full.size)
    inv_name, a_inv, b_inv = pick_inverse(
        y_true_full[:Lcal], y_pred_model_full[:Lcal], train_mean, train_std,
        mtrue_col[:Lcal] if mtrue_col is not None else None
    )

    # 6) Denormalize & persistence
    y_pred_full = y_pred_model_full * a_inv + b_inv
    persist_full = build_persistence(y, idx)

    # 7) Slice the day
    mask = (ts >= DAY_START) & (ts <= DAY_END)
    if not mask.any():
        raise RuntimeError(f"No data for {target} on {DAY.date()}.")
    df = pd.DataFrame({
        "ts": ts[mask],
        "y_true": y_true_full[mask],
        "y_pred": y_pred_full[mask],
        "persist": persist_full[mask]
    }).reset_index(drop=True)

    # 8) Metrics
    m_mae  = mae(df.y_true, df.y_pred)
    m_rmse = rmse(df.y_true, df.y_pred)
    m_mape = mape(df.y_true, df.y_pred, 1.0)
    p_mae  = mae(df.y_true, df.persist)
    skill  = 1.0 - (m_mae / max(p_mae, 1e-12))
    corr   = float(np.corrcoef(df.y_true, df.y_pred)[0,1]) if df.y_true.std()>1e-12 else 0.0

    # 9) Readable plots
    plot_day_readable(df, target, FIG_DIR / f"{target.replace(' ', '_')}_timeseries_broken.png")
    add_spike_window(df, target, FIG_DIR / f"{target.replace(' ', '_')}_spike_window.png")

    # 10) Extras: scatter & residuals
    # Scatter
    plt.figure(figsize=(4.2,4.2))
    plt.scatter(df.y_true, df.y_pred, s=6, alpha=0.5)
    lims = [min(df.y_true.min(), df.y_pred.min()), max(df.y_true.max(), df.y_pred.max())]
    plt.plot(lims, lims, 'k-', linewidth=1)
    plt.xlabel("Actual ($/MWh)"); plt.ylabel("Predicted ($/MWh)")
    plt.title(f"{target} — Pred vs Actual ({DAY.date()})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{target.replace(' ', '_')}_scatter.png", dpi=180)
    plt.close()

    # Residual histogram
    resid = df.y_true.values - df.y_pred.values
    plt.figure(figsize=(8,3.2))
    plt.hist(resid, bins=40, alpha=0.95)
    plt.axvline(0, color="k", linewidth=1)
    plt.title(f"{target} — Residuals (y_true − y_pred), {DAY.date()}")
    plt.xlabel("Residual ($/MWh)"); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{target.replace(' ', '_')}_residual_hist.png", dpi=180)
    plt.close()

    # 11) Save stream & return summary
    df.to_csv(OUT_DIR / f"{target.replace(' ', '_')}_stream_{DAY.date()}.csv", index=False)
    return {
        "target": target, "n_points": int(len(df)),
        "MAE": m_mae, "RMSE": m_rmse, "MAPE%": m_mape,
        "MAE_persist": p_mae, "Skill_vs_persist": skill, "Corr": corr,
        "picked_col": picked_j, "corr_pickstage": picked_corr,
        "inverse": inv_name, "inv_a": a_inv, "inv_b": b_inv,
    }

def main():
    rows = []
    for tgt in [BEST_TARGET, WORST_TARGET]:
        try:
            row = run_for_day(tgt)
            rows.append(row)
            print(f"[OK] {tgt}: MAE={row['MAE']:.3f} | Pers={row['MAE_persist']:.3f} | "
                  f"Skill={row['Skill_vs_persist']:.3f} | Corr={row['Corr']:.3f} | "
                  f"inv={row['inverse']} a={row['inv_a']:.3g} b={row['inv_b']:.3g} | col={row['picked_col']}")
        except Exception as e:
            print(f"[WARN] Failed for {tgt}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / f"NY_vs_Manitoba_{DAY.date()}_summary.csv", index=False)
        cols = ["target","n_points","MAE","MAE_persist","Skill_vs_persist","RMSE","MAPE%","Corr",
                "inverse","inv_a","inv_b","picked_col"]
        print("\n=== One-day summary (", DAY.date(), ") ===")
        print(df[cols].to_string(index=False))
        print("\nSaved:")
        print("  Streams:", OUT_DIR)
        print("  Figures:", FIG_DIR)
        print("  Summary CSV:", OUT_DIR / f"NY_vs_Manitoba_{DAY.date()}_summary.csv")

if __name__ == "__main__":
    main()
