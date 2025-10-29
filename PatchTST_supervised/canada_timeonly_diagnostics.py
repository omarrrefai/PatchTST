#!/usr/bin/env python3
# canada_timeonly_diagnostics.py
#
# Minimal, robust diagnostics:
# - Denormalize model to $/MWh (std_inv or affine if available)
# - Build a SIMPLE hybrid: if |PatchTST - Persistence| <= τ use Persistence else PatchTST
# - Choose τ per-target on a short calibration slice by maximizing DAILY win-rate (ties -> lower MAE)
# - Emit daily CSV with mae_persist, mae_model, mae_hybrid
# - Plot daily MAE (3 curves)
#
# Run: python3 canada_timeonly_diagnostics.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
MASTER_CSV = Path("/home/omaralrefai/dev/PatchTST/.dataset/canada/canada_realtime_ENGY_2010_2025.csv")
DATA_ROOT  = MASTER_CSV.parent / "per_target_timeonly_clean"
RESULTS_ROOT = Path("results")
OUT_DIR    = RESULTS_ROOT / "diagnostics_simple"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "Manitoba", "Manitoba SK", "Michigan", "Minnesota", "New-York", "Ontario",
    "Quebec AT", "Quebec B5D.B31L", "Quebec D4Z", "Quebec D5A", "Quebec H4Z",
    "Quebec H9A", "Quebec P33C", "Quebec Q4C", "Quebec X2Y",
]

# Train/test split params used in training
SEQ_LEN    = 576
LABEL_LEN  = 72
PRED_LEN   = 1
SPLIT      = (0.7, 0.1, 0.2)

# Calibration slice inside test window to pick τ
CAL_SLICE_POINTS = 8_000               # ≈ ~28 days of 5-min data
CAL_MIN_POINTS   = 2_000

# Candidate τ values (absolute $/MWh). We’ll add data-driven percentiles too.
BASE_TAU = [2.0, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0]

# Optional: one-day zoom plot (set to None to skip)
ZOOM_DAY = None  # e.g. "2025-01-24"
# -----------------------------

def mae(a, b):  return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def latest_results_dir_for(target: str) -> Path:
    prefix = f"PTST_CAN_timeonly_t+1_{target.replace(' ', '_')}_PatchTST_custom_ftM_sl{SEQ_LEN}_ll{LABEL_LEN}_pl{PRED_LEN}_"
    cands = sorted([p for p in RESULTS_ROOT.glob(f"{prefix}*") if (p / "pred.npy").exists()],
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if cands: return cands[0]
    fb = sorted([p for p in RESULTS_ROOT.rglob("*") if (p / "pred.npy").exists() and target.replace(' ', '_') in p.name],
                key=lambda p: p.stat().st_mtime, reverse=True)
    if fb: return fb[0]
    raise FileNotFoundError(f"No results with pred.npy for {target}")

def load_target_series(per_target_csv: Path, target: str) -> pd.Series:
    df = pd.read_csv(per_target_csv, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True).set_index("date")
    return df[target].astype(float)

def to_time_channels(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None: return None
    s = np.squeeze(arr)
    if s.ndim == 0: return s.reshape(-1,1)
    if s.ndim == 1: return s.reshape(-1,1)
    if s.ndim == 2: return s if s.shape[0] >= s.shape[1] else s.T
    if s.ndim == 3:
        t_ax = int(np.argmax(s.shape))
        s2 = np.moveaxis(s, t_ax, 0)  # time first
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
            corr = float(np.corrcoef(pj, ywin)[0, 1])
        if np.isfinite(corr) and corr > best_corr:
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

def build_persistence(y: pd.Series, idx: np.ndarray) -> np.ndarray:
    out = np.empty(len(idx), dtype=float)
    for k, t in enumerate(idx):
        out[k] = float(y.iloc[t-1]) if t-1 >= 0 else float(y.iloc[0])
    return out

def build_hybrid_abs(y_pred: np.ndarray, persist: np.ndarray, tau: float) -> np.ndarray:
    gap = np.abs(y_pred - persist)
    return np.where(gap <= float(tau), persist, y_pred)

def daily_mae_three_plot(daily_df: pd.DataFrame, target: str, out_png: Path):
    plt.figure(figsize=(12, 4))
    plt.plot(daily_df["date"], daily_df["mae_persist"], label="Persistence", linewidth=1.4)
    plt.plot(daily_df["date"], daily_df["mae_model"],   label="PatchTST",  linewidth=1.6)
    plt.plot(daily_df["date"], daily_df["mae_hybrid"],  label="Hybrid",    linewidth=2.0)
    plt.title(f"{target} — Daily MAE")
    plt.xlabel("Date"); plt.ylabel("MAE ($/MWh)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()

def day_zoom_plot(df_stream: pd.DataFrame, day_str: str, target: str, out_png: Path):
    day = pd.to_datetime(day_str).normalize()
    d = df_stream[df_stream["date"] == day]
    if d.empty: return
    plt.figure(figsize=(12, 4))
    plt.plot(d["ts"], d["y_true"],  label="Actual",      linewidth=2.0)
    plt.plot(d["ts"], d["persist"], label="Persistence", linewidth=1.4)
    plt.plot(d["ts"], d["y_pred"],  label="PatchTST",    linewidth=1.6)
    plt.plot(d["ts"], d["hybrid"],  label="Hybrid",      linewidth=2.0)
    plt.title(f"{target} — {day_str}")
    plt.xlabel("Time"); plt.ylabel("Price ($/MWh)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=160); plt.close()

def pick_tau_on_calibration(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            persist: np.ndarray,
                            ts: np.ndarray) -> tuple[float, float, float]:
    """
    Try a small τ grid (plus % points of |y_pred - persist|) on a calibration slice.
    Return (best_tau, win_rate, cal_mae_hybrid).
    """
    L = len(y_true)
    Lcal = min(CAL_SLICE_POINTS, L)
    if Lcal < CAL_MIN_POINTS:
        Lcal = L
    yt = y_true[:Lcal]; yp = y_pred[:Lcal]; per = persist[:Lcal]
    # Add data-driven thresholds from the gap distribution
    gap = np.abs(yp - per)
    qs  = np.percentile(gap, [60, 70, 80, 85, 90])
    tau_grid = sorted(set(list(BASE_TAU) + [float(x) for x in qs if np.isfinite(x) and x>0]))

    # Build daily grouping once
    dates = pd.to_datetime(ts[:Lcal]).normalize()
    df = pd.DataFrame({"date": dates, "y": yt, "m": yp, "p": per})

    best_tau, best_win, best_mae = tau_grid[0], -1.0, np.inf
    for tau in tau_grid:
        h = build_hybrid_abs(df["m"].values, df["p"].values, tau)
        # Daily MAE
        # (avoid deprecated include_groups warning by manual loop)
        gd = []
        for d, block in df.groupby("date", sort=True):
            gd.append({
                "mae_p": mae(block["y"].values, block["p"].values),
                "mae_h": mae(block["y"].values, h[block.index.values - df.index[0]])
            })
        daily_p = np.array([g["mae_p"] for g in gd])
        daily_h = np.array([g["mae_h"] for g in gd])
        win = float(np.mean(daily_h < daily_p)) if daily_p.size else 0.0
        m   = mae(df["y"].values, h)
        if (win > best_win) or (np.isclose(win, best_win) and m < best_mae):
            best_tau, best_win, best_mae = float(tau), float(win), float(m)
    return best_tau, best_win, best_mae

def main():
    rows = []

    for target in TARGETS:
        try:
            # 1) Load series
            per_csv = DATA_ROOT / f"{target.replace(' ', '_')}_timeonly_clean.csv"
            if not per_csv.exists():
                print(f"[WARN] Missing per-target CSV: {per_csv}")
                continue

            y = load_target_series(per_csv, target)
            N = len(y)
            num_train = int(N * SPLIT[0]); num_val = int(N * SPLIT[1])

            train_mean = float(y.iloc[:num_train].mean())
            train_std  = float(y.iloc[:num_train].std(ddof=0)) if num_train>1 else float(y.std(ddof=0))
            if not np.isfinite(train_std) or train_std < 1e-12: train_std = 1.0

            # 2) Load predictions (time x channels), pick best channel by corr
            resdir = latest_results_dir_for(target)
            pred = np.load(resdir / "pred.npy")
            true_path = resdir / "true.npy"
            true_model = np.load(true_path) if true_path.exists() else None

            pred_WC = to_time_channels(pred)
            true_WC = to_time_channels(true_model) if true_model is not None else None
            W = pred_WC.shape[0]

            # 3) Align to test split
            test_start = num_train + num_val
            if test_start + W + PRED_LEN > N:
                test_start = max(SEQ_LEN + LABEL_LEN, N - (W + PRED_LEN))
            idx = np.arange(test_start, min(test_start + W, N))
            ts  = y.index[idx]
            y_true_full = y.iloc[idx].to_numpy(float)

            # choose column and invert scaling
            y_pred_model_full, picked_j, picked_corr = choose_col_by_corr(pred_WC[:len(idx), :], y_true_full)

            # std_inv by default; if model-space true exists for that column, try affine
            y_pred_std = y_pred_model_full * train_std + train_mean
            best_name, a, b, best_mae = "std_inv", train_std, train_mean, mae(y_true_full, y_pred_std)
            if true_WC is not None and picked_j < true_WC.shape[1]:
                aa, bb = affine_from_true(true_WC[:len(idx), picked_j], y_true_full)
                y_aff  = y_pred_model_full * aa + bb
                m_aff  = mae(y_true_full, y_aff)
                if m_aff < best_mae:
                    best_name, a, b, best_mae = "affine", aa, bb, m_aff

            y_pred = y_pred_model_full * a + b
            persist = build_persistence(y, idx)

            # 4) Crop to same length
            L = min(len(y_true_full), len(y_pred), len(persist))
            y_true = y_true_full[:L]; y_pred = y_pred[:L]; persist = persist[:L]; ts = ts[:L]

            # 5) Pick τ on calibration slice
            tau, cal_win, cal_mae = pick_tau_on_calibration(y_true, y_pred, persist, ts)

            # 6) Build HYBRID on FULL test
            hybrid = build_hybrid_abs(y_pred, persist, tau)

            # 7) Per-point stream & daily MAE (3 curves)
            df_stream = pd.DataFrame({
                "ts": pd.to_datetime(ts),
                "y_true": y_true,
                "persist": persist,
                "y_pred": y_pred,
                "hybrid": hybrid,           # <-- ensure exists before daily ops
            })
            df_stream["date"] = df_stream["ts"].dt.normalize()

            # daily aggregation (no deprecated args)
            daily_rows = []
            for d, block in df_stream.groupby("date", sort=True):
                daily_rows.append({
                    "date": d,
                    "mae_persist": mae(block["y_true"].values, block["persist"].values),
                    "mae_model":   mae(block["y_true"].values, block["y_pred"].values),
                    "mae_hybrid":  mae(block["y_true"].values, block["hybrid"].values),
                })
            daily = pd.DataFrame(daily_rows).sort_values("date")

            # 8) Overall metrics
            mae_pers = mae(df_stream["y_true"].values, df_stream["persist"].values)
            mae_model = mae(df_stream["y_true"].values, df_stream["y_pred"].values)
            mae_hybrid = mae(df_stream["y_true"].values, df_stream["hybrid"].values)
            skill_model  = 1.0 - (mae_model  / max(1e-12, mae_pers))
            skill_hybrid = 1.0 - (mae_hybrid / max(1e-12, mae_pers))

            win_days_model  = int((daily["mae_model"]  < daily["mae_persist"]).sum())
            win_days_hybrid = int((daily["mae_hybrid"] < daily["mae_persist"]).sum())
            days = int(len(daily))
            win_rate_model  = float(win_days_model  / days) if days else 0.0
            win_rate_hybrid = float(win_days_hybrid / days) if days else 0.0

            # 9) Save CSVs and plots
            base = OUT_DIR / target.replace(" ", "_")
            df_stream.to_csv(base.with_suffix(".stream.csv"), index=False)
            daily.to_csv(base.with_suffix(".daily.csv"), index=False)
            daily_mae_three_plot(daily, target, base.with_suffix(".daily_mae.png"))
            if ZOOM_DAY:
                day_zoom_plot(df_stream, ZOOM_DAY, target, base.with_suffix(".zoom.png"))

            print(f"[OK] {target}: τ={tau:.2f} ({best_name}, a={a:.3g}, b={b:.3g}) | "
                  f"MAE P={mae_pers:.3f}, M={mae_model:.3f}, H={mae_hybrid:.3f} | "
                  f"Win% M={win_rate_model:.1%}, H={win_rate_hybrid:.1%} | cal_win={cal_win:.1%}")

            rows.append({
                "target": target,
                "results_dir": str(resdir),
                "picked_pred_column": int(picked_j),
                "corr_pred_vs_y": float(picked_corr),
                "inverse_mapping": best_name,
                "inv_a": float(a), "inv_b": float(b),
                "tau": float(tau),
                "overall_mae_persist": float(mae_pers),
                "overall_mae_model": float(mae_model),
                "overall_mae_hybrid": float(mae_hybrid),
                "skill_vs_persist_model": float(skill_model),
                "skill_vs_persist_hybrid": float(skill_hybrid),
                "days": days,
                "win_days_model": win_days_model,
                "win_rate_model": float(win_rate_model),
                "win_days_hybrid": win_days_hybrid,
                "win_rate_hybrid": float(win_rate_hybrid),
                "cal_win_rate": float(cal_win),
                "cal_mae": float(cal_mae),
            })

        except Exception as e:
            print(f"[WARN] Failed for {target}: {e}")

    if not rows:
        print("No targets processed."); return

    df = pd.DataFrame(rows).sort_values(
        ["win_rate_hybrid","skill_vs_persist_hybrid","overall_mae_hybrid"], ascending=[False, False, True]
    )
    out_csv = OUT_DIR / "diagnostics_summary.csv"
    df.to_csv(out_csv, index=False)

    cols = ["target","tau","inverse_mapping","inv_a","inv_b",
            "overall_mae_persist","overall_mae_model","overall_mae_hybrid",
            "skill_vs_persist_model","skill_vs_persist_hybrid",
            "win_rate_model","win_rate_hybrid","days","cal_win_rate","cal_mae"]
    print("\n=== SIMPLE HYBRID SUMMARY ===")
    print(df[cols].to_string(index=False))
    print("\nSaved:", out_csv)
    print("Per-target outputs in:", OUT_DIR)

if __name__ == "__main__":
    main()
