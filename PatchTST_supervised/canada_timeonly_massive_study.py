#!/usr/bin/env python3
# canada_timeonly_massive_study.py
"""
Massive analysis pipeline for PatchTST time-only runs on Canada markets.

What this does:
- Loads per-target time series and saved PatchTST predictions (any pred.npy shape).
- Denormalizes predictions back to $/MWh via best of:
    identity, train-std inverse, or affine fit using true.npy (if available).
- Compares ONLY vs Persistence (t-1) baseline.
- Computes diagnostics per target:
    * Skill vs persistence (1 - MAE_model/MAE_persist)
    * Day-by-day MAE + win-rate
    * Residual stats (mean, std, p95)
    * Autocorrelation & Partial ACF
    * Periodogram (spectral peaks; daily/weekly)
    * Volatility regime analysis (quartiles)
    * Approximate entropy (ApEn) & Permutation entropy (PE)
    * Cross-correlation between regions (optional pairwise)
- Exports:
    * CSVs with per-target and global summaries (results/study/*.csv)
    * Publication-ready plots (results/study/figs/*.png)

Dependencies: numpy, pandas, matplotlib (no seaborn).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================= CONFIG =======================
MASTER_CSV = Path("/home/omaralrefai/dev/PatchTST/.dataset/canada/canada_realtime_ENGY_2010_2025.csv")
DATA_ROOT  = MASTER_CSV.parent / "per_target_timeonly_clean"   # *_timeonly_clean.csv live here
RESULTS_ROOT = Path("results")
STUDY_DIR = RESULTS_ROOT / "study"
FIG_DIR   = STUDY_DIR / "figs"
STUDY_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "Manitoba", "Manitoba SK", "Michigan", "Minnesota", "New-York", "Ontario",
    "Quebec AT", "Quebec B5D.B31L", "Quebec D4Z", "Quebec D5A", "Quebec H4Z",
    "Quebec H9A", "Quebec P33C", "Quebec Q4C", "Quebec X2Y",
]

# Must match training
SEQ_LEN    = 576
LABEL_LEN  = 72
PRED_LEN   = 1
SPLIT      = (0.7, 0.1, 0.2)

FREQ_5MIN  = True     # for labels only
PER_HOUR   = 12
PERSIST_LAG = 1       # t-1, 5 minutes
CAL_SLICE  = 500      # points from start of test to pick best inverse mapping
MAX_LAGS_ACF = 144    # up to 12 hours on 5-min data
MAX_LAGS_PACF = 48

# Cross-correlation heatmap across targets (set False to skip)
DO_PAIRWISE_XCORR = True

# ======================================================

def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

# ----------------- UTIL: load/shape preds -----------------
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
    if arr is None:
        return None
    s = np.squeeze(arr)
    if s.ndim == 0:
        return s.reshape(-1, 1)
    if s.ndim == 1:
        return s.reshape(-1, 1)
    if s.ndim == 2:
        return s if s.shape[0] >= s.shape[1] else s.T
    if s.ndim == 3:
        axes = np.array(s.shape)
        t_ax = int(np.argmax(axes))
        s2 = np.moveaxis(s, t_ax, 0)  # time first
        W = s2.shape[0]; C = int(np.prod(s2.shape[1:]))
        return s2.reshape(W, C)
    raise ValueError(f"Unsupported pred.npy ndim={s.ndim}")

def choose_col_by_corr(pred_WC: np.ndarray, y_true_raw: np.ndarray) -> tuple[np.ndarray, int, float]:
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

# ----------------- DENORMALIZATION -----------------
def affine_from_true(model_true: np.ndarray, actual_true: np.ndarray) -> tuple[float, float]:
    L = min(model_true.size, actual_true.size)
    x = model_true[:L].reshape(-1, 1)
    y = actual_true[:L].reshape(-1, 1)
    X = np.hstack([x, np.ones_like(x)])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a = float(beta[0, 0]); b = float(beta[1, 0])
        if not np.isfinite(a) or not np.isfinite(b):
            a, b = 1.0, 0.0
    except Exception:
        a, b = 1.0, 0.0
    return a, b

def pick_inverse_mapping(y_true_cal, y_pred_model_cal, train_mean, train_std, model_true_cal=None):
    # identity
    y_id = y_pred_model_cal
    mae_id = mae(y_true_cal, y_id)
    best = ("identity", 1.0, 0.0, mae_id)

    # std inverse
    if np.isfinite(train_std) and train_std > 1e-12:
        y_std = y_pred_model_cal * train_std + train_mean
        mae_std = mae(y_true_cal, y_std)
        if mae_std < best[3]:
            best = ("std_inv", train_std, train_mean, mae_std)

    # affine using model true
    if model_true_cal is not None and model_true_cal.size >= 10:
        a, b = affine_from_true(model_true_cal, y_true_cal)
        y_aff = y_pred_model_cal * a + b
        mae_aff = mae(y_true_cal, y_aff)
        if mae_aff < best[3]:
            best = ("affine", a, b, mae_aff)

    name, a, b, _ = best
    return name, a, b

# ----------------- BASELINES -----------------
def build_persistence(y: pd.Series, idx: np.ndarray) -> np.ndarray:
    persist = np.empty(len(idx), dtype=float)
    for k, t in enumerate(idx):
        persist[k] = float(y.iloc[t-1]) if t-1 >= 0 else float(y.iloc[0])
    return persist

# ----------------- STATS: ACF/PACF -----------------
def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    var = np.var(x)
    if var < 1e-12: return np.zeros(max_lag+1)
    corr = np.correlate(x, x, mode="full")
    mid = len(corr) // 2
    acf_vals = corr[mid:mid+max_lag+1] / (var * len(x))
    return acf_vals

def pacf_yw(x: np.ndarray, max_lag: int) -> np.ndarray:
    # Yule-Walker PACF
    r = acf(x, max_lag)
    pacf_vals = np.zeros(max_lag+1)
    pacf_vals[0] = 1.0
    phi = np.zeros((max_lag+1, max_lag+1))
    sigma = np.zeros(max_lag+1)
    sigma[0] = r[0]
    for k in range(1, max_lag+1):
        num = r[k] - np.dot(phi[1:k, k-1], r[1:k][::-1])
        den = sigma[k-1]
        phi[k, k] = num / den if abs(den) > 1e-12 else 0.0
        for j in range(1, k):
            phi[j, k] = phi[j, k-1] - phi[k, k] * phi[k-j, k-1]
        sigma[k] = sigma[k-1] * (1 - phi[k, k]**2)
        pacf_vals[k] = phi[k, k]
    return pacf_vals

# ----------------- STATS: SPECTRUM -----------------
def periodogram(x: np.ndarray, fs: float = 12.0):
    """
    Simple periodogram (no windowing). fs=12 means 12 samples/hour for 5-min data.
    Returns frequencies (cycles/hour) and power.
    """
    n = len(x)
    x = x - np.mean(x)
    fft = np.fft.rfft(x)
    ps = (np.abs(fft)**2) / n
    freqs = np.fft.rfftfreq(n, d=1.0/fs)  # cycles/hour
    return freqs, ps

# ----------------- STATS: ENTROPY -----------------
def approx_entropy(U: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """Approximate Entropy (ApEn). Small sample-friendly; simplistic implementation."""
    U = np.asarray(U, dtype=float)
    n = len(U)
    if n <= m + 1: return np.nan
    if r is None:
        r = 0.2 * np.std(U) if np.std(U) > 0 else 0.2
    def _phi(m):
        x = np.array([U[i:i+m] for i in range(n - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None, :] - x[None, :, :]), axis=2) <= r, axis=0) / (n - m + 1)
        return np.sum(np.log(C + 1e-12)) / (n - m + 1)
    return _phi(m) - _phi(m+1)

def permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1) -> float:
    """Permutation Entropy (Bandt & Pompe)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < (m - 1) * tau + 1:
        return np.nan
    patterns = {}
    for i in range(n - (m - 1) * tau):
        window = x[i:i + m * tau:tau]
        key = tuple(np.argsort(window))
        patterns[key] = patterns.get(key, 0) + 1
    counts = np.array(list(patterns.values()), dtype=float)
    p = counts / counts.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    Hmax = np.log(np.math.factorial(m))
    return H / Hmax  # normalized [0,1]

# ----------------- VISUALS -----------------
def save_skill_bar(df: pd.DataFrame, outpath: Path):
    plt.figure(figsize=(12,5))
    order = df.sort_values("skill_vs_persistence", ascending=False)
    plt.bar(order["target"], order["skill_vs_persistence"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Skill vs Persistence (1 - MAE_model/MAE_persist)")
    ttl = "Skill vs Persistence (t+1, 5-min)"
    plt.title(ttl)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def save_residual_boxplot(residuals_dict: dict[str, np.ndarray], outpath: Path):
    plt.figure(figsize=(12,6))
    targets = list(residuals_dict.keys())
    data = [residuals_dict[t] for t in targets]
    plt.boxplot(data, showfliers=False, labels=targets)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Residual ($/MWh)")
    plt.title("Residual Distribution per Region (no outliers)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def save_acf_pacf_overlay(good_series: np.ndarray, bad_series: np.ndarray, labels: tuple[str, str], out_prefix: Path):
    # ACF
    acf_g = acf(good_series, MAX_LAGS_ACF)
    acf_b = acf(bad_series,  MAX_LAGS_ACF)
    x = np.arange(MAX_LAGS_ACF+1)
    plt.figure(figsize=(10,4))
    plt.plot(x, acf_g, label=f"ACF {labels[0]}")
    plt.plot(x, acf_b, label=f"ACF {labels[1]}")
    plt.xlabel("Lag (5-min steps)"); plt.ylabel("Autocorr")
    plt.title("ACF Comparison")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_acf.png"), dpi=160); plt.close()

    # PACF
    pacf_g = pacf_yw(good_series, MAX_LAGS_PACF)
    pacf_b = pacf_yw(bad_series,  MAX_LAGS_PACF)
    x = np.arange(MAX_LAGS_PACF+1)
    plt.figure(figsize=(10,4))
    plt.plot(x, pacf_g, label=f"PACF {labels[0]}")
    plt.plot(x, pacf_b, label=f"PACF {labels[1]}")
    plt.xlabel("Lag (5-min steps)"); plt.ylabel("Partial Autocorr")
    plt.title("PACF Comparison")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_prefix.with_name(out_prefix.name + "_pacf.png"), dpi=160); plt.close()

def save_periodogram_overlay(good_series: np.ndarray, bad_series: np.ndarray, labels: tuple[str,str], outpath: Path):
    fs = 12.0  # samples/hour
    f1, p1 = periodogram(good_series, fs=fs)
    f2, p2 = periodogram(bad_series,  fs=fs)
    plt.figure(figsize=(10,4))
    plt.plot(f1, p1, label=f"Spectrum {labels[0]}")
    plt.plot(f2, p2, label=f"Spectrum {labels[1]}")
    plt.xlabel("Frequency (cycles/hour)"); plt.ylabel("Power")
    plt.title("Periodogram Comparison")
    plt.legend(); plt.tight_layout()
    plt.savefig(outpath, dpi=160); plt.close()

def save_daily_mae_plot(daily_df: pd.DataFrame, target: str, outdir: Path):
    plt.figure(figsize=(12,4))
    plt.plot(daily_df["date"], daily_df["mae_model"], label="PatchTST")
    plt.plot(daily_df["date"], daily_df["mae_persist"], label="Persistence")
    plt.title(f"{target}  Daily MAE")
    plt.xlabel("Date"); plt.ylabel("MAE ($/MWh)")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"{target.replace(' ', '_')}_daily_mae.png", dpi=150)
    plt.close()

def save_scatter_pred_vs_actual(ts, y_true, y_pred, target: str, outdir: Path):
    # Sample evenly to avoid gigantic plots
    n = len(y_true)
    step = max(1, n // 5000)
    plt.figure(figsize=(5,5))
    plt.scatter(y_true[::step], y_pred[::step], s=4, alpha=0.4)
    plt.xlabel("Actual ($/MWh)"); plt.ylabel("Predicted ($/MWh)")
    plt.title(f"{target}  Pred vs Actual")
    plt.tight_layout()
    plt.savefig(outdir / f"{target.replace(' ', '_')}_scatter_pred_actual.png", dpi=150)
    plt.close()

def save_skill_vs_vol_quartiles(df_stream: pd.DataFrame, target: str, outdir: Path):
    # rolling volatility (1h window = 12 samples)
    df = df_stream.copy()
    df["ret"] = df["y_true"].diff()
    df["vol"] = df["ret"].rolling(12, min_periods=6).std()
    df = df.dropna(subset=["vol"])
    q = df["vol"].quantile([0.25, 0.5, 0.75]).to_dict()
    def qlabel(v):
        if v <= q[0.25]: return "Q1 (low)"
        if v <= q[0.5]:  return "Q2"
        if v <= q[0.75]: return "Q3"
        return "Q4 (high)"
    df["vol_q"] = df["vol"].apply(qlabel)
    grp = df.groupby("vol_q")
    skill = []
    for name, g in grp:
        m = mae(g["y_true"].values, g["y_pred"].values)
        p = mae(g["y_true"].values, g["persist"].values)
        skill.append((name, 1.0 - m/max(p,1e-12)))
    # sort quartiles Q1..Q4
    order = ["Q1 (low)","Q2","Q3","Q4 (high)"]
    skill = [s for s in skill if s[0] in order]
    skill.sort(key=lambda x: order.index(x[0]))
    plt.figure(figsize=(6,4))
    plt.bar([s[0] for s in skill], [s[1] for s in skill])
    plt.ylim(min(-1.0, min([s[1] for s in skill]) - 0.05), 1.0)
    plt.title(f"{target}  Skill by Volatility Quartile")
    plt.ylabel("Skill vs Persistence")
    plt.tight_layout()
    plt.savefig(outdir / f"{target.replace(' ', '_')}_skill_by_vol_quartile.png", dpi=150)
    plt.close()

# ----------------- MAIN PIPELINE -----------------
def main():
    per_target_rows = []
    residuals_for_box = {}
    per_target_series_cache = {}  # for ACF/spectrum comparisons
    per_target_daily = {}
    per_target_denorm = {}

    for target in TARGETS:
        try:
            per_csv = DATA_ROOT / f"{target.replace(' ', '_')}_timeonly_clean.csv"
            if not per_csv.exists():
                print(f"[WARN] Missing per-target CSV: {per_csv}")
                continue
            y = load_target_series(per_csv, target)
            N = len(y)
            num_train = int(N * SPLIT[0]); num_val = int(N * SPLIT[1])
            train_mean = float(y.iloc[:num_train].mean())
            train_std  = float(y.iloc[:num_train].std(ddof=0)) if num_train > 1 else float(y.std(ddof=0))
            if not np.isfinite(train_std) or train_std < 1e-12: train_std = 1.0

            resdir = latest_results_dir_for(target)
            pred = np.load(resdir / "pred.npy")
            true_path = resdir / "true.npy"
            model_true = np.load(true_path) if true_path.exists() else None

            pred_WC = to_time_channels(pred)
            true_WC = to_time_channels(model_true) if model_true is not None else None
            W = pred_WC.shape[0]

            test_start = num_train + num_val
            if test_start + W + PRED_LEN > N:
                test_start = max(SEQ_LEN + LABEL_LEN, N - (W + PRED_LEN))
            idx = np.arange(test_start, min(test_start + W, N))
            ts  = y.index[idx]
            y_true_raw_full = y.iloc[idx].to_numpy(dtype=float)

            # pick pred col
            y_pred_model_full, picked_j, picked_corr = choose_col_by_corr(pred_WC[:len(idx), :], y_true_raw_full)
            model_true_col_full = None
            if true_WC is not None and picked_j < true_WC.shape[1]:
                model_true_col_full = true_WC[:len(idx), picked_j].astype(float)

            # calibration
            Lcal = min(CAL_SLICE, y_true_raw_full.size)
            y_true_cal = y_true_raw_full[:Lcal]
            y_pred_model_cal = y_pred_model_full[:Lcal]
            model_true_cal = model_true_col_full[:Lcal] if model_true_col_full is not None else None
            inv_name, a_inv, b_inv = pick_inverse_mapping(y_true_cal, y_pred_model_cal, train_mean, train_std, model_true_cal)

            # denormalize
            y_pred_raw_full = y_pred_model_full * a_inv + b_inv

            # persistence
            persist_full = build_persistence(y, idx)

            # align
            L = min(len(y_true_raw_full), len(y_pred_raw_full), len(persist_full))
            y_true_raw = y_true_raw_full[:L]
            y_pred_raw = y_pred_raw_full[:L]
            persist    = persist_full[:L]
            ts         = ts[:L]

            # metrics
            mae_model = mae(y_true_raw, y_pred_raw)
            mae_pers  = mae(y_true_raw, persist)
            skill_vs_pers = 1.0 - (mae_model / max(mae_pers, 1e-12))

            # residuals
            residuals = y_true_raw - y_pred_raw
            residuals_for_box[target] = residuals

            # daily MAE & win rate
            df_stream = pd.DataFrame({"ts": pd.to_datetime(ts),
                                      "y_true": y_true_raw,
                                      "y_pred": y_pred_raw,
                                      "persist": persist})
            df_stream["date"] = df_stream["ts"].dt.normalize()
            g = df_stream.groupby("date", sort=True)
            daily = pd.DataFrame({
                "date": g["ts"].min().values,
                "mae_model": g.apply(lambda d: mae(d["y_true"].values, d["y_pred"].values)).values,
                "mae_persist": g.apply(lambda d: mae(d["y_true"].values, d["persist"].values)).values,
            }).sort_values("date")
            wins = (daily["mae_model"] < daily["mae_persist"]).sum()
            total_days = len(daily)
            win_rate = wins / total_days if total_days > 0 else 0.0
            daily.to_csv(STUDY_DIR / f"{target.replace(' ', '_')}_daily_mae.csv", index=False)
            save_daily_mae_plot(daily, target, FIG_DIR)
            save_scatter_pred_vs_actual(ts, y_true_raw, y_pred_raw, target, FIG_DIR)
            save_skill_vs_vol_quartiles(df_stream, target, FIG_DIR)

            # series cache for ACF/spectrum
            per_target_series_cache[target] = y_true_raw  # actuals on test window
            per_target_daily[target] = daily
            per_target_denorm[target] = dict(inv=inv_name, a=a_inv, b=b_inv, picked_col=picked_j, corr=picked_corr)

            per_target_rows.append({
                "target": target,
                "n_eval_points": int(L),
                "picked_pred_column": int(picked_j),
                "corr_pred_vs_y": float(picked_corr),
                "inverse_mapping": inv_name,
                "inv_a": float(a_inv),
                "inv_b": float(b_inv),
                "overall_mae_model": mae_model,
                "overall_mae_persist": mae_pers,
                "skill_vs_persistence": skill_vs_pers,
                "resid_mean": float(np.mean(residuals)),
                "resid_std": float(np.std(residuals)),
                "resid_p95": float(np.percentile(residuals, 95)),
                "days": total_days,
                "win_days": int(wins),
                "win_rate": float(win_rate),
            })

            print(f"[OK] {target}: MAE={mae_model:.3f} | Pers={mae_pers:.3f} | Skill={skill_vs_pers:.3f} | "
                  f"wins={wins}/{total_days} ({win_rate:.1%}) | inv={inv_name} a={a_inv:.3g} b={b_inv:.3g} | col={picked_j} corr={picked_corr:.3f}")

        except Exception as e:
            print(f"[WARN] Study failed for {target}: {e}")

    if not per_target_rows:
        print("No targets processed; exiting.")
        return

    # ---- Save per-target summary table
    df_sum = pd.DataFrame(per_target_rows).sort_values(["skill_vs_persistence","overall_mae_model"], ascending=[False, True])
    df_sum.to_csv(STUDY_DIR / "summary_per_target.csv", index=False)

    # ---- Global plots
    save_skill_bar(df_sum[["target","skill_vs_persistence"]], FIG_DIR / "skill_bar.png")
    save_residual_boxplot(residuals_for_box, FIG_DIR / "residual_boxplot.png")

    # ---- ACF/PACF and Spectrum: compare best vs worst
    best_row  = df_sum.iloc[0]
    worst_row = df_sum.iloc[-1]
    best_t = best_row["target"]; worst_t = worst_row["target"]
    try:
        save_acf_pacf_overlay(per_target_series_cache[best_t], per_target_series_cache[worst_t],
                              (best_t, worst_t), FIG_DIR / "acf_pacf_compare")
        save_periodogram_overlay(per_target_series_cache[best_t], per_target_series_cache[worst_t],
                                 (best_t, worst_t), FIG_DIR / "spectrum_compare.png")
    except Exception as e:
        print(f"[WARN] ACF/PACF/Spectrum compare failed: {e}")

    # ---- Entropy & Volatility metrics per target
    rows_entropy = []
    for t, y_series in per_target_series_cache.items():
        try:
            apen = approx_entropy(y_series, m=2)
            pe   = permutation_entropy(y_series, m=3, tau=1)
            vol  = float(pd.Series(y_series).diff().rolling(12, min_periods=6).std().mean())  # avg 1h rolling vol
        except Exception:
            apen, pe, vol = np.nan, np.nan, np.nan
        rows_entropy.append({"target": t, "approx_entropy_m2": apen, "perm_entropy_m3": pe, "avg_1h_rolling_vol": vol})
    df_entropy = pd.DataFrame(rows_entropy)
    df_entropy.to_csv(STUDY_DIR / "complexity_volatility_metrics.csv", index=False)

    # Merge entropy/vol with summary and save a paper-ready table
    df_paper = df_sum.merge(df_entropy, on="target", how="left")
    df_paper = df_paper.sort_values(["skill_vs_persistence","overall_mae_model"], ascending=[False, True])
    df_paper.to_csv(STUDY_DIR / "paper_table_metrics.csv", index=False)

    # ---- Pairwise cross-correlation heatmap (optional)
    if DO_PAIRWISE_XCORR:
        try:
            # Build aligned matrix (targets x time)
            common_len = min(len(v) for v in per_target_series_cache.values())
            targets_order = list(per_target_series_cache.keys())
            M = np.vstack([per_target_series_cache[t][-common_len:] for t in targets_order])
            # Pearson corr across targets
            C = np.corrcoef(M)
            # Simple heatmap (matplotlib)
            plt.figure(figsize=(10,8))
            plt.imshow(C, aspect='auto', vmin=-1, vmax=1)
            plt.colorbar(label="Correlation")
            plt.xticks(range(len(targets_order)), targets_order, rotation=60, ha="right")
            plt.yticks(range(len(targets_order)), targets_order)
            plt.title("Cross-correlation of Actual Price Series (Test Window)")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "xcorr_heatmap.png", dpi=160)
            plt.close()
            # Save as CSV too
            pd.DataFrame(C, index=targets_order, columns=targets_order).to_csv(STUDY_DIR / "xcorr_matrix.csv")
        except Exception as e:
            print(f"[WARN] Cross-correlation heatmap failed: {e}")

    # ---- Also dump per-target denorm choices
    pd.DataFrame.from_dict(per_target_denorm, orient="index").to_csv(STUDY_DIR / "denorm_choices.csv")

    print("\n=== STUDY COMPLETE ===")
    print("Per-target summary:      ", STUDY_DIR / "summary_per_target.csv")
    print("Paper metrics table:     ", STUDY_DIR / "paper_table_metrics.csv")
    print("Complexity/vol metrics:  ", STUDY_DIR / "complexity_volatility_metrics.csv")
    if DO_PAIRWISE_XCORR:
        print("XCorr matrix:            ", STUDY_DIR / "xcorr_matrix.csv")
    print("Figures saved under:     ", FIG_DIR)

if __name__ == "__main__":
    main()
