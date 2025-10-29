#!/usr/bin/env python3
# canada_timeonly_summary.py
# Re-evaluate PatchTST runs and fairly compare against trivial baselines with proper de-normalization.
# - Robust to pred.npy shapes (1D/2D/3D).
# - Chooses best inverse mapping among: identity, train-std inverse, affine-from-true (if available).
# - Picks best prediction column by correlation.
# - Saves per-target stream CSV/PNG and a global summary CSV.
# - Also saves a random-day plot per target.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ---------- CONFIG ----------
MASTER_CSV = Path("/home/omaralrefai/dev/PatchTST/.dataset/canada/canada_realtime_ENGY_2010_2025.csv")
DATA_ROOT  = MASTER_CSV.parent / "per_target_timeonly_clean"   # *_timeonly_clean.csv live here
RESULTS_ROOT = Path("results")

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

# 5-minute grid baselines
PER_HOUR   = 12
DAILY_LAG  = 24 * PER_HOUR  # 288

SUMMARY_CSV = RESULTS_ROOT / "CANADA_timeonly_t+1_summary_reeval_denorm.csv"
PLOTS_DIR   = RESULTS_ROOT / "day_plots"
MAKE_DAY_PLOTS = True
CAL_SLICE = 500  # number of earliest test points to decide best inverse mapping
# -----------------------------

def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mape(a, b, eps=1.0): return float(np.mean(np.abs(a - b) / np.maximum(eps, np.abs(a))) * 100.0)

def latest_results_dir_for(target: str) -> Path:
    prefix = f"PTST_CAN_timeonly_t+1_{target.replace(' ', '_')}_PatchTST_custom_ftM_sl{SEQ_LEN}_ll{LABEL_LEN}_pl{PRED_LEN}_"
    cands = sorted([p for p in RESULTS_ROOT.glob(f"{prefix}*") if (p / "pred.npy").exists()],
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        return cands[0]
    fb = sorted(
        [p for p in RESULTS_ROOT.rglob("*") if (p / "pred.npy").exists() and target.replace(' ', '_') in p.name],
        key=lambda p: p.stat().st_mtime, reverse=True
    )
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
    # More dims unexpected here
    raise ValueError(f"Unsupported ndim={s.ndim}")

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

def best_inverse_mapping(y_true_raw_cal: np.ndarray,
                         y_pred_model_cal: np.ndarray,
                         train_mean: float,
                         train_std: float,
                         model_true_cal: np.ndarray | None) -> tuple[str, dict, np.ndarray]:
    """
    Try three mappings on the calibration window:
      IDENTITY: y_hat = y_pred_model
      STD_INV:  y_hat = y_pred_model * std_train + mean_train
      AFFINE:   y_hat = a*y_pred_model + b  (if model_true_cal provided, fit a,b on it vs y_true_raw_cal)
    Return (chosen_name, params_dict, y_hat_cal).
    """
    candidates = {}

    # Identity
    y_id = y_pred_model_cal
    candidates["identity"] = {"mae": mae(y_true_raw_cal, y_id), "params": {}, "yhat": y_id}

    # Standard inverse if std nonzero
    if np.isfinite(train_std) and train_std > 1e-12:
        y_std = y_pred_model_cal * train_std + train_mean
        candidates["std_inv"] = {"mae": mae(y_true_raw_cal, y_std),
                                 "params": {"a": train_std, "b": train_mean}, "yhat": y_std}

    # Affine from true (if provided)
    if model_true_cal is not None:
        a, b = affine_from_true(model_true_cal, y_true_raw_cal)
        y_aff = a * y_pred_model_cal + b
        candidates["affine"] = {"mae": mae(y_true_raw_cal, y_aff), "params": {"a": a, "b": b}, "yhat": y_aff}

    # Pick best (smallest MAE)
    best_name, best = min(candidates.items(), key=lambda kv: kv[1]["mae"])
    return best_name, best["params"], best["yhat"]

def build_baselines(y: pd.Series, idx: np.ndarray, daily_lag: int) -> tuple[np.ndarray, np.ndarray]:
    persist = np.empty(len(idx), dtype=float)
    daily   = np.empty(len(idx), dtype=float)
    for k, t in enumerate(idx):
        last = float(y.iloc[t-1]) if t-1 >= 0 else float(y.iloc[0])
        dlag = float(y.iloc[t-daily_lag]) if t-daily_lag >= 0 else last
        persist[k] = last
        daily[k]   = dlag
    return persist, daily

def plot_random_day(ts, y_true, y_pred, pers, daily, target_name, outdir: Path):
    ts = pd.to_datetime(ts)
    dates = ts.normalize()
    uniq = dates.unique()
    if len(uniq) == 0:
        return
    day = pd.Timestamp(uniq[np.random.randint(len(uniq))])
    mask = (dates == day)
    if mask.sum() < 12:
        return
    t = ts[mask]
    fig = plt.figure(figsize=(12,4))
    plt.plot(t, y_true[mask], label="Actual")
    plt.plot(t, y_pred[mask], label="PatchTST")
    plt.plot(t, pers[mask], label="Persistence (t-1)")
    plt.plot(t, daily[mask], label="Daily (t-288)")
    plt.title(f"{target_name} — random day slice: {day.date()}")
    plt.xlabel("Time"); plt.ylabel("ENGY ($/MWh)")
    plt.legend(); plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"{target_name.replace(' ', '_')}_random_day.png", dpi=150)
    plt.close()

def main():
    RESULTS_ROOT.mkdir(exist_ok=True)
    rows = []

    for target in TARGETS:
        try:
            per_csv = DATA_ROOT / f"{target.replace(' ', '_')}_timeonly_clean.csv"
            if not per_csv.exists():
                raise FileNotFoundError(f"Missing per-target CSV: {per_csv}")

            # Load full target series
            y = load_target_series(per_csv, target)
            N = len(y)

            # Training mean/std from TRAIN split only (what scalers typically use)
            num_train = int(N * SPLIT[0])
            num_val   = int(N * SPLIT[1])
            train_mean = float(y.iloc[:num_train].mean())
            train_std  = float(y.iloc[:num_train].std(ddof=0)) if num_train > 1 else float(y.std(ddof=0))
            if not np.isfinite(train_std) or train_std < 1e-12:
                train_std = 1.0  # safe fallback

            # Locate run & load pred/true
            resdir = latest_results_dir_for(target)
            pred = np.load(resdir / "pred.npy")
            true_path = resdir / "true.npy"
            model_true = np.load(true_path) if true_path.exists() else None

            pred_WC = to_time_channels(pred)
            true_WC = to_time_channels(model_true) if model_true is not None else None
            W = pred_WC.shape[0]

            # Test window alignment
            test_start = num_train + num_val
            if test_start + W + PRED_LEN > N:
                test_start = max(SEQ_LEN + LABEL_LEN, N - (W + PRED_LEN))
            idx = np.arange(test_start, min(test_start + W, N))
            y_true_raw_full = y.iloc[idx].to_numpy(dtype=float)

            # Choose the most likely pred column by corr vs raw truth
            y_pred_model_full, picked_j, picked_corr_pred = choose_col_by_corr(pred_WC[:len(idx), :], y_true_raw_full)

            # If we have model-space truth, use SAME column from it for calibration
            model_true_col_full = None
            if true_WC is not None and picked_j < true_WC.shape[1]:
                model_true_col_full = true_WC[:len(idx), picked_j].astype(float)

            # Calibration slice (front of the test window)
            Lcal = min(CAL_SLICE, y_true_raw_full.size)
            y_true_cal = y_true_raw_full[:Lcal]
            y_pred_model_cal = y_pred_model_full[:Lcal]
            model_true_cal = model_true_col_full[:Lcal] if model_true_col_full is not None else None

            # Pick best inverse mapping
            best_name, params, yhat_cal = best_inverse_mapping(
                y_true_cal, y_pred_model_cal, train_mean, train_std, model_true_cal
            )

            # Apply chosen inverse to FULL window
            if best_name == "identity":
                y_pred_raw_full = y_pred_model_full
                a, b = 1.0, 0.0
            elif best_name == "std_inv":
                a, b = params.get("a", train_std), params.get("b", train_mean)
                y_pred_raw_full = y_pred_model_full * a + b
            else:  # "affine"
                a, b = params.get("a", 1.0), params.get("b", 0.0)
                y_pred_raw_full = y_pred_model_full * a + b

            # Baselines on same indices
            persist_full, daily_full = build_baselines(y, idx, DAILY_LAG)

            # Align lengths
            L = min(len(y_true_raw_full), len(y_pred_raw_full), len(persist_full), len(daily_full))
            y_true_raw = y_true_raw_full[:L]
            y_pred_raw = y_pred_raw_full[:L]
            persist    = persist_full[:L]
            daily      = daily_full[:L]
            ts         = y.index[idx][:L]

            # Metrics
            m_model = {"MAE": mae(y_true_raw, y_pred_raw), "RMSE": rmse(y_true_raw, y_pred_raw), "MAPE%": mape(y_true_raw, y_pred_raw, 1.0)}
            m_pers  = {"MAE": mae(y_true_raw, persist),    "RMSE": rmse(y_true_raw, persist),    "MAPE%": mape(y_true_raw, persist, 1.0)}
            m_daily = {"MAE": mae(y_true_raw, daily),      "RMSE": rmse(y_true_raw, daily),      "MAPE%": mape(y_true_raw, daily, 1.0)}
            skill_vs_persist = 1.0 - (m_model["MAE"] / max(1e-12, m_pers["MAE"]))

            # Save stream + plot
            out_stream = pd.DataFrame({
                "date": ts,
                "y_true": y_true_raw,
                "PatchTST_t+1": y_pred_raw,
                "Persistence_t+1": persist,
                "Daily_t+1(t-288)": daily
            })
            out_stream.to_csv(resdir / f"{target.replace(' ', '_')}_tplus1_stream_reeval_denorm.csv", index=False)

            plt.figure(figsize=(12,4))
            plt.plot(ts, y_true_raw, label="Actual")
            plt.plot(ts, y_pred_raw, label=f"PatchTST t+1 (col {picked_j}, inv={best_name}, a={a:.3g}, b={b:.3g})")
            plt.plot(ts, persist, label="Persistence (t-1)")
            plt.plot(ts, daily, label="Daily (t-288)")
            plt.title(f"{target} — t+1 (5-min) re-eval (denormalized)")
            plt.xlabel("Time"); plt.ylabel("ENGY ($/MWh)")
            plt.legend(); plt.tight_layout()
            plt.savefig(resdir / f"{target.replace(' ', '_')}_tplus1_stream_reeval_denorm.png", dpi=140)
            plt.close()

            if MAKE_DAY_PLOTS:
                plot_random_day(ts, y_true_raw, y_pred_raw, persist, daily, target, PLOTS_DIR)

            print(f"[OK] {target}: MAE={m_model['MAE']:.4f} | Pers={m_pers['MAE']:.4f} | Daily={m_daily['MAE']:.4f} | Skill={skill_vs_persist:.3f} | "
                  f"picked_col={picked_j} corr_pred={picked_corr_pred:.3f} | inv={best_name} a={a:.4g} b={b:.4g}")

            rows.append({
                "target": target,
                "results_dir": str(resdir),
                "n_eval_points": int(L),
                "picked_pred_column": int(picked_j),
                "corr_pred_vs_y": float(picked_corr_pred),
                "inverse_mapping": best_name,
                "inv_a": float(a),
                "inv_b": float(b),
                "mae_patch": m_model["MAE"],
                "rmse_patch": m_model["RMSE"],
                "mape_patch_pct": m_model["MAPE%"],
                "mae_persistence": m_pers["MAE"],
                "rmse_persistence": m_pers["RMSE"],
                "mae_daily": m_daily["MAE"],
                "rmse_daily": m_daily["RMSE"],
                "skill_vs_persistence": skill_vs_persist,
                "better_than_persistence": int(m_model["MAE"] < m_pers["MAE"]),
            })

        except Exception as e:
            print(f"[WARN] Re-eval failed for {target}: {e}")

    if not rows:
        print("No results to summarize."); return

    df = pd.DataFrame(rows)
    df["rank_mae_patch"]  = df["mae_patch"].rank(method="min")
    df["rank_skill"]      = (-df["skill_vs_persistence"]).rank(method="min")
    df = df.sort_values(["mae_patch", "skill_vs_persistence"], ascending=[True, False])
    df.to_csv(SUMMARY_CSV, index=False)

    print("\n=== SUMMARY (t+1, 5-min) — denormalized to $/MWh ===")
    print(df[["target","n_eval_points","mae_patch","mae_persistence","mae_daily","skill_vs_persistence","inverse_mapping","inv_a","inv_b","picked_pred_column","corr_pred_vs_y"]].to_string(index=False))
    print("\nSaved summary CSV:", SUMMARY_CSV)
    if MAKE_DAY_PLOTS:
        print("Random day plots saved to:", PLOTS_DIR)

if __name__ == "__main__":
    main()
