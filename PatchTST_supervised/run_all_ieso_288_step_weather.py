
#!/usr/bin/env python3
# batch_canada_da288_timeonly_gpu.py
# Train & evaluate PatchTST per area to predict NEXT 24h at 5-min resolution (288-step horizon).
# Uses engineered time/weather features.

import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========== USER CONFIG ==========
DATA_DIR = Path("/home/omaralrefai/dev/PatchTST/.dataset/canada").resolve()

# Only these files will be considered
WHITELIST_FILES = [
    "manitoba_features.csv",
    "new-york_features.csv",
    "quebec_q4c_features.csv",
    "ontario_features.csv",
    "quebec_p33c_features.csv",
    "manitoba_sk_features.csv",
]

# We'll write cleaned per-area CSVs (with numeric features + exact cols) here
DATA_ROOT  = DATA_DIR / "per_area_features_clean"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

RUN_LONGEXP  = Path("run_longExp.py")
RESULTS_ROOT = Path("results"); RESULTS_ROOT.mkdir(exist_ok=True)

RUN_TRAIN = True

FREQ       = "5min"
PER_HOUR   = 12
DAILY_LAG  = 24 * PER_HOUR  # 288

# features='M' => X excludes target; we pass 7 engineered inputs as channels
ENC_IN     = 7   # DATE_ORD, hour, day_of_week_idx, interval, month, windspeed_10m, temperature_2m
DEC_IN     = 7
C_OUT      = 1   # price

SEQ_LEN    = 576    # two days of 5-min history
LABEL_LEN  = 72
PRED_LEN   = 288    # <<< one full day at 5-min resolution

D_MODEL    = 512
N_HEADS    = 8
E_LAYERS   = 3
D_LAYERS   = 2
D_FF       = 2048
DROPOUT    = 0.05
BATCH      = 32
LR         = 1e-4
EPOCHS     = 30
PATIENCE   = 8
PATCH_LEN  = 48
STRIDE     = 24
USE_GPU    = 1
NUM_WORKERS= 0
EMBED      = "fixed"

REVIN      = 0
AFFINE     = 0
SUBTRACT_LAST = 0
USE_AMP    = False

SPLIT      = (0.7, 0.1, 0.2)
EPS_MAPE   = 1.0

SUMMARY_CSV = RESULTS_ROOT / "CANADA_features_DA288_summary_gpu.csv"
# =================================

def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mape(a, b, eps=1.0): return float(np.mean(np.abs(a - b) / np.maximum(eps, np.abs(a))) * 100.0)

def discover_area_files(root: Path):
    found, missing = [], []
    for fname in WHITELIST_FILES:
        p = (root / fname)
        (found if p.is_file() else missing).append(p if p.is_file() else fname)
    if missing:
        print("[WARN] Missing whitelisted files (skipped):")
        for m in missing:
            print("   -", m)
    return sorted(found)

def area_name_from_path(p: Path) -> str:
    base = p.stem
    if base.endswith("_features"):
        base = base[:-9]
    return base

def build_per_area_csv(src_csv: Path) -> Path:
    """
    Create per-area CSV with exact numeric columns for PatchTST:
      date, DATE_ORD, hour, day_of_week_idx, interval, month, windspeed_10m, temperature_2m, price
    """
    df = pd.read_csv(src_csv)
    df.columns = [c.strip().lower() for c in df.columns]

    req = {"timestamp","hour","day_of_week","interval","month","windspeed_10m","temperature_2m","price"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"{src_csv.name}: missing required columns: {', '.join(sorted(missing))}")

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{src_csv.name}: unparsable timestamps found.")

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

    out = out[(out["interval"]>=1) & (out["interval"]<=12)]
    out = out[out["hour"].between(0,23)]
    out = out[out["month"].between(1,12)]
    out = out.dropna().sort_values("date").reset_index(drop=True)

    area = area_name_from_path(src_csv)
    out_path = DATA_ROOT / f"{area}_clean.csv"
    out.to_csv(out_path, index=False)
    return out_path

def run_train_for_area(per_area_csv: Path, area: str):
    model_id = f"PTST_CAN_features_DA288_{area}"
    cmd = [
        sys.executable, str(RUN_LONGEXP),
        "--is_training", "1",
        "--model_id", model_id,
        "--model", "PatchTST",
        "--data", "custom",
        "--root_path", str(DATA_ROOT),
        "--data_path", per_area_csv.name,
        "--features", "M",
        "--target", "price",
        "--seq_len", str(SEQ_LEN),
        "--label_len", str(LABEL_LEN),
        "--pred_len", str(PRED_LEN),            # <<< 288 steps
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
    ]
    if USE_AMP:
        cmd.append("--use_amp")

    print(f"\n[TRAIN] {area} -> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

def find_latest_results_dir(model_id_prefix: str) -> Path:
    prefix = f"{model_id_prefix}_PatchTST_custom_ftM_sl{SEQ_LEN}_ll{LABEL_LEN}_pl{PRED_LEN}_"
    cands = sorted(
        [p for p in RESULTS_ROOT.glob(f"{prefix}*") if (p / "pred.npy").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not cands:
        any_cands = sorted((p for p in RESULTS_ROOT.rglob("*") if (p / "pred.npy").exists()),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        if any_cands:
            return any_cands[0]
        raise FileNotFoundError(f"No results found for prefix {prefix}")
    return cands[0]

def load_series(per_area_csv: Path):
    df = pd.read_csv(per_area_csv, parse_dates=["date"]).sort_values("date").set_index("date")
    return df["price"].astype(float), df.index

def _shape_pred_array(pred: np.ndarray) -> np.ndarray:
    """
    Ensure pred has shape (W, PRED_LEN).
    Common PatchTST dumps: (W, PRED_LEN, 1) or (W, PRED_LEN).
    """
    arr = np.array(pred)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2 or arr.shape[1] != PRED_LEN:
        raise ValueError(f"Unexpected pred shape {arr.shape}, expect (W,{PRED_LEN}) or (W,{PRED_LEN},1)")
    return arr

def eval_da288_for_area(per_area_csv: Path, area: str) -> dict:
    model_id_prefix = f"PTST_CAN_features_DA288_{area}"
    resdir = find_latest_results_dir(model_id_prefix)
    pred_path = resdir / "pred.npy"
    pred = np.load(pred_path)
    pred = _shape_pred_array(pred)  # (W, 288)
    W = pred.shape[0]

    y, y_index = load_series(per_area_csv)
    N = len(y)

    num_train = int(N * SPLIT[0]); num_val = int(N * SPLIT[1])
    test_start = num_train + num_val

    # Make sure we have enough room for each window of length 288
    max_possible_W = N - test_start - PRED_LEN + 1
    if max_possible_W <= 0:
        raise ValueError(f"{area}: not enough test samples for PRED_LEN={PRED_LEN}")
    if W > max_possible_W:
        # Align to available ground truth windows
        W = max_possible_W
        pred = pred[:W, :]

    # Allocate truth and baselines
    y_true = np.zeros((W, PRED_LEN))
    y_persist = np.zeros((W, PRED_LEN))  # last observed value repeated
    y_daily = np.zeros((W, PRED_LEN))    # value from t-288 for each horizon

    starts = []
    for k in range(W):
        start = test_start + k
        end = start + PRED_LEN
        y_true[k, :] = y.iloc[start:end].to_numpy()

        last_val = float(y.iloc[start-1]) if start-1 >= 0 else float(y.iloc[0])
        y_persist[k, :] = last_val

        # daily seasonal baseline: y[t+h-288]
        daily_start = start - DAILY_LAG
        if daily_start >= 0:
            y_daily[k, :] = y.iloc[daily_start:daily_start+PRED_LEN].to_numpy()
        else:
            # fallback to persistence if no daily history
            y_daily[k, :] = last_val

        starts.append(y_index[start])

    # Flatten for overall metrics
    y_true_f = y_true.flatten()
    y_pred_f = pred.flatten()
    y_pers_f = y_persist.flatten()
    y_daily_f = y_daily.flatten()

    m_model = {"MAE": mae(y_true_f, y_pred_f), "RMSE": rmse(y_true_f, y_pred_f), "MAPE%": mape(y_true_f, y_pred_f, EPS_MAPE)}
    m_pers  = {"MAE": mae(y_true_f, y_pers_f), "RMSE": rmse(y_true_f, y_pers_f), "MAPE%": mape(y_true_f, y_pers_f, EPS_MAPE)}
    m_daily = {"MAE": mae(y_true_f, y_daily_f),"RMSE": rmse(y_true_f, y_daily_f),"MAPE%": mape(y_true_f, y_daily_f, EPS_MAPE)}
    skill_vs_persist = 1.0 - (m_model["MAE"] / max(1e-12, m_pers["MAE"]))

    # Per-horizon error curves (useful to see where model helps most)
    per_h_mae_patch = np.mean(np.abs(y_true - pred), axis=0)
    per_h_mae_pers  = np.mean(np.abs(y_true - y_persist), axis=0)
    per_h_mae_daily = np.mean(np.abs(y_true - y_daily), axis=0)
    horiz_df = pd.DataFrame({
        "horizon": np.arange(1, PRED_LEN+1),
        "mae_patch": per_h_mae_patch,
        "mae_persistence": per_h_mae_pers,
        "mae_daily": per_h_mae_daily
    })
    horiz_df.to_csv(resdir / f"{area}_per_horizon_mae_DA288.csv", index=False)

    # Save first-window stream plot for visual check
    ts0 = pd.date_range(starts[0], periods=PRED_LEN, freq="5min")
    plt.figure(figsize=(12,4))
    plt.plot(ts0, y_true[0], label="Actual")
    plt.plot(ts0, pred[0], label="PatchTST DA288 (5-min)")
    plt.plot(ts0, y_persist[0], label="Persistence (last)")
    plt.plot(ts0, y_daily[0], label="Daily (t-288)")
    plt.title(f"{area} — Next 24h (5-min) forecast — first test window")
    plt.xlabel("Time"); plt.ylabel("Price ($/MWh)")
    plt.legend(); plt.tight_layout()
    plt.savefig(resdir / f"{area}_DA288_first_window.png", dpi=150); plt.close()

    # Export hourly-aggregated day-ahead forecast (from first test window) for planning
    df_hourly = pd.DataFrame({"date": ts0, "forecast_5min": pred[0]})
    hourly = df_hourly.set_index("date").resample("1H").mean().reset_index()
    hourly.to_csv(resdir / f"{area}_DA288_hourly_24.csv", index=False)

    print(f"[EVAL-DA288] {area}: PatchTST MAE={m_model['MAE']:.3f} | "
          f"Persist={m_pers['MAE']:.3f} | Daily={m_daily['MAE']:.3f} | Skill={skill_vs_persist:.3f}")

    return {
        "area": area,
        "results_dir": str(resdir),
        "mae_patch": m_model["MAE"],
        "rmse_patch": m_model["RMSE"],
        "mape_patch_pct": m_model["MAPE%"],
        "mae_persistence": m_pers["MAE"],
        "rmse_persistence": m_pers["RMSE"],
        "mae_daily": m_daily["MAE"],
        "rmse_daily": m_daily["RMSE"],
        "skill_vs_persistence": skill_vs_persist,
    }

def main():
    # 1) Discover ONLY whitelisted files
    area_files = discover_area_files(DATA_DIR)
    if not area_files:
        print(f"No target *_features.csv found in {DATA_DIR}. Nothing to do.", file=sys.stderr)
        return
    print("Processing files:")
    for p in area_files:
        print("  -", p.name)

    # 2) Build per-area cleaned CSV with numeric features
    per_area = {}
    for p in area_files:
        area = area_name_from_path(p)
        try:
            per_area[area] = build_per_area_csv(p)
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")

    if not per_area:
        print("No valid area files to process.", file=sys.stderr)
        return

    # 3) Train
    if RUN_TRAIN:
        for area, csv_path in per_area.items():
            try:
                run_train_for_area(csv_path, area)
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Training failed for {area}: {e}. Skipping evaluation for this area.")
                continue

    # 4) Evaluate (DA288)
    rows = []
    for area, csv_path in per_area.items():
        try:
            rows.append(eval_da288_for_area(csv_path, area))
        except Exception as e:
            print(f"[WARN] Evaluation failed for {area}: {e}")

    if not rows:
        print("No results to summarize."); return

    df = pd.DataFrame(rows)
    df["rank_mae_patch"] = df["mae_patch"].rank(method="min")
    df["rank_skill"]     = (-df["skill_vs_persistence"]).rank(method="min")
    df = df.sort_values(["mae_patch", "skill_vs_persistence"], ascending=[True, False])
    df.to_csv(SUMMARY_CSV, index=False)

    print("\n=== SUMMARY (DA288: next 24h @5-min) across selected areas — features inputs, GPU, AMP off ===")
    print(df[["area","mae_patch","mae_persistence","mae_daily","skill_vs_persistence"]].to_string(index=False))
    print("\nSaved summary CSV:", SUMMARY_CSV)

if __name__ == "__main__":
    main()
