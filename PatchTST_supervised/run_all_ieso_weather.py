#!/usr/bin/env python3
# batch_canada_t+1_timeonly_gpu.py
# Train & evaluate PatchTST per area using engineered time/weather features to predict price.
# Only processes a fixed whitelist of *_features.csv files under DATA_DIR.

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
PRED_LEN   = 1

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

SUMMARY_CSV = RESULTS_ROOT / "CANADA_features_t+1_summary_gpu.csv"
# =================================

def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mape(a, b, eps=1.0): return float(np.mean(np.abs(a - b) / np.maximum(eps, np.abs(a))) * 100.0)

def discover_area_files(root: Path):
    """
    Return list of Paths for the fixed whitelist of *_features.csv.
    Warn about any missing files so you know what's being skipped.
    """
    found = []
    missing = []
    for fname in WHITELIST_FILES:
        p = (root / fname)
        if p.is_file():
            found.append(p)
        else:
            missing.append(fname)
    if missing:
        print("[WARN] The following whitelisted files were not found and will be skipped:")
        for m in missing:
            print("       -", m)
    return sorted(found)

def area_name_from_path(p: Path) -> str:
    """Human-ish area name from filename stem (without _features)."""
    base = p.stem
    if base.endswith("_features"):
        base = base[:-9]
    return base  # keep simple; used only for IDs/labels

def build_per_area_csv(src_csv: Path) -> Path:
    """
    Create per-area CSV with exact numeric columns for PatchTST:
      date, DATE_ORD, hour, day_of_week_idx, interval, month, windspeed_10m, temperature_2m, price
    """
    df = pd.read_csv(src_csv)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    req = {"timestamp","hour","day_of_week","interval","month","windspeed_10m","temperature_2m","price"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"{src_csv.name}: missing required columns: {', '.join(sorted(missing))}")

    # Parse/engineer
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{src_csv.name}: unparsable timestamps found.")

    out = pd.DataFrame({
        "date": ts,                                  # PatchTST expects a date column
        "DATE_ORD": ts.map(lambda x: x.toordinal()), # numeric channel for "timestamp"
        "hour": pd.to_numeric(df["hour"], errors="coerce"),
        "day_of_week_idx": pd.Categorical(df["day_of_week"]).codes,  # 0..6
        "interval": pd.to_numeric(df["interval"], errors="coerce"),
        "month": pd.to_numeric(df["month"], errors="coerce"),
        "windspeed_10m": pd.to_numeric(df["windspeed_10m"], errors="coerce"),
        "temperature_2m": pd.to_numeric(df["temperature_2m"], errors="coerce"),
        "price": pd.to_numeric(df["price"], errors="coerce"),
    })

    # Clean ranges
    out = out[(out["interval"]>=1) & (out["interval"]<=12)]
    out = out[out["hour"].between(0,23)]
    out = out[out["month"].between(1,12)]
    out = out.dropna().sort_values("date").reset_index(drop=True)

    # Save
    area = area_name_from_path(src_csv)
    out_path = DATA_ROOT / f"{area}_clean.csv"
    out.to_csv(out_path, index=False)
    return out_path

def run_train_for_area(per_area_csv: Path, area: str):
    model_id = f"PTST_CAN_features_t+1_{area}"
    cmd = [
        sys.executable, str(RUN_LONGEXP),
        "--is_training", "1",
        "--model_id", model_id,
        "--model", "PatchTST",
        "--data", "custom",
        "--root_path", str(DATA_ROOT),
        "--data_path", per_area_csv.name,
        "--features", "M",                     # X = 7 inputs, target excluded
        "--target", "price",
        "--seq_len", str(SEQ_LEN),
        "--label_len", str(LABEL_LEN),
        "--pred_len", str(PRED_LEN),
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
        # last resort: any pred.npy
        any_cands = sorted((p for p in RESULTS_ROOT.rglob("*") if (p / "pred.npy").exists()),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        if any_cands:
            return any_cands[0]
        raise FileNotFoundError(f"No results found for prefix {prefix}")
    return cands[0]

def load_series(per_area_csv: Path):
    df = pd.read_csv(per_area_csv, parse_dates=["date"]).sort_values("date").set_index("date")
    return df["price"].astype(float)

def eval_t1_for_area(per_area_csv: Path, area: str) -> dict:
    model_id_prefix = f"PTST_CAN_features_t+1_{area}"
    resdir = find_latest_results_dir(model_id_prefix)
    pred_path = resdir / "pred.npy"
    pred = np.load(pred_path).squeeze()
    if pred.ndim != 1:
        raise ValueError(f"{area}: unexpected pred.npy shape {pred.shape} (expect (W,) for pred_len=1)")
    W = len(pred)

    y = load_series(per_area_csv)
    N = len(y)

    num_train = int(N * SPLIT[0]); num_val = int(N * SPLIT[1])
    test_start = num_train + num_val
    if test_start + W + PRED_LEN > N:
        test_start = max(SEQ_LEN + LABEL_LEN, N - (W + PRED_LEN))

    ts = []
    y_true = np.zeros(W); y_pred = np.zeros(W)
    persist = np.zeros(W); daily = np.zeros(W)

    for k in range(W):
        start = test_start + k
        y_true[k] = float(y.iloc[start])
        y_pred[k] = float(pred[k])
        last     = float(y.iloc[start-1]) if start-1 >= 0 else float(y.iloc[0])             # t-1
        daily_k  = float(y.iloc[start-DAILY_LAG]) if start-DAILY_LAG >= 0 else last         # t-288
        persist[k] = last
        daily[k]   = daily_k
        ts.append(y.index[start])

    ts = pd.to_datetime(ts)

    m_model = {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE%": mape(y_true, y_pred, EPS_MAPE)}
    m_pers  = {"MAE": mae(y_true, persist), "RMSE": rmse(y_true, persist), "MAPE%": mape(y_true, persist, EPS_MAPE)}
    m_daily = {"MAE": mae(y_true, daily),   "RMSE": rmse(y_true, daily),   "MAPE%": mape(y_true, daily, EPS_MAPE)}
    skill_vs_persist = 1.0 - (m_model["MAE"] / max(1e-12, m_pers["MAE"]))

    stream = pd.DataFrame({
        "date": ts,
        "y_true": y_true,
        "PatchTST_t+1": y_pred,
        "Persistence_t+1": persist,
        "Daily_t+1(t-288)": daily
    })
    stream_csv = resdir / f"{area}_tplus1_stream.csv"
    stream.to_csv(stream_csv, index=False)

    plt.figure(figsize=(12,4))
    plt.plot(ts, y_true, label="Actual")
    plt.plot(ts, y_pred, label="PatchTST t+1 (5 min)")
    plt.plot(ts, persist, label="Persistence (t-1)")
    plt.plot(ts, daily, label="Daily (t-288)")
    plt.title(f"{area} — t+1 (5-min ahead) stream")
    plt.xlabel("Time"); plt.ylabel("Price ($/MWh)")
    plt.legend(); plt.tight_layout()
    plt.savefig(resdir / f"{area}_tplus1_stream.png", dpi=150); plt.close()

    print(f"[EVAL] {area}: PatchTST MAE={m_model['MAE']:.3f} | Persistence={m_pers['MAE']:.3f} | Daily={m_daily['MAE']:.3f} | Skill={skill_vs_persist:.3f}")

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
        "better_than_persistence": int(m_model["MAE"] < m_pers["MAE"])
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
                print(f"[WARN] Training failed for {area}: {e}. Skipping evaluation.")
                continue

    # 4) Evaluate
    rows = []
    for area, csv_path in per_area.items():
        try:
            rows.append(eval_t1_for_area(csv_path, area))
        except Exception as e:
            print(f"[WARN] Evaluation failed for {area}: {e}")

    if not rows:
        print("No results to summarize."); return

    df = pd.DataFrame(rows)
    df["rank_mae_patch"] = df["mae_patch"].rank(method="min")
    df["rank_skill"]     = (-df["skill_vs_persistence"]).rank(method="min")
    df = df.sort_values(["mae_patch", "skill_vs_persistence"], ascending=[True, False])
    df.to_csv(SUMMARY_CSV, index=False)

    print("\n=== SUMMARY (t+1, 5-min) across selected areas — features inputs, GPU, AMP off ===")
    print(df[["area","mae_patch","mae_persistence","mae_daily","skill_vs_persistence","better_than_persistence"]].to_string(index=False))
    print("\nSaved summary CSV:", SUMMARY_CSV)

if __name__ == "__main__":
    main()
