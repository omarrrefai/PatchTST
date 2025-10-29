#!/usr/bin/env python3
# batch_canada_t+1_timeonly_gpu.py
# Train & evaluate PatchTST for each Canada ENGY target using ONLY time inputs:
#   date (index), DELIVERY_DATE_ORD, DELIVERY_HOUR, INTERVAL
# Compare vs naïve: Persistence (t-1) and Daily (t-288).
# Runs on GPU (AMP disabled for stability).

import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========== USER CONFIG ==========
MASTER_CSV = Path("/home/omaralrefai/dev/PatchTST/.dataset/canada/canada_realtime_ENGY_2010_2025.csv")

DATA_ROOT  = MASTER_CSV.parent / "per_target_timeonly_clean"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

RUN_LONGEXP  = Path("run_longExp.py")
RESULTS_ROOT = Path("results"); RESULTS_ROOT.mkdir(exist_ok=True)

TARGETS = [
    "Manitoba", "Manitoba SK", "Michigan", "Minnesota", "New-York", "Ontario",
    "Quebec AT", "Quebec B5D.B31L", "Quebec D4Z", "Quebec D5A", "Quebec H4Z",
    "Quebec H9A", "Quebec P33C", "Quebec Q4C", "Quebec X2Y",
]

RUN_TRAIN = True

FREQ       = "5min"
PER_HOUR   = 12
DAILY_LAG  = 24 * PER_HOUR  # 288

# features='M' => X excludes target; we pass 3 time inputs as channels
ENC_IN     = 3
DEC_IN     = 3
C_OUT      = 1

SEQ_LEN    = 576    # two days of 5-min history
LABEL_LEN  = 72
PRED_LEN   = 1

D_MODEL    = 512
N_HEADS    = 8
E_LAYERS   = 3
D_LAYERS   = 2
D_FF       = 2048
DROPOUT    = 0.05
BATCH      = 32           # GPU-friendly but safer numerics
LR         = 1e-4
EPOCHS     = 30
PATIENCE   = 8
PATCH_LEN  = 48
STRIDE     = 24
USE_GPU    = 1            # <— GPU ON
NUM_WORKERS= 0
EMBED      = "fixed"

# Keep these off for stability with small-scale inputs
REVIN      = 0
AFFINE     = 0
SUBTRACT_LAST = 0
USE_AMP    = False        # <— explicitly disable AMP on GPU

SPLIT      = (0.7, 0.1, 0.2)
EPS_MAPE   = 1.0

SUMMARY_CSV = RESULTS_ROOT / "CANADA_timeonly_t+1_summary_gpu.csv"
# =================================

def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mape(a, b, eps=1.0): return float(np.mean(np.abs(a - b) / np.maximum(eps, np.abs(a))) * 100.0)

def build_per_target_csv(master_csv: Path, target: str) -> Path:
    """
    Create per-target CSV with strict cleaning and exact columns:
      date, DELIVERY_DATE_ORD, DELIVERY_HOUR, INTERVAL, <target>
    """
    df = pd.read_csv(master_csv, parse_dates=["timestamp"])
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in master CSV.")

    # rename to 'date' for PatchTST
    df = df.rename(columns={"timestamp": "date"})
    df = df.sort_values("date").reset_index(drop=True)

    # Coerce & build inputs
    df["DELIVERY_DATE"] = pd.to_datetime(df["DELIVERY_DATE"], errors="coerce")
    df["DELIVERY_DATE_ORD"] = df["DELIVERY_DATE"].map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    df["DELIVERY_HOUR"] = pd.to_numeric(df["DELIVERY_HOUR"], errors="coerce")
    df["INTERVAL"] = pd.to_numeric(df["INTERVAL"], errors="coerce")
    df[target] = pd.to_numeric(df[target], errors="coerce")

    # Valid ranges
    df = df[(df["INTERVAL"]>=1) & (df["INTERVAL"]<=12)]
    df = df[df["DELIVERY_HOUR"].between(-1, 25)]

    # Keep only required columns and drop any NaNs
    keep = ["date", "DELIVERY_DATE_ORD", "DELIVERY_HOUR", "INTERVAL", target]
    df = df[keep].dropna().reset_index(drop=True)

    out = DATA_ROOT / f"{target.replace(' ', '_')}_timeonly_clean.csv"
    df.to_csv(out, index=False)
    return out

def run_train_for_target(per_target_csv: Path, target: str):
    model_id = f"PTST_CAN_timeonly_t+1_{target.replace(' ', '_')}"
    cmd = [
        sys.executable, str(RUN_LONGEXP),
        "--is_training", "1",
        "--model_id", model_id,
        "--model", "PatchTST",
        "--data", "custom",
        "--root_path", str(DATA_ROOT),
        "--data_path", per_target_csv.name,
        "--features", "M",                     # X = 3 time inputs only
        "--target", target,
        "--seq_len", str(SEQ_LEN),
        "--label_len", str(LABEL_LEN),
        "--pred_len", str(PRED_LEN),
        "--enc_in", str(ENC_IN),               # 3 inputs: DATE_ORD, HOUR, INTERVAL
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
        cmd.append("--use_amp")  # flag, no argument

    print(f"\n[TRAIN] {target} -> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def find_latest_results_dir_for_target(target: str) -> Path:
    prefix = f"PTST_CAN_timeonly_t+1_{target.replace(' ', '_')}_PatchTST_custom_ftM_sl{SEQ_LEN}_ll{LABEL_LEN}_pl{PRED_LEN}_"
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
        raise FileNotFoundError(f"No results found for target {target} (prefix {prefix})")
    return cands[0]

def load_series(per_target_csv: Path, target: str):
    df = pd.read_csv(per_target_csv, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True).set_index("date")
    return df[target].astype(float)

def eval_t1_for_target(per_target_csv: Path, target: str) -> dict:
    resdir = find_latest_results_dir_for_target(target)
    pred_path = resdir / "pred.npy"
    pred = np.load(pred_path).squeeze()
    if pred.ndim != 1:
        raise ValueError(f"{target}: unexpected pred.npy shape {pred.shape} (expect (W,) for pred_len=1)")
    W = len(pred)

    y = load_series(per_target_csv, target)
    N = len(y)

    num_train = int(N * SPLIT[0]); num_val = int(N * SPLIT[1])
    test_start = num_train + num_val
    if test_start + W + PRED_LEN > N:
        test_start = max(SEQ_LEN + LABEL_LEN, N - (W + PRED_LEN))

    # With RevIN off, predictions should already be on data scale
    ts = []
    y_true = np.zeros(W); y_pred = np.zeros(W)
    persist = np.zeros(W); daily = np.zeros(W)

    for k in range(W):
        start = test_start + k

        y_true[k] = float(y.iloc[start])
        y_pred[k] = float(pred[k])

        last = float(y.iloc[start-1]) if start-1 >= 0 else float(y.iloc[0])             # t-1 (5 min)
        daily_k = float(y.iloc[start-DAILY_LAG]) if start-DAILY_LAG >= 0 else last       # t-288 (prev day)
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
    stream_csv = resdir / f"{target.replace(' ', '_')}_tplus1_stream.csv"
    stream.to_csv(stream_csv, index=False)

    plt.figure(figsize=(12,4))
    plt.plot(ts, y_true, label="Actual")
    plt.plot(ts, y_pred, label="PatchTST t+1 (5 min)")
    plt.plot(ts, persist, label="Persistence (t-1)")
    plt.plot(ts, daily, label="Daily (t-288)")
    plt.title(f"{target} — t+1 (5-min ahead) stream")
    plt.xlabel("Time"); plt.ylabel("ENGY ($/MWh)")
    plt.legend(); plt.tight_layout()
    plt.savefig(resdir / f"{target.replace(' ', '_')}_tplus1_stream.png", dpi=150); plt.close()

    print(f"[EVAL] {target}: PatchTST MAE={m_model['MAE']:.3f} | Persistence={m_pers['MAE']:.3f} | Daily={m_daily['MAE']:.3f} | Skill={skill_vs_persist:.3f}")

    return {
        "target": target,
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
    per_target = {t: build_per_target_csv(MASTER_CSV, t) for t in TARGETS}

    if RUN_TRAIN:
        for tgt in TARGETS:
            try:
                run_train_for_target(per_target[tgt], tgt)
            except subprocess.CalledProcessError as e:
                print(f"[WARN] Training failed for {tgt}: {e}. Skipping evaluation for this target.")
                continue

    rows = []
    for tgt in TARGETS:
        try:
            rows.append(eval_t1_for_target(per_target[tgt], tgt))
        except Exception as e:
            print(f"[WARN] Evaluation failed for {tgt}: {e}")

    if not rows:
        print("No results to summarize."); return

    df = pd.DataFrame(rows)
    df["rank_mae_patch"] = df["mae_patch"].rank(method="min")
    df["rank_skill"]     = (-df["skill_vs_persistence"]).rank(method="min")
    df = df.sort_values(["mae_patch", "skill_vs_persistence"], ascending=[True, False])
    df.to_csv(SUMMARY_CSV, index=False)

    print("\n=== SUMMARY (t+1, 5-min) across all targets — time-only inputs, GPU, AMP off ===")
    print(df[["target","mae_patch","mae_persistence","mae_daily","skill_vs_persistence","better_than_persistence"]].to_string(index=False))
    print("\nSaved summary CSV:", SUMMARY_CSV)

if __name__ == "__main__":
    main()
