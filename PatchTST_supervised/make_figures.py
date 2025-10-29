# t15_eval_specific.py
# Purpose: Evaluate a PatchTST model trained with pred_len=1 (t+15 only)
# using the specific pred.npy you provided. No retraining required.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========= CONFIG (specific to your run) =========
PRED_PATH = Path("/home/omaralrefai/dev/PatchTST/PatchTST_supervised/results/PTST_t15_PatchTST_custom_ftMS_sl192_ll48_pl1_dm512_nh8_el3_dl2_df2048_fc1_ebfixed_dtTrue_test_0/pred.npy")
CSV_PATH  = Path("/home/omaralrefai/dev/PatchTST/.dataset/ercot/ercot_15min_with_time.csv")
TARGET    = "HB_BUSAVG"

# Training/dataloader settings used in your run:
SEQ_LEN   = 192
LABEL_LEN = 48
PRED_LEN  = 1     # t+15 only
SPLIT     = (0.7, 0.1, 0.2)  # 70/10/20

# Metrics/plotting
EPS_MAPE  = 1.0   # avoid tiny denominators for MAPE
DOWNSAMPLE_PLOT = 1  # set >1 to thin the plot if too dense
# ================================================

def mae(a, b):   return float(np.mean(np.abs(a - b)))
def rmse(a, b):  return float(np.sqrt(np.mean((a - b) ** 2)))
def mape(a, b, eps=1.0): return float(np.mean(np.abs(a - b) / np.maximum(eps, np.abs(a))) * 100.0)

def load_pred(pred_path: Path) -> np.ndarray:
    pred = np.load(pred_path)
    # Expected shapes: (W, 1) or (W, 1, 1); squeeze to (W,)
    pred = np.squeeze(pred)
    if pred.ndim != 1:
        raise ValueError(f"Unexpected pred.npy shape after squeeze: {pred.shape}")
    return pred  # normalized (RevIN space), horizon 1 only

def load_series(csv_path: Path, target: str) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in CSV columns.")
    return df[target].astype(float)

def undo_revin_and_rebuild(y: pd.Series, pred_norm_1: np.ndarray,
                           seq_len: int, label_len: int, pred_len: int, split) -> dict:
    """
    Reconstruct true t+15 and undo RevIN for the one-step prediction stream.
    Returns dict with timestamps, y_true, y_pred, persistence, daily, and indices.
    """
    N = len(y)
    W = len(pred_norm_1)
    num_train = int(N * split[0])
    num_val   = int(N * split[1])
    test_start = num_train + num_val

    # Safety: make sure we can index y[start] for all k
    # If bounds are tight, shift test_start so that start+pred_len <= N
    if test_start + W + pred_len > N:
        test_start = max(seq_len + label_len, N - (W + pred_len))

    ts = []
    y_true = np.zeros(W)
    y_pred = np.zeros(W)
    persist = np.zeros(W)
    daily = np.zeros(W)

    for k in range(W):
        start = test_start + k  # forecast origin
        # encoder window statistics for RevIN inverse
        enc_start = max(0, start - (seq_len + label_len))
        enc_win = y.iloc[enc_start:start].to_numpy()
        if enc_win.size == 0:
            # fallback to earlier history
            enc_win = y.iloc[:start].to_numpy()
        mu = float(enc_win.mean()) if enc_win.size else float(y.iloc[:start+1].mean())
        sigma = float(enc_win.std(ddof=0)) if enc_win.size else float(y.iloc[:start+1].std(ddof=0))
        if sigma < 1e-8:
            sigma = 1.0  # avoid degenerate scaling

        # t+15 truth and unnormalized prediction
        y_true[k] = float(y.iloc[start])             # horizon=1 target
        y_pred[k] = float(pred_norm_1[k] * sigma + mu)

        # baselines
        last = float(y.iloc[start - 1]) if start - 1 >= 0 else float(y.iloc[0])
        persist[k] = last
        daily[k] = float(y.iloc[start - 96]) if start - 96 >= 0 else last

        ts.append(y.index[start])

    return {
        "timestamps": np.array(ts),
        "y_true": y_true,
        "y_pred": y_pred,
        "persistence": persist,
        "daily": daily,
        "test_start": test_start
    }

def main():
    # Paths & loading
    res_dir = PRED_PATH.parent
    pred_norm_1 = load_pred(PRED_PATH)
    y = load_series(CSV_PATH, TARGET)

    # Rebuild & un-RevIN
    out = undo_revin_and_rebuild(y, pred_norm_1, SEQ_LEN, LABEL_LEN, PRED_LEN, SPLIT)
    ts = pd.to_datetime(out["timestamps"])
    y_true = out["y_true"]; y_pred = out["y_pred"]
    persist = out["persistence"]; daily = out["daily"]

    # Metrics on full t+15 stream
    m_model = {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE%": mape(y_true, y_pred, EPS_MAPE)}
    m_pers  = {"MAE": mae(y_true, persist), "RMSE": rmse(y_true, persist), "MAPE%": mape(y_true, persist, EPS_MAPE)}
    m_daily = {"MAE": mae(y_true, daily),   "RMSE": rmse(y_true, daily),   "MAPE%": mape(y_true, daily, EPS_MAPE)}

    skill_vs_persist = 1.0 - (m_model["MAE"] / max(1e-12, m_pers["MAE"]))

    # Print summary
    print("\n=== t+15 SLIDING STREAM (entire test span) ===")
    print(f"PatchTST_t15   MAE={m_model['MAE']:.3f}  RMSE={m_model['RMSE']:.3f}  MAPE%={m_model['MAPE%']:.2f}")
    print(f"Persistence    MAE={m_pers['MAE']:.3f}  RMSE={m_pers['RMSE']:.3f}  MAPE%={m_pers['MAPE%']:.2f}")
    print(f"Daily (t-96)   MAE={m_daily['MAE']:.3f}  RMSE={m_daily['RMSE']:.3f}  MAPE%={m_daily['MAPE%']:.2f}")
    print(f"Skill vs Persistence = {skill_vs_persist:.3f}")

    # Save CSV of the stream
    stream = pd.DataFrame({
        "timestamp": ts,
        "y_true": y_true,
        "PatchTST_t15": y_pred,
        "Persistence_t15": persist,
        "Daily_t15": daily
    })
    csv_out = res_dir / f"{TARGET}_tplus15_stream_specific.csv"
    stream.to_csv(csv_out, index=False)
    print("Saved stream CSV:", csv_out)

    # Plot stitched stream (downsample for legibility if desired)
    ds = DOWNSAMPLE_PLOT if DOWNSAMPLE_PLOT >= 1 else 1
    plt.figure(figsize=(12, 4))
    plt.plot(ts[::ds], y_true[::ds], label="Actual")
    plt.plot(ts[::ds], y_pred[::ds], label="PatchTST t+15")
    plt.plot(ts[::ds], persist[::ds], label="Persistence")
    plt.plot(ts[::ds], daily[::ds], label="Daily (t-96)")
    plt.title(f"{TARGET} — Sliding t+15 Stream")
    plt.xlabel("Time"); plt.ylabel("Price ($/MWh)")
    plt.legend(); plt.tight_layout()
    png_out = res_dir / f"{TARGET}_tplus15_stream_specific.png"
    plt.savefig(png_out, dpi=150); plt.close()
    print("Saved plot:", png_out)

    # Also save arrays for future analysis
    np.save(res_dir / f"{TARGET}_tplus15_true.npy", y_true)
    np.save(res_dir / f"{TARGET}_tplus15_pred_raw.npy", y_pred)
    np.save(res_dir / f"{TARGET}_tplus15_persistence.npy", persist)
    np.save(res_dir / f"{TARGET}_tplus15_daily.npy", daily)
    print("Saved numpy arrays to:", res_dir)

if __name__ == "__main__":
    main()
