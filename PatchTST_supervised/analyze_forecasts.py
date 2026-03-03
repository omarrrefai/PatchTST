#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def to_nhc(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3:
        if a.shape[1] > a.shape[2]:
            return a
        return np.transpose(a, (0, 2, 1))
    if a.ndim == 2:
        return a[:, :, None]
    if a.ndim == 1:
        return a[:, None, None]
    raise ValueError(a.shape)


def load_long(long_dir: Path):
    pred = to_nhc(np.load(long_dir / "pred.npy")).astype(float)
    true = to_nhc(np.load(long_dir / "true.npy")).astype(float)
    if pred.shape != true.shape:
        raise ValueError(f"LONG pred/true mismatch: {pred.shape} vs {true.shape}")
    return pred, true


def load_short_pred(short_dir: Path) -> np.ndarray:
    p = np.load(short_dir / "pred.npy")
    if p.ndim == 3:
        if p.shape[2] == 1:
            return p[:, :, 0].astype(float)  # (N,C)
        if p.shape[1] == 1:
            return p[:, 0, :].astype(float)  # (N,C)
        raise ValueError(p.shape)
    if p.ndim == 2:
        return p.astype(float)
    if p.ndim == 1:
        return p[:, None].astype(float)
    raise ValueError(p.shape)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short-dir", required=True, type=Path)
    ap.add_argument("--long-dir", required=True, type=Path)
    ap.add_argument("--target-idx", type=int, default=0)
    ap.add_argument("--max-h", type=int, default=288)
    ap.add_argument("--window-start", type=int, default=0)
    ap.add_argument("--window-len", type=int, default=2000)
    args = ap.parse_args()

    long_pred, long_true = load_long(args.long_dir)   # (N,H,C)
    short_pred = load_short_pred(args.short_dir)      # (N,C)

    C = min(long_true.shape[2], short_pred.shape[1])
    idx = args.target_idx
    if idx >= C:
        raise ValueError(f"target-idx {idx} out of range (C={C})")

    N = min(long_true.shape[0], short_pred.shape[0])
    H = min(long_true.shape[1], args.max_h)

    y1 = long_true[:N, 0, idx]
    p_short = short_pred[:N, idx]
    p_long_t1 = long_pred[:N, 0, idx]

    print("t+1 MAE:")
    print("  SHORT:", mae(y1, p_short))
    print("  LONG(h=1):", mae(y1, p_long_t1))

    # horizon MAE curve
    maes = []
    for h in range(H):
        maes.append(mae(long_true[:N, h, idx], long_pred[:N, h, idx]))

    plt.figure(figsize=(10, 3.2))
    plt.plot(range(1, H+1), maes)
    plt.title("Day-ahead Forecast MAE vs Horizon Step")
    plt.xlabel("Horizon step (5-min ahead)")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(args.long_dir / "horizon_mae_curve.png", dpi=250)
    plt.close()
    print("Saved:", args.long_dir / "horizon_mae_curve.png")

    # window plot: show truth vs short vs long(h=1)
    s = args.window_start
    e = s + args.window_len
    plt.figure(figsize=(14, 3.2))
    plt.plot(y1[s:e], label="truth(t+1)")
    plt.plot(p_short[s:e], label="short(t+1)")
    plt.plot(p_long_t1[s:e], label="long(h=1)")
    plt.title(f"t+1 Forecast vs Truth (window {s}..{e})")
    plt.xlabel("Step")
    plt.ylabel("$/MWh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.long_dir / "tplus1_window_compare.png", dpi=250)
    plt.close()
    print("Saved:", args.long_dir / "tplus1_window_compare.png")


if __name__ == "__main__":
    main()