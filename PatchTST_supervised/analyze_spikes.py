#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def to_nhc(a):
    if a.ndim==3:
        return a if a.shape[1] > a.shape[2] else np.transpose(a,(0,2,1))
    raise ValueError(a.shape)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-dir", required=True, type=Path)
    ap.add_argument("--target-idx", type=int, default=0)
    ap.add_argument("--h", type=int, default=0, help="horizon step index (0=t+1)")
    ap.add_argument("--q", type=float, default=0.99, help="tail quantile threshold")
    args = ap.parse_args()

    d = args.long_dir
    pred = to_nhc(np.load(d/"pred.npy")).astype(float)
    true = to_nhc(np.load(d/"true.npy")).astype(float)

    y = true[:, args.h, args.target_idx]
    p = pred[:, args.h, args.target_idx]

    thr = np.quantile(y, args.q)
    mask = y >= thr

    mae_all = np.mean(np.abs(y - p))
    mae_tail = np.mean(np.abs(y[mask] - p[mask])) if mask.any() else np.nan
    bias_tail = np.mean((p[mask] - y[mask])) if mask.any() else np.nan

    print(f"H={args.h}, channel={args.target_idx}")
    print("Truth quantiles:", {q: float(np.quantile(y,q)) for q in [0.5,0.9,0.95,0.99,0.999]})
    print("Pred  quantiles:", {q: float(np.quantile(p,q)) for q in [0.5,0.9,0.95,0.99,0.999]})
    print(f"Threshold @ q={args.q}: {thr:.3f}  | tail points: {mask.sum()} / {len(y)}")
    print(f"MAE(all)  = {mae_all:.3f}")
    print(f"MAE(tail) = {mae_tail:.3f}")
    print(f"Tail bias (pred-true) = {bias_tail:.3f}  (negative means underpredict)")

if __name__ == "__main__":
    main()