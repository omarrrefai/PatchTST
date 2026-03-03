#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd


AREAS = ["manitoba", "new-york", "ontario", "quebec_p33c", "manitoba_sk"]

AREA_TO_CSV_COL = {
    "manitoba": "Manitoba",
    "new-york": "New-York",
    "ontario": "Ontario",
    "quebec_p33c": "Quebec P33C",
    "manitoba_sk": "Manitoba SK",
}

PRED_NAME = "pred.npy"
TRUE_NAME = "true.npy"
TRUE_TS_NAME = "true_timestamps.npy"  # timestamps aligned to true windows

def parse_pred_len_from_folder(folder: Path) -> int | None:
    """
    Extract pred_len from folder name like ..._pl288_...
    Returns None if not found.
    """
    m = re.search(r"_pl(\d+)_", folder.name)
    if not m:
        return None
    return int(m.group(1))

def find_pred_paths(base_results_dir: Path, area: str) -> list[Path]:
    candidates = list(base_results_dir.rglob(PRED_NAME))
    area_lower = area.lower()
    matches = [p for p in candidates if area_lower in str(p.parent).lower()]
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches

def load_truth_series(csv_path: Path, csv_col: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Prefer timestamp column
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    else:
        required = {"DELIVERY_DATE", "DELIVERY_HOUR", "INTERVAL"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"No 'timestamp' and missing fields to construct it: {sorted(missing)}"
            )

        df["DELIVERY_DATE"] = pd.to_datetime(df["DELIVERY_DATE"], errors="coerce")
        df = df.dropna(subset=["DELIVERY_DATE"])

        # Common energy convention: DELIVERY_HOUR is 1..24; map 1->0
        hour0 = pd.to_numeric(df["DELIVERY_HOUR"], errors="coerce") - 1
        interval0 = pd.to_numeric(df["INTERVAL"], errors="coerce") - 1

        hour0 = hour0.fillna(0).astype(int)
        interval0 = interval0.fillna(0).astype(int)

        df["timestamp"] = (
            df["DELIVERY_DATE"]
            + pd.to_timedelta(hour0, unit="h")
            + pd.to_timedelta(interval0 * 5, unit="m")
        )
        df = df.sort_values("timestamp").set_index("timestamp")

    if csv_col not in df.columns:
        raise ValueError(f"CSV column '{csv_col}' not found. Columns: {df.columns.tolist()}")

    s = pd.to_numeric(df[csv_col], errors="coerce").dropna()
    return s

def sliding_windows_1d(x: np.ndarray, window: int) -> np.ndarray:
    """
    Return shape (N, window) sliding windows with stride 1.
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if len(x) < window:
        raise ValueError(f"len(x)={len(x)} < window={window}")
    return np.lib.stride_tricks.sliding_window_view(x, window_shape=window)

def align_truth_to_pred_shape(
    truth_series: pd.Series,
    pred_shape: tuple,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build true array matching pred.npy shape using tail alignment.

    Handles pred shapes such as:
      (N, pred_len)
      (N, pred_len, 1)
      (N, 1, pred_len)

    Returns:
      true_arr: np.ndarray matching pred_shape (float32)
      true_ts: timestamps for each window position, aligned to pred windows (int64 ns)
    """
    # Determine which axis equals pred_len
    axes = [i for i, d in enumerate(pred_shape) if d == pred_len]
    if not axes:
        raise ValueError(f"Could not find pred_len={pred_len} in pred shape {pred_shape}")

    # Assume first axis is number of windows
    N = pred_shape[0]

    # To build N windows of length pred_len with stride 1, we need N + pred_len - 1 points
    needed = N + pred_len - 1
    if len(truth_series) < needed:
        raise ValueError(
            f"Truth length={len(truth_series)} < needed={needed} for N={N}, pred_len={pred_len}. "
            "Your CSV may not cover the same test window or pred is not windowed stride=1."
        )

    # Tail-align to match most common PatchTST test split behavior
    s_tail = truth_series.tail(needed)
    x = s_tail.to_numpy(dtype=np.float32)

    # Windows: shape (N, pred_len)
    win = sliding_windows_1d(x, pred_len)
    if win.shape[0] != N:
        # This should be exact if stride=1 and needed was correct
        raise ValueError(f"Window count mismatch: win.shape[0]={win.shape[0]} vs N={N}")

    # Build timestamps aligned to each window start (same N as windows)
    # We store window start timestamps (not each point inside the window) for reproducibility.
    ts_index = s_tail.index
    # For each window, start timestamp is ts_index[i]
    true_ts = ts_index[:N].view("int64")

    # Now reshape to match pred_shape
    if pred_shape == (N, pred_len):
        true_arr = win
    elif len(pred_shape) == 3:
        # Could be (N, pred_len, C) or (N, C, pred_len)
        a = axes[0]
        if a == 1:
            # (N, pred_len, C)
            C = pred_shape[2]
            true_arr = np.repeat(win[:, :, None], C, axis=2) if C > 1 else win[:, :, None]
        elif a == 2:
            # (N, C, pred_len)
            C = pred_shape[1]
            base = win[:, None, :]
            true_arr = np.repeat(base, C, axis=1) if C > 1 else base
        else:
            raise ValueError(f"Unexpected pred_len axis {a} for pred_shape {pred_shape}")
    else:
        raise ValueError(
            f"Unsupported pred shape {pred_shape}. "
            "If your pred has 4D shape, tell me the shape and I’ll adapt it."
        )

    return true_arr.astype(np.float32), true_ts

def main():
    ap = argparse.ArgumentParser(
        description="Construct true.npy from CSV for multiple areas and save next to pred.npy."
    )
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--pick", choices=["newest", "all"], default="newest")
    ap.add_argument("--save-timestamps", action="store_true")
    args = ap.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {args.results_dir}")

    for area in AREAS:
        csv_col = AREA_TO_CSV_COL[area]
        print(f"\n=== Area: {area}  (CSV column: '{csv_col}') ===")

        s_truth = load_truth_series(args.csv, csv_col)

        pred_paths = find_pred_paths(args.results_dir, area)
        if not pred_paths:
            print(f"  [WARN] No pred.npy found for area '{area}'")
            continue

        target_pred_paths = pred_paths[:1] if args.pick == "newest" else pred_paths

        for pred_path in target_pred_paths:
            y_pred = np.load(pred_path)
            pred_shape = y_pred.shape

            pred_len = parse_pred_len_from_folder(pred_path.parent)
            if pred_len is None:
                # fallback: try to infer from shape by picking a "typical" horizon length
                # but safest is to have _pl###_ in folder name
                raise ValueError(
                    f"Could not parse pred_len from folder name: {pred_path.parent.name}. "
                    "Expected pattern like _pl288_."
                )

            true_arr, true_ts = align_truth_to_pred_shape(s_truth, pred_shape, pred_len)

            out_dir = pred_path.parent
            true_path = out_dir / TRUE_NAME
            np.save(true_path, true_arr)
            print(f"  pred shape: {pred_shape}  | saved true shape: {true_arr.shape}")
            print(f"  Saved: {true_path}")

            if args.save_timestamps:
                ts_path = out_dir / TRUE_TS_NAME
                np.save(ts_path, true_ts)
                print(f"  Saved: {ts_path} (window start timestamps, int64 ns)")

if __name__ == "__main__":
    main()