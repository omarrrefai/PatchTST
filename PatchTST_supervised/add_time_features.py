#!/usr/bin/env python3
"""
Add time-derived columns to Canada area CSVs.

Will ONLY process CSVs that contain all required columns:
  - timestamp
  - price
  - windspeed_10m
  - temperature_2m

New columns added:
  - hour           : 0–23
  - day_of_week    : Monday..Sunday
  - interval       : 1–12 (5-min buckets within each hour)
  - month          : 1–12

Usage:
  python add_time_features.py \
      --data-dir /home/omaralrefai/dev/PatchTST/.dataset/canada \
      [--overwrite]
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

REQ_COLS = {"timestamp", "price", "windspeed_10m", "temperature_2m"}

def _read_header(csv_path: Path) -> set:
    """Read only the header row and return a lowercased set of column names."""
    try:
        hdr = pd.read_csv(csv_path, nrows=0)
        return {c.strip().lower() for c in hdr.columns}
    except Exception:
        return set()

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names to lower-case for robustness
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    missing = REQ_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    bad = int(ts.isna().sum())
    if bad:
        raise ValueError(f"{bad} unparsable timestamps.")

    out = df.copy()
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.day_name()
    out["interval"] = (ts.dt.minute // 5) + 1
    out["month"] = ts.dt.month

    # Order: timestamp, derived time columns, then the rest (keep original order)
    cols = list(out.columns)
    for c in ["month", "interval", "day_of_week", "hour"]:
        if c in cols:
            cols.remove(c)
    insert_at = cols.index("timestamp") + 1
    for c in ["hour", "day_of_week", "interval", "month"]:
        cols.insert(insert_at, c)
        insert_at += 1
    return out[cols]

def process_one(csv_path: Path, overwrite: bool) -> Path:
    df = pd.read_csv(csv_path)
    df2 = add_time_features(df)
    if overwrite:
        out_path = csv_path
    else:
        out_path = csv_path.with_name(csv_path.stem + "_features.csv")
    df2.to_csv(out_path, index=False)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing area CSV files.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite original files instead of writing *_features.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        print(f"ERROR: {data_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    csvs = sorted(p for p in data_dir.glob("*.csv") if p.is_file())
    if not csvs:
        print(f"No CSV files found in {data_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(csvs)} CSV file(s) in {data_dir} ...")
    candidates = []
    skipped_schema = []
    for p in csvs:
        header = _read_header(p)
        if REQ_COLS.issubset(header):
            candidates.append(p)
        else:
            skipped_schema.append(p.name)

    if not candidates:
        print("No CSVs with required columns found. Exiting.", file=sys.stderr)
        if skipped_schema:
            print("Skipped (missing columns):")
            for name in skipped_schema:
                print(f"  - {name}", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(candidates)} matching CSV(s). "
          f"Writing {'in-place' if args.overwrite else 'copies with _features suffix'}...")
    successes, failures = 0, 0
    for p in candidates:
        try:
            out = process_one(p, args.overwrite)
            print(f"✓ {p.name} -> {out.name}")
            successes += 1
        except Exception as e:
            print(f"✗ {p.name}: {e}", file=sys.stderr)
            failures += 1

    if skipped_schema:
        print("\nSkipped (non-matching schema):")
        for name in skipped_schema:
            print(f"  - {name}")

    print(f"\nDone. {successes} succeeded, {failures} failed.")
    if failures:
        sys.exit(3)

if __name__ == "__main__":
    main()
