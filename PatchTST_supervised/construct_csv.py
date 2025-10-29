#!/usr/bin/env python3
"""
Consolidate PUB_RealtimeMktPriceYear_*.csv (2010–2025) into one tidy CSV
containing only ENGY columns per area.

Input folder (hard-coded): /home/omaralrefai/dev/PatchTST/.dataset/canada
Output file: canada_realtime_ENGY_2010_2025.csv (saved in the same folder)

Notes
- Skips the first 3 metadata lines in each file.
- Uses row 4 (areas) + row 5 (metrics) to build column names.
- Keeps DELIVERY_DATE, DELIVERY_HOUR, INTERVAL for reference.
- Creates a 'timestamp' = DELIVERY_DATE + (DELIVERY_HOUR-1) hours.
  If INTERVAL>1 exists, spreads minutes evenly within the hour to make unique timestamps.
"""

from pathlib import Path
import math
import pandas as pd

# ----------- CONFIG (hard-coded) -----------
BASE_DIR = Path("/home/omaralrefai/dev/PatchTST/.dataset/canada")
OUT_PATH = BASE_DIR / "canada_realtime_ENGY_2010_2025.csv"
YEAR_START, YEAR_END = 2010, 2025
# ------------------------------------------

def read_year_file(path: Path) -> pd.DataFrame:
    # Read raw file with no header; we'll reconstruct
    raw = pd.read_csv(path, header=None)
    if raw.shape[0] < 5:
        raise ValueError(f"Unexpected format (too few rows) in {path}")

    # Skip metadata (first 3 lines)
    data = raw.iloc[3:].reset_index(drop=True)

    # The next two rows are the header: areas (row 0) and metrics (row 1)
    areas_row = data.iloc[0].tolist()
    metrics_row = data.iloc[1].tolist()

    # First three columns are the keys
    areas_row[:3] = ["DELIVERY_DATE", "DELIVERY_HOUR", "INTERVAL"]
    metrics_row[:3] = ["", "", ""]

    # Build column labels: first three stay as strings, others become (area, metric)
    columns = []
    for ar, mr in zip(areas_row, metrics_row):
        if ar in ["DELIVERY_DATE", "DELIVERY_HOUR", "INTERVAL"]:
            columns.append(ar)
        else:
            columns.append((str(ar).strip(), str(mr).strip()))

    # Actual data starts after the two header rows
    df = data.iloc[2:].reset_index(drop=True)
    df.columns = columns

    # Types
    df["DELIVERY_DATE"] = pd.to_datetime(df["DELIVERY_DATE"], errors="coerce")
    for c in ["DELIVERY_HOUR", "INTERVAL"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only ENGY columns
    engy_cols = [
        col for col in df.columns
        if isinstance(col, tuple) and len(col) == 2 and str(col[1]).upper() == "ENGY"
    ]
    # Rename ENGY columns to just area names
    rename_map = {col: col[0] for col in engy_cols}

    keep = ["DELIVERY_DATE", "DELIVERY_HOUR", "INTERVAL"] + engy_cols
    engy_df = df[keep].copy().rename(columns=rename_map)

    # Compute timestamp:
    # - Hours typically 1..24, convert to 0..23 by subtracting 1
    # - If INTERVAL>1, spread minutes uniformly within the hour
    max_interval = pd.to_numeric(engy_df["INTERVAL"], errors="coerce").max()
    try:
        max_interval = int(max_interval) if not math.isnan(max_interval) else 1
    except Exception:
        max_interval = 1

    def compute_ts(row):
        date = row["DELIVERY_DATE"]
        hour = row["DELIVERY_HOUR"]
        itv = row["INTERVAL"]
        if pd.isna(date) or pd.isna(hour):
            return pd.NaT
        hour_adj = int(hour) - 1
        minute = 0
        if max_interval and max_interval > 1 and not pd.isna(itv):
            minute = int((int(itv) - 1) * (60 / max_interval))
        return pd.to_datetime(date) + pd.to_timedelta(hour_adj, unit="h") + pd.to_timedelta(minute, unit="m")

    engy_df["timestamp"] = engy_df.apply(compute_ts, axis=1)

    # Ensure numeric for ENGY columns
    for c in engy_df.columns:
        if c not in ["timestamp", "DELIVERY_DATE", "DELIVERY_HOUR", "INTERVAL"]:
            engy_df[c] = pd.to_numeric(engy_df[c], errors="coerce")

    # Order columns: timestamp first
    ordered = ["timestamp", "DELIVERY_DATE", "DELIVERY_HOUR", "INTERVAL"] + [
        c for c in engy_df.columns if c not in ["timestamp", "DELIVERY_DATE", "DELIVERY_HOUR", "INTERVAL"]
    ]
    engy_df = engy_df[ordered]

    # Sort for safety
    engy_df = engy_df.sort_values(["timestamp", "DELIVERY_DATE", "DELIVERY_HOUR", "INTERVAL"]).reset_index(drop=True)
    return engy_df

def main():
    # Gather files 2010..2025
    files = [(BASE_DIR / f"PUB_RealtimeMktPriceYear_{y}.csv") for y in range(YEAR_START, YEAR_END + 1)]
    files = [p for p in files if p.exists()]

    if not files:
        raise SystemExit(f"No year files found under {BASE_DIR}")

    parts = []
    for p in sorted(files):
        try:
            df = read_year_file(p)
            print(f"Parsed {p.name:<32} rows={df.shape[0]}")
            parts.append(df)
        except Exception as e:
            print(f"[warn] Skipped {p.name}: {e}")

    if not parts:
        raise SystemExit("No valid files parsed.")

    full = pd.concat(parts, ignore_index=True)

    # Deduplicate in case of overlaps
    full = full.drop_duplicates(subset=["timestamp", "DELIVERY_DATE", "DELIVERY_HOUR", "INTERVAL"])

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    print(full.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
