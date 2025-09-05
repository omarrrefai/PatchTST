import argparse
from pathlib import Path
import pandas as pd

REQ_COLS = {
    "Delivery Date",
    "Delivery Hour",
    "Delivery Interval",
    "Repeated Hour Flag",
    "Settlement Point Name",
    "Settlement Point Type",
    "Settlement Point Price",
}

def load_one(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p, engine="openpyxl")
    else:
        df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    missing = REQ_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{p.name}: missing columns: {sorted(missing)}")

    # Parse + coerce
    date = pd.to_datetime(df["Delivery Date"], format="%m/%d/%Y", errors="coerce")
    hour = pd.to_numeric(df["Delivery Hour"], errors="coerce").astype("Int64")
    interval = pd.to_numeric(df["Delivery Interval"], errors="coerce").astype("Int64")

    bad = date.isna() | hour.isna() | interval.isna()
    if bad.any():
        # Drop obviously bad rows rather than failing the whole file
        df = df.loc[~bad].copy()
        date = date.loc[~bad]
        hour = hour.loc[~bad].astype(int)
        interval = interval.loc[~bad].astype(int)

    # Build naive timestamps (no tz/DST logic)
    ts = (
        date
        + pd.to_timedelta(hour - 1, unit="h")
        + pd.to_timedelta((interval - 1) * 15, unit="m")
    )
    df["date"] = ts

    # Pivot to wide
    wide = df.pivot_table(
        index="date",
        columns="Settlement Point Name",
        values="Settlement Point Price",
        aggfunc="mean",
    ).sort_index()

    # Validate datetime index explicitly
    wide.index = pd.to_datetime(wide.index, errors="coerce")
    if wide.index.isna().any():
        raise ValueError("pivot produced non-datetime index values")
    if not isinstance(wide.index, pd.DatetimeIndex):
        raise TypeError("pivot index is not DatetimeIndex")

    return wide

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./dataset/ercot")
    ap.add_argument("--keep_types", type=str, default="")
    ap.add_argument("--out", type=str, default="ercot_15min.csv")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    paths = sorted(list(root.rglob("*.csv")) + list(root.rglob("*.xlsx")) + list(root.rglob("*.xls")))
    print(f"[info] searching in: {root}")
    print(f"[info] found {len(paths)} files.")
    for p in paths[:10]:
        print(f"  - {p.relative_to(root)}")
    if not paths:
        raise SystemExit("[error] No files found.")

    keep_types = {s.strip() for s in args.keep_types.split(",") if s.strip()} or None

    frames = []
    for p in paths:
        try:
            wide = load_one(p)
            if keep_types:
                # filter by type using any available type columns—do before pivot if needed
                pass
            if wide.empty:
                print(f"[warn] {p.name}: empty after pivot; skipping.")
                continue
            frames.append(wide)
        except Exception as e:
            print(f"[warn] skipping {p.name}: {e}")

    if not frames:
        raise SystemExit("[error] No usable tables were produced. (Check date/hour/interval parsing.)")

    wide_all = pd.concat(frames).sort_index()

    # Reindex to strict 15-minute grid and fill small gaps
    full_idx = pd.date_range(wide_all.index.min(), wide_all.index.max(), freq="15min")
    wide_all = wide_all.reindex(full_idx)
    wide_all = wide_all.interpolate(limit=8).ffill().bfill()

    out_path = root / args.out
    wide_all.reset_index(inplace=True)
    wide_all.rename(columns={"index": "date"}, inplace=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide_all.to_csv(out_path, index=False)

    print(f"[done] wrote {out_path}  rows={wide_all.shape[0]}  series={wide_all.shape[1]-1}")

if __name__ == "__main__":
    main()
