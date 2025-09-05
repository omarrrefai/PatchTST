# 5_min_data_builder.py
# pip install gridstatusio pandas openpyxl python-dateutil pytz pyarrow

import os, time, json, pathlib, traceback
import pandas as pd
from dateutil.relativedelta import relativedelta
from gridstatusio import GridStatusClient

# ========== CONFIG ==========
TZ = "America/Chicago"
START_ALL = pd.Timestamp("2013-01-01", tz=TZ)
END_ALL   = pd.Timestamp("2025-08-27 23:59:59", tz=TZ)

YEARS = range(2013, 2026)
SETTLEMENT_POINTS = ["HB_NORTH", "HB_HOUSTON", "LZ_NORTH", "LZ_HOUSTON"]

# Datasets
DS_LMP_ADDERS = "ercot_lmp_with_adders_by_settlement_point"   # preferred (5-min SPP-like)
DS_LMP_ONLY   = "ercot_lmp_by_settlement_point"               # fallback LMP (5-min)
DS_ADDERS     = "ercot_real_time_adders_and_reserves"         # ORDC/reliability adders

OUT_DIR   = pathlib.Path("ercot_spp_5min_excel"); OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = pathlib.Path("cache_v6"); CACHE_DIR.mkdir(exist_ok=True)  # new cache to avoid stale empties
DEBUG_DIR = pathlib.Path("debug");    DEBUG_DIR.mkdir(exist_ok=True)

# ORDC start (SPP uses adders >= this date)
ORDC_START = pd.Timestamp("2014-06-01", tz=TZ)

# Column name candidates (we also use dataset schema)
TIME_CANDIDATES  = [
    "sced_timestamp_local","interval_start_local","time_local","timestamp_local",
    "timestamp","time","sced_timestamp_utc","interval_start_utc","time_utc"
]
VALUE_LMPA_CANDS = [
    "settlement_point_price","spp","spp_5m","price_with_adders","price",
    "lmp_with_adders","lmp_w_adders","value"
]
VALUE_LMP_CANDS  = ["lmp","price","settlement_point_price","value"]
ADDER_SUM_CANDS  = ["rtorpa","rtordpa","rtoffpa","ordc_adder","reliability_adder"]

# API safety
CALL_GAP_SEC = 7.0         # one call ~every 7s � < 600/hr
BACKOFF_BASE = 5.0
MAX_RETRIES  = 8

client = GridStatusClient(api_key="5be21b4d1ba242bc800f1d68da76c033")
BASE_URL = "https://api.gridstatus.io/v1"

_last_call = 0.0


# ========== UTILITIES ==========
def rate_limit():
    global _last_call
    now = time.monotonic()
    wait = CALL_GAP_SEC - (now - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()

def cache_path(year, month, sp):
    p = CACHE_DIR / f"{year}" / f"{month:02d}"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{sp}.parquet"

def progress_path(year):
    return CACHE_DIR / f"{year}" / "_progress.json"

def load_prog(year):
    p = progress_path(year)
    return json.loads(p.read_text()) if p.exists() else {}

def save_prog(year, d):
    progress_path(year).write_text(json.dumps(d, indent=2))

def debug_dump(tag, df, dataset_id, year, month, sp):
    try:
        out = DEBUG_DIR / f"dbg_{tag}_{dataset_id}_{year}_{month:02d}_{sp}.json"
        info = {
            "dataset": dataset_id,
            "year": int(year),
            "month": int(month),
            "sp": sp,
            "columns": list(map(str, df.columns)),
            "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
            "head": df.head(5).to_dict(orient="records"),
        }
        out.write_text(json.dumps(info, indent=2, default=str))
        print(f"[debug] wrote {out}")
    except Exception:
        pass


# ---------- schema-aware query ----------
def query_with_schema(dataset_id, start, end, extra_params=None):
    """
    Low-level GET that returns (df, meta, dataset_metadata).
    Requests json_schema='array-of-objects' (valid); if server returns array-of-arrays,
    we rename columns using dataset_metadata['columns'].
    """
    url = f"{BASE_URL}/datasets/{dataset_id}/query"
    params = {
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "limit": 1_000_000,
        "timezone": TZ,
        "return_format": "json",
        "json_schema": "array-of-objects",   # <-- FIXED: valid values: 'array-of-objects' or 'array-of-arrays'
        "cursor": "",
    }
    if extra_params:
        params.update(extra_params)

    for attempt in range(MAX_RETRIES):
        try:
            rate_limit()
            df, meta, dataset_meta = client.get(url, params=params, verbose=True)
            # If we still got array-of-arrays (numeric column names), rename using schema
            if df is not None and len(df.columns) and str(df.columns[0]).isdigit():
                schema_cols = []
                for col in (dataset_meta or {}).get("columns", []):
                    name = col.get("name") or col.get("id") or col.get("column_name")
                    schema_cols.append(name or None)
                if schema_cols and len(schema_cols) >= len(df.columns):
                    df.columns = schema_cols[:len(df.columns)]
            return df, meta, dataset_meta
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                sleep_s = min(BACKOFF_BASE * (2 ** attempt), 120)
                print(f"[fetch {dataset_id}] 429; sleeping {sleep_s}s&")
                time.sleep(sleep_s)
                continue
            raise


def pick_time_col(df, schema=None):
    names = set(map(str, df.columns))
    if schema and schema.get("time_index_column") and schema["time_index_column"] in names:
        return schema["time_index_column"]
    for c in TIME_CANDIDATES:
        if c in names:
            return c
    for c in names:
        lc = c.lower()
        if "local" in lc and ("time" in lc or "timestamp" in lc):
            return c
    for c in names:
        lc = c.lower()
        if "utc" in lc and ("time" in lc or "timestamp" in lc):
            return c
    for c in names:
        lc = c.lower()
        if ("time" in lc or "timestamp" in lc):
            return c
    raise RuntimeError(f"No time column found. Columns: {sorted(names)[:20]}&")


def pick_value_col(df, hints):
    lower = {c.lower(): c for c in df.columns}
    for h in hints:
        if h in df.columns:
            return h
        if h.lower() in lower:
            return lower[h.lower()]
    for c in df.columns:
        lc = str(c).lower()
        if ("price" in lc or "lmp" in lc or "value" in lc):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return c
    nums = df.select_dtypes(include="number").columns
    return nums[0] if len(nums) else None


def discover_sp_field(dataset_id):
    s = max(START_ALL, pd.Timestamp("2019-01-01", tz=TZ))
    e = s + pd.Timedelta(days=7)
    df, meta, schema = query_with_schema(dataset_id, s, e, {})
    if df is None or df.empty:
        e = s + pd.Timedelta(days=31)
        df, meta, schema = query_with_schema(dataset_id, s, e, {})
        if df is None or df.empty:
            raise RuntimeError(f"Probe returned no data for {dataset_id}")

    # Prefer explicit names
    for cand in ["settlement_point_name","settlement_point","location","name","point_name","node","hub","zone"]:
        if cand in df.columns and df[cand].astype(str).str.startswith(("HB_","LZ_")).any():
            print(f"[discover {dataset_id}] SP field = {cand}")
            return cand
    # Heuristic
    for col in df.columns:
        try:
            if df[col].astype(str).str.startswith(("HB_","LZ_")).any():
                print(f"[discover {dataset_id}] SP field (heuristic) = {col}")
                return col
        except Exception:
            pass
    raise RuntimeError(f"Could not find SP field for {dataset_id}. Columns: {list(df.columns)[:20]}")


def normalize_times(df, tcol):
    ts = pd.to_datetime(df[tcol], errors="coerce", utc=("utc" in tcol.lower()))
    if getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_convert(TZ)
    return pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize(TZ)


# ---------- tidy builders ----------
def tidy_from_lmp_adders(df, schema, sp_value):
    if df is None or df.empty:
        return pd.DataFrame(columns=["time_local","sp","spp_5m"])
    tcol = pick_time_col(df, schema)
    vcol = pick_value_col(df, VALUE_LMPA_CANDS)
    if vcol is None:
        debug_dump("no_value_lmpa", df, DS_LMP_ADDERS, 0, 0, sp_value)
        return pd.DataFrame(columns=["time_local","sp","spp_5m"])
    out = pd.DataFrame({
        "time_local": normalize_times(df, tcol),
        "sp": sp_value,  # force the SP we asked for
        "spp_5m": pd.to_numeric(df[vcol], errors="coerce"),
    }).dropna(subset=["time_local"])
    out = out.dropna(subset=["spp_5m"])
    return out[["time_local","sp","spp_5m"]]


def tidy_from_lmp(df, schema, sp_value):
    if df is None or df.empty:
        return pd.DataFrame(columns=["time_local","sp","lmp"])
    tcol = pick_time_col(df, schema)
    vcol = pick_value_col(df, VALUE_LMP_CANDS)
    if vcol is None:
        debug_dump("no_value_lmp", df, DS_LMP_ONLY, 0, 0, sp_value)
        return pd.DataFrame(columns=["time_local","sp","lmp"])
    out = pd.DataFrame({
        "time_local": normalize_times(df, tcol),
        "sp": sp_value,
        "lmp": pd.to_numeric(df[vcol], errors="coerce"),
    }).dropna(subset=["time_local"])
    out = out.dropna(subset=["lmp"])
    return out[["time_local","sp","lmp"]]


def tidy_adders(df, schema):
    if df is None or df.empty:
        return pd.DataFrame(columns=["time_local","adder"])
    tcol = pick_time_col(df, schema)
    ts = normalize_times(df, tcol)
    adder = None
    for c in ADDER_SUM_CANDS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            adder = s if adder is None else adder.add(s, fill_value=0.0)
    if adder is None:
        nums = df.select_dtypes(include="number").columns
        if len(nums) == 0:
            debug_dump("no_numeric_adders", df, DS_ADDERS, 0, 0, "ALL")
            return pd.DataFrame(columns=["time_local","adder"])
        adder = pd.to_numeric(df[nums[0]], errors="coerce").fillna(0.0)
    out = pd.DataFrame({"time_local": ts, "adder": adder}).dropna(subset=["time_local"])
    return out


def join_lmp_adders_to_spp(lmp_df, add_df, start, end):
    if lmp_df.empty:
        return pd.DataFrame(columns=["time_local","sp","spp_5m"])
    add_df = add_df[(add_df["time_local"] >= start) & (add_df["time_local"] <= end)].copy()
    if add_df.empty:
        return lmp_df.rename(columns={"lmp":"spp_5m"})[["time_local","sp","spp_5m"]]
    lmp_df = lmp_df.copy()
    lmp_df["time_local"] = lmp_df["time_local"].dt.floor("5T")
    add_df["time_local"] = add_df["time_local"].dt.floor("5T")
    merged = lmp_df.merge(add_df, on="time_local", how="left")
    merged["adder"] = merged["adder"].fillna(0.0)
    merged["spp_5m"] = merged["lmp"] + merged["adder"]
    return merged[["time_local","sp","spp_5m"]]


# ---------- build & save ----------
def rebuild_year_excel(year):
    y0 = pd.Timestamp(f"{year}-01-01", tz=TZ)
    y1 = pd.Timestamp(f"{year}-12-31 23:59:59", tz=TZ)
    y0 = max(y0, START_ALL); y1 = min(y1, END_ALL)

    parts = []
    ydir = CACHE_DIR / f"{year}"
    if not ydir.exists():
        return
    for mdir in sorted(ydir.glob("[0-1][0-9]")):
        for f in mdir.glob("*.parquet"):
            try:
                parts.append(pd.read_parquet(f))
            except Exception:
                pass
    if not parts:
        return
    tidy = pd.concat(parts, ignore_index=True)
    tidy = tidy.dropna(subset=["time_local","spp_5m"]).sort_values("time_local")
    if tidy.empty:
        return

    wide = (tidy.pivot_table(index="time_local", columns="sp", values="spp_5m",
                             aggfunc="mean", observed=True).sort_index())

    full_idx = pd.date_range(start=y0, end=y1, freq="5T", tz=TZ)
    wide = wide.reindex(full_idx)

    for sp in SETTLEMENT_POINTS:
        if sp not in wide.columns:
            wide[sp] = pd.NA
    wide = wide[SETTLEMENT_POINTS]

    # -> Excel rows in ERCOT-like format
    idx = wide.index
    naive = idx.tz_localize(None)
    key = pd.MultiIndex.from_arrays([naive.date, naive.hour, naive.minute])
    counts = (pd.Series(1, index=key).groupby(level=[0,1,2]).cumsum().reindex(key).values)
    repeated = pd.Series(counts, index=idx).apply(lambda n: "Y" if n and n > 1 else "N")

    rows = []
    for sp in wide.columns:
        sp_type = "HU" if sp.startswith("HB_") else "LZ" if sp.startswith("LZ_") else "RN"
        rows.append(pd.DataFrame({
            "Delivery Date": idx.date,
            "Delivery Hour": idx.hour + 1,
            "Delivery Interval": (idx.minute // 5) + 1,
            "Repeated Hour Flag": repeated.values,
            "Settlement Point Name": sp,
            "Settlement Point Type": sp_type,
            "Settlement Point Price": wide[sp].values,
        }))
    out = pd.concat(rows, ignore_index=True)
    out = out[[
        "Delivery Date","Delivery Hour","Delivery Interval","Repeated Hour Flag",
        "Settlement Point Name","Settlement Point Type","Settlement Point Price"
    ]].sort_values(["Delivery Date","Delivery Hour","Delivery Interval","Settlement Point Name"])

    out_path = OUT_DIR / f"ercot_spp_5min_{year}.xlsx"
    out.to_excel(out_path, index=False)
    print(f"[save] {out_path} rows={len(out):,}")


# ========== MAIN ==========
if __name__ == "__main__":
    # Discover SP field for each dataset (they *can* differ)
    try:
        SP_FIELD_LMPA = discover_sp_field(DS_LMP_ADDERS)
    except Exception as e:
        print("[warn] Could not discover SP field for LMP+Adders; will still force SP from filter value.")
        SP_FIELD_LMPA = None
    try:
        SP_FIELD_LMP = discover_sp_field(DS_LMP_ONLY)
    except Exception as e:
        print("[warn] Could not discover SP field for LMP; will still force SP from filter value.")
        SP_FIELD_LMP = None

    for year in YEARS:
        y0 = pd.Timestamp(f"{year}-01-01", tz=TZ)
        y1 = pd.Timestamp(f"{year}-12-31 23:59:59", tz=TZ)
        y0 = max(y0, START_ALL); y1 = min(y1, END_ALL)
        if y0 > y1:
            continue

        print(f"\n[year] {year}  ({y0.date()} � {y1.date()})")
        prog = load_prog(year)

        cur = y0.normalize()
        while cur <= y1:
            nxt = (cur + relativedelta(months=1)) - pd.Timedelta(seconds=1)
            if nxt > y1:
                nxt = y1
            mkey = f"{cur.year}-{cur.month:02d}"
            prog.setdefault(mkey, {})

            for sp in SETTLEMENT_POINTS:
                cfile = cache_path(cur.year, cur.month, sp)
                if cfile.exists():
                    prog[mkey][sp] = "cached"; continue

                try:
                    # 1) Preferred: 5-min SPP (LMP+adders) dataset
                    extra = {}
                    if SP_FIELD_LMPA:
                        extra = {"filter_column": SP_FIELD_LMPA, "filter_value": sp, "filter_operator": "=",}
                    df1, meta1, schema1 = query_with_schema(DS_LMP_ADDERS, cur, nxt, extra)
                    tidy = tidy_from_lmp_adders(df1, schema1, sp)

                    # 2) Fallback: 5-min LMP (+ adders when applicable)
                    if tidy.empty:
                        extra2 = {}
                        if SP_FIELD_LMP:
                            extra2 = {"filter_column": SP_FIELD_LMP, "filter_value": sp, "filter_operator": "=",}
                        df_lmp, meta2, schema2 = query_with_schema(DS_LMP_ONLY, cur, nxt, extra2)
                        lmp_tidy = tidy_from_lmp(df_lmp, schema2, sp)

                        if not lmp_tidy.empty:
                            if nxt < ORDC_START:
                                tidy = lmp_tidy.rename(columns={"lmp":"spp_5m"})
                            else:
                                df_add, meta3, schema3 = query_with_schema(DS_ADDERS, cur, nxt, {})
                                add_tidy = tidy_adders(df_add, schema3)
                                tidy = join_lmp_adders_to_spp(lmp_tidy, add_tidy, cur, nxt)

                    # 3) Cache immediately (never lose progress)
                    if tidy.empty:
                        # write an empty marker & debug dump
                        pd.DataFrame(columns=["time_local","sp","spp_5m"]).to_parquet(cfile, index=False)
                        prog[mkey][sp] = "empty"
                        save_prog(year, prog)
                        debug_dump("empty_after_all", df1 if df1 is not None else pd.DataFrame(),
                                   DS_LMP_ADDERS, cur.year, cur.month, sp)
                    else:
                        tidy.to_parquet(cfile, index=False)
                        prog[mkey][sp] = f"rows:{len(tidy)}"
                        save_prog(year, prog)

                except Exception as e:
                    prog[mkey][sp] = "error"
                    save_prog(year, prog)
                    print(f"[warn] {sp} {mkey} failed: {e}")
                    traceback.print_exc()

            # Rebuild the year's Excel after each month so it fills as we go
            rebuild_year_excel(year)
            cur = (cur + relativedelta(months=1)).normalize()

    print("\n[done] Completed with schema-driven mapping, fallbacks, caching, and frequent saves.")
