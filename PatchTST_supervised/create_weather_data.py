#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Programmatic API:

    build_area_weather_panels(

        master_csv="/path/to/master.csv",

        out_dir="/path/to/out_dir",

        hourly_vars=["windspeed_10m", "temperature_2m"],

        cache_dir=None,  # optional

        area_latlon_map=None,          # optional: dict {"Area": (lat, lon)}

        area_latlon_csv=None           # optional: path to a CSV with columns: area,lat,lon

    )
 
Behavior:

- Reads 5-min price data from master_csv.

- For each area column (everything not in {timestamp, DELIVERY_DATE, DELIVERY_HOUR, INTERVAL}):

  * Fetches hourly weather from Open-Meteo for full date span (TZ America/Chicago).

  * Repeats hourly values across each hour’s 12×5-min slots (forward-fill).

  * Writes one CSV per area: [timestamp, price, <weather columns>].

"""
 
from __future__ import annotations

from pathlib import Path

import json

import time

import re

from typing import Dict, Tuple, List, Iterable, Optional
 
import pandas as pd

import requests
 
OPENMETEO_BASE = "https://archive-api.open-meteo.com/v1/archive"

TIMEZONE = "America/Chicago"

META_COLS = {"timestamp", "DELIVERY_DATE", "DELIVERY_HOUR", "INTERVAL"}
 
# --------- Default coordinates (edit as needed) ----------

DEFAULT_AREA_LATLON = {

    "Manitoba": (49.8951, -97.1384),      # Winnipeg

    "Manitoba SK": (49.8951, -97.1384),

    "Michigan": (42.3314, -83.0458),      # Detroit

    "Minnesota": (44.9537, -93.0900),     # Saint Paul

    "New-York": (42.6526, -73.7562),      # Albany

    "Ontario": (43.6511, -79.3470),       # Toronto

    # Quebec family -> Montreal fallback

    "Quebec": (45.5017, -73.5673),

    "Quebec AT": (45.5017, -73.5673),

    "Quebec B5D.B31L": (45.5017, -73.5673),

    "Quebec D4Z": (45.5017, -73.5673),

    "Quebec D5A": (45.5017, -73.5673),

    "Quebec H4Z": (45.5017, -73.5673),

    "Quebec H9A": (45.5017, -73.5673),

    "Quebec P33C": (45.5017, -73.5673),

    "Quebec Q4C": (45.5017, -73.5673),

    "Quebec X2Y": (45.5017, -73.5673),

}

# ---------------------------------------------------------
 
 
def slugify(name: str) -> str:

    s = name.strip().lower()

    s = re.sub(r"[^\w\-]+", "_", s)

    s = re.sub(r"_+", "_", s).strip("_")

    return s or "area"
 
 
def _load_area_latlon_csv(path: str | Path) -> Dict[str, Tuple[float, float]]:

    path = Path(path)

    df = pd.read_csv(path)

    need = {"area", "lat", "lon"}

    missing = need - set(df.columns.str.lower())

    if missing:

        raise ValueError(f"CSV must include columns {need}, missing {missing}")

    # Normalize column names

    cols = {c: c.lower() for c in df.columns}

    df = df.rename(columns=cols)

    mapping: Dict[str, Tuple[float, float]] = {}

    for _, row in df.iterrows():

        mapping[str(row["area"]).strip()] = (float(row["lat"]), float(row["lon"]))

    return mapping
 
 
def _infer_area_latlon(area: str,

                       area_latlon_map: Optional[Dict[str, Tuple[float, float]]] = None

                       ) -> Optional[Tuple[float, float]]:

    if area_latlon_map and area in area_latlon_map:

        return area_latlon_map[area]

    if area in DEFAULT_AREA_LATLON:

        return DEFAULT_AREA_LATLON[area]

    if area.startswith("Quebec"):

        return DEFAULT_AREA_LATLON["Quebec"]

    return None
 
 
def _chunk_dates(start_date: str, end_date: str, years_per_chunk: int = 4) -> List[Tuple[str, str]]:

    rng = pd.date_range(start_date, end_date, freq="YS")

    if len(rng) == 0:

        return [(start_date, end_date)]

    chunks = []

    cur = pd.Timestamp(start_date)

    last = pd.Timestamp(end_date)

    while cur <= last:

        end = min(cur + pd.DateOffset(years=years_per_chunk) - pd.DateOffset(days=1), last)

        chunks.append((cur.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))

        cur = end + pd.DateOffset(days=1)

    return chunks
 
 
def _fetch_hourly_weather(lat: float,

                          lon: float,

                          start: str,

                          end: str,

                          hourly_vars: Iterable[str],

                          cache_dir: Optional[str | Path] = None,

                          max_retries: int = 3,

                          pause_sec: float = 1.0) -> pd.DataFrame:

    hourly_vars = list(hourly_vars)

    frames = []

    cache_dir_path = Path(cache_dir) if cache_dir else None
 
    for s, e in _chunk_dates(start, end):

        cache_file = None

        if cache_dir_path:

            cache_dir_path.mkdir(parents=True, exist_ok=True)

            cache_file = cache_dir_path / f"weather_{lat:.4f}_{lon:.4f}_{s}_{e}_{'-'.join(hourly_vars)}.json"

            if cache_file.exists():

                payload = json.loads(cache_file.read_text(encoding="utf-8"))

                df = pd.DataFrame(payload["hourly"])

                df["time"] = pd.to_datetime(df["time"])

                df = df.set_index("time").sort_index()

                frames.append(df)

                continue
 
        params = {

            "latitude": lat,

            "longitude": lon,

            "start_date": s,

            "end_date": e,

            "hourly": ",".join(hourly_vars),

            "timezone": TIMEZONE,

        }
 
        for attempt in range(1, max_retries + 1):

            try:

                r = requests.get(OPENMETEO_BASE, params=params, timeout=60)

                r.raise_for_status()

                payload = r.json()

                if cache_file:

                    cache_file.write_text(json.dumps(payload), encoding="utf-8")

                df = pd.DataFrame(payload["hourly"])

                df["time"] = pd.to_datetime(df["time"])

                df = df.set_index("time").sort_index()

                frames.append(df)

                break

            except Exception:

                if attempt == max_retries:

                    raise

                time.sleep(pause_sec * attempt)
 
    if not frames:

        return pd.DataFrame(index=pd.DatetimeIndex([], name="time"))

    out = pd.concat(frames).sort_index()

    out = out[~out.index.duplicated(keep="first")]

    return out
 
def _repeat_hourly_to_5min(hourly_df: pd.DataFrame, target_index_5min: pd.DatetimeIndex) -> pd.DataFrame:
    hourly_df = hourly_df.sort_index()
    # forward-fill projects each top-of-hour reading to its 5-min slots until the next hour’s reading
    return hourly_df.reindex(target_index_5min, method="ffill")
 
 
def build_area_weather_panels(master_csv: str | Path,
                              out_dir: str | Path,
                              hourly_vars: Iterable[str],
                              cache_dir: Optional[str | Path] = None,
                              area_latlon_map: Optional[Dict[str, Tuple[float, float]]] = None,
                              area_latlon_csv: Optional[str | Path] = None) -> None:
    """
    Create one CSV per area: [timestamp, price, <hourly weather columns repeated to 5-min>].
 
    Args:
        master_csv: Path to 5-min master CSV (must contain 'timestamp' and per-area columns).
        out_dir: Output directory for per-area CSVs.
        hourly_vars: e.g., ["windspeed_10m", "temperature_2m"] or ["windspeed_100m","temperature_10m"].
        cache_dir: Optional folder to cache Open-Meteo responses.
        area_latlon_map: Optional dict mapping area -> (lat, lon).
        area_latlon_csv: Optional CSV path with columns: area, lat, lon (overrides map if given).
    """
    hourly_vars = list(hourly_vars)
    master_csv = Path(master_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    # Load price data
    df = pd.read_csv(master_csv)
    if "timestamp" not in df.columns:
        raise ValueError("Input must have a 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    time_index_5min = pd.DatetimeIndex(df["timestamp"].values, name="timestamp")
 
    # Determine areas
    area_cols = [c for c in df.columns if c not in META_COLS]
    if not area_cols:
        raise ValueError("No area columns found. Area columns are all columns except {timestamp, DELIVERY_DATE, DELIVERY_HOUR, INTERVAL}.")
 
    start_date = df["timestamp"].min().strftime("%Y-%m-%d")
    end_date = df["timestamp"].max().strftime("%Y-%m-%d")
 
    # Area coordinates
    final_map = dict(DEFAULT_AREA_LATLON)
    if area_latlon_map:
        final_map.update(area_latlon_map)
    if area_latlon_csv:
        final_map.update(_load_area_latlon_csv(area_latlon_csv))
 
    print(f"Areas: {len(area_cols)} | Date span: {start_date} → {end_date} | Rows: {len(df)}")
 
    base = df.set_index("timestamp")
 
    for area in area_cols:
        latlon = _infer_area_latlon(area, final_map)
        if latlon is None:
            print(f"[WARN] Missing coordinates for '{area}'. Add to map or CSV. Skipping.")
            continue
        lat, lon = latlon
        print(f"- {area}: ({lat:.4f}, {lon:.4f})")
 
        # Fetch hourly weather across span
        w_hourly = _fetch_hourly_weather(lat, lon, start_date, end_date, hourly_vars, cache_dir)
        if w_hourly.empty:
            print(f"  [WARN] Empty weather for {area}. Skipping.")
            continue
 
        # Expand to 5-min grid (repeat/ffill)
        w_5min = _repeat_hourly_to_5min(w_hourly, time_index_5min)
        w_5min.columns = [str(c) for c in w_5min.columns]
 
        panel = pd.concat(
            [base[area].rename("price"), w_5min],
            axis=1
        )
 
        out_path = out_dir / f"{slugify(area)}.csv"
        panel.reset_index().to_csv(out_path, index=False)
        print(f"  -> wrote {out_path}  shape={panel.shape}")
 
 
# --------------- CONFIG YOU SET (manual usage) ----------------
if __name__ == "__main__":
    # Edit these three (or import and call build_area_weather_panels elsewhere)
    MASTER_CSV = "/home/omaralrefai/dev/PatchTST/.dataset/canada/canada_realtime_ENGY_2010_2025.csv"
    OUT_DIR = "/home/omaralrefai/dev/PatchTST/.dataset/canada"
    HOURLY_VARS = ["windspeed_10m", "temperature_2m"]   # or ["windspeed_100m", "temperature_10m"]
 
    # Optional:
    CACHE_DIR = "/home/omaralrefai/dev/PatchTST/.dataset/canada/.cache_weather"        # or None to disable caching
    AREA_LATLON_CSV = None               # e.g., "./areas_latlon.csv" with columns: area,lat,lon
    EXTRA_AREA_MAP = {
        # "Your Custom Area Name": (lat, lon),
    }
 
    # Run
    build_area_weather_panels(
        master_csv=MASTER_CSV,
        out_dir=OUT_DIR,
        hourly_vars=HOURLY_VARS,
        cache_dir=CACHE_DIR,
        area_latlon_map=EXTRA_AREA_MAP,
        area_latlon_csv=AREA_LATLON_CSV,
    )