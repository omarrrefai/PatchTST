# simulation_api_server.py
# FastAPI + PuLP MPC over historical timeline (20102025), looping forever.
# 3-layer forecasts: DA (288), ST (12), RT (t+1).
# Realized cost uses ACTUAL CSV; MILP uses RT for step-0, ST for next 12, DA beyond.

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading, time
import os
import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import pulp as pl

# ----------------------------
# Config
# ----------------------------
AREAS = ["manitoba", "new-york", "ontario", "quebec_p33c", "manitoba_sk"]

# Files/paths
_DATA_ROOT_CANDIDATES = [
    os.getenv("PATCHTST_DATA_ROOT", "").strip(),
    str(Path(__file__).resolve().parents[1] / ".dataset" / "canada" / "per_area_features_clean"),
    "/home/omaralrefai/dev/PatchTST/.dataset/canada/per_area_features_clean",
]
DATA_ROOT = next((Path(p) for p in _DATA_ROOT_CANDIDATES if p and Path(p).exists()), Path(_DATA_ROOT_CANDIDATES[1]))
RESULTS_ROOT = Path("results")

DA_GLOB  = "PTST_CAN_features_DA288_{area}_PatchTST_custom_ftM_sl576_ll72_pl288_*"
ST_GLOB  = "PTST_CAN_features_ST12_{area}_PatchTST_custom_ftM_sl576_ll72_pl12_*"   # optional
RT_GLOB  = "PTST_CAN_features_t+1_{area}_PatchTST_custom_ftM_sl576_ll72_pl1_*"

PRICE_CH_IDX = -1  # if pred has channels, pick this for price

# Time
FREQ_MIN = 5
DT_HOURS = FREQ_MIN / 60.0
H_DA = 288
H_ST = 12
N30  = 6

# MPC horizon (in 5-min steps)
MPC_H = 24   # 2 hours look-ahead
MODE_CONFIG = {
    "short": {"horizon": 12, "priority": ["RT", "ST", "DA"]},
    "medium": {"horizon": 72, "priority": ["ST", "DA", "RT"]},
    "long": {"horizon": 288, "priority": ["DA", "ST", "RT"]},
}
AUTO_MODE_PERIOD_STEPS = 24  # switch every 2 hours (24x5min)

# Load composition
NONSHIFTABLE_FRACTION = 0.35
AREA_WEIGHTS = {a: 1.0 for a in AREAS}
AREA_PHASE = {a: i * 0.85 for i, a in enumerate(AREAS)}

# Battery (per site)
BATTERY_KWH   = 4000.0
P_CH_MAX_KW   = 2000.0
P_DIS_MAX_KW  = 2000.0
EFF_CH        = 0.95
EFF_DIS       = 0.95
DEG_COST_PER_KWH = 0.003
SOC_MIN_FRAC  = 0.10
E_MIN = SOC_MIN_FRAC * BATTERY_KWH
E_MAX = BATTERY_KWH

# Grid limits/fees
PIMP_MAX_KW = {a: 20000.0 for a in AREAS}
PEXP_MAX_KW = 0.0
TAU_FEE_PER_MWH = {a: 0.0 for a in AREAS}
SELL_PRICE_PER_MWH = 0.0

# Smoothness
HYSTERESIS_PENALTY = 0.02
USE_BINARIES_NO_SIM_CH_DIS = True

# Optional de-normalization hook for predictions
USE_DENO = True

# Step logging (NDJSON) for later offline analysis/paper plots
LOG_DIR = Path(os.getenv("PATCHTST_LOG_DIR", "logs/simulation"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "steps.ndjson"


# ----------------------------
# App & state
# ----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

state = {
    "status": "starting",
    "message": "",
    "idx": 0,                # index into the aligned historical timeline (5-min ticks)
    "timeline": None,        # pandas.DatetimeIndex
    "soc": {a: BATTERY_KWH * 0.5 for a in AREAS},
    "last_import": {a: 0.0 for a in AREAS},

    # price series aligned to timeline
    "actual":   {},          # $/MWh
    "rt":       {},          # $/MWh (t+1)
    "st12":     {},          # $/MWh (12)
    "da288":    {},          # $/MWh (288)

    # total load profile (MW) aligned to timeline
    "total_load": None,

    # cumulative accounting
    "opt_cost_forecast": 0.0,
    "equal_cost_forecast": 0.0,
    "opt_cost_actual": 0.0,
    "equal_cost_actual": 0.0,

    # last-step forecast info for API
    "last_forecast_price": {},
    "last_forecast_src": {},

    # history tails (for UI)
    "history": [],
    "history_steps": [],
    "history_total_load": [],
    "history_total_cost_forecast": [],
    "history_total_savings_forecast": [],
    "history_total_cost_actual": [],
    "history_total_savings_actual": [],
    "history_prediction_source": {a: [] for a in AREAS},
    "history_soc": {a: [] for a in AREAS},
    "history_area_load": {a: [] for a in AREAS},
    "history_area_price": {a: [] for a in AREAS},  # forecast used at each step
    "history_mode": [],

    # optimization method selection
    "mode_selection": "auto",  # auto | short | medium | long
    "active_mode": "short",
}

SOURCE_TO_WINDOW = {
    "RT": "5m (t+1)",
    "ST": "60m (12x5m)",
    "DA": "24h (288x5m)",
    "FWD": "carry-forward",
    "ACT": "actual fallback",
    "?": "unknown",
}

# ----------------------------
# Utilities (loading & mapping)
# ----------------------------
def _latest_result_dir(glob_pat: str) -> Path | None:
    cands = sorted(
        (p for p in RESULTS_ROOT.glob(glob_pat) if (p / "pred.npy").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    # Return the NEWEST, not the oldest
    return cands[0] if cands else None


def _load_actual_series(area: str) -> pd.Series:
    p = DATA_ROOT / f"{area}_clean.csv"
    if not p.is_file():
        return pd.Series(dtype="float64")
    df = pd.read_csv(p, parse_dates=["date"])
    if "price" not in df.columns:
        return pd.Series(dtype="float64")
    s = df.sort_values("date").set_index("date")["price"].astype("float64")
    s = s[~s.index.duplicated(keep="last")]
    return s

def _load_st_matrix(area: str) -> tuple[np.ndarray, pd.DatetimeIndex] | None:
    resdir = _latest_result_dir(ST_GLOB.format(area=area))
    if resdir is None:
        return None
    pred = np.asarray(np.load(resdir / "pred.npy"))
    # squeeze to (W,12,C?)
    if pred.ndim == 2 and pred.shape[1] == H_ST:
        mat = pred[:, :, np.newaxis]
    elif pred.ndim == 3 and pred.shape[1] == H_ST:
        mat = pred
    elif pred.ndim == 4 and pred.shape[2] == H_ST:
        mat = pred[-1]
    else:
        return None
    mat = _maybe_denorm(area, mat, resdir)

    if mat.shape[2] == 1:
        mat2 = mat[:, :, 0]
    else:
        s_actual = _load_actual_series(area).dropna()
        y = np.asarray([np.nanmean(s_actual.values[-H_ST:]) for _ in range(mat.shape[0])])
        corrs = []
        for c in range(mat.shape[2]):
            x = np.nanmean(mat[:, :, c], axis=1)
            corrs.append(-np.inf if np.allclose(x.std(), 0) or np.allclose(y.std(), 0) else np.corrcoef(x, y)[0, 1])
        mat2 = mat[:, :, int(np.nanargmax(corrs))]

    s = _load_actual_series(area)
    hour_starts = pd.date_range(s.index.min().floor("H"), s.index.max().floor("H"), freq="H")
    W = mat2.shape[0]
    if len(hour_starts) < W:
        hour_starts = pd.date_range(hour_starts.min(), periods=W, freq="H")
    anchors = hour_starts[-W:]
    return mat2, anchors

def _load_rt_vector(area: str) -> np.ndarray:
    resdir = _latest_result_dir(RT_GLOB.format(area=area))
    if resdir is None:
        raise FileNotFoundError(f"RT results not found for {area}")
    arr = np.asarray(np.load(resdir / "pred.npy"))
    # squeeze batch/time -> (W,C) or (W,)
    if arr.ndim == 1:
        vec = arr
    elif arr.ndim == 2:
        vec = arr
    elif arr.ndim == 3:
        vec = arr[-1]
    else:
        raise ValueError(f"{area} RT pred shape {arr.shape} unsupported")
    vec = _maybe_denorm(area, vec, resdir)
    if vec.ndim == 1:
        return vec.astype(float)

    # pick channel via correlation vs actual
    W, C = vec.shape
    s_actual = _load_actual_series(area).dropna()
    if len(s_actual) < W:
        W_use = len(s_actual)
        X = vec[-W_use:]
        y = s_actual.to_numpy()[-W_use:]
    else:
        X = vec[-W:]
        y = s_actual.to_numpy()[-W:]
    try:
        corrs = []
        for c in range(X.shape[1]):
            xc = X[:, c].astype(float)
            corrs.append(-np.inf if np.allclose(xc.std(), 0) or np.allclose(y.std(), 0) else np.corrcoef(xc, y)[0, 1])
        best_c = int(np.nanargmax(corrs))
        return vec[:, best_c].astype(float)
    except Exception:
        return vec[:, PRICE_CH_IDX].astype(float)

def _align_series_to_timeline():
    # prefer full actual coverage
    actual_raw = {a: _load_actual_series(a).dropna() for a in AREAS}
    have_actual = [a for a, s in actual_raw.items() if len(s)]
    if len(have_actual) == len(AREAS):
        start = max(s.index.min() for s in actual_raw.values())
        end   = min(s.index.max() for s in actual_raw.values())
        if pd.notnull(start) and pd.notnull(end) and start < end:
            idx = pd.date_range(start.floor("5min"), end.ceil("5min"), freq="5min")
            actual = {}
            for a, s in actual_raw.items():
                ss = s.reindex(idx).interpolate(limit=6, limit_direction="both")
                ss = ss.fillna(method="ffill").fillna(method="bfill")
                actual[a] = ss.to_numpy(dtype=float)

            # align available forecasts to same real timeline (do not leave all-NaN)
            rt = {a: np.full(len(idx), np.nan, float) for a in AREAS}
            st = {a: np.full(len(idx), np.nan, float) for a in AREAS}
            da = {a: np.full(len(idx), np.nan, float) for a in AREAS}

            for a in AREAS:
                try:
                    vec = _load_rt_vector(a)
                    take = min(len(vec), len(idx))
                    if take > 0:
                        rt[a][-take:] = vec[-take:].astype(float)
                except Exception:
                    pass

            for a in AREAS:
                try:
                    mat, anchors = _load_da_matrix(a)
                    for k, day0 in enumerate(anchors):
                        pos = idx.get_indexer([pd.Timestamp(day0).normalize()])[0]
                        if pos >= 0 and pos + 288 <= len(idx):
                            da[a][pos:pos+288] = mat[k]
                except Exception:
                    pass

            for a in AREAS:
                try:
                    st_loaded = _load_st_matrix(a)
                    if st_loaded:
                        mat, anchors = st_loaded
                        for k, h0 in enumerate(anchors):
                            pos = idx.get_indexer([pd.Timestamp(h0).floor("H")])[0]
                            if pos >= 0 and pos + 12 <= len(idx):
                                st[a][pos:pos+12] = mat[k]
                except Exception:
                    pass

            # synthetic load
            t = np.arange(len(idx))
            diurnal = 2.2 * np.sin(2*np.pi*(t % 288)/288 - np.pi/2)
            weekly  = 0.7 * np.sin(2*np.pi*(t % (288*7))/(288*7))
            total_load = np.clip(20.0 + diurnal + weekly, 5.0, None).astype(float)
            return idx, actual, rt, st, da, total_load

    # fallback: timeline from forecasts
    da_len = st_len = rt_len = 0
    for a in AREAS:
        try:
            mat, _ = _load_da_matrix(a)
            da_len = max(da_len, int(mat.shape[0] * 288))
        except Exception:
            pass
        try:
            st_loaded = _load_st_matrix(a)
            if st_loaded:
                M, H = st_loaded[0].shape
                st_len = max(st_len, int(M * H))
        except Exception:
            pass
        try:
            vec = _load_rt_vector(a)
            rt_len = max(rt_len, int(vec.shape[0]))
        except Exception:
            pass

    total_steps = max(da_len, st_len, rt_len)
    if total_steps <= 0:
        raise RuntimeError("No actuals and no forecasts found. Check CSV/result paths.")

    start = pd.Timestamp("2010-01-01 00:00:00")
    idx   = pd.date_range(start=start, periods=total_steps, freq="5min")

    actual = {a: np.full(len(idx), np.nan, float) for a in AREAS}
    rt     = {a: np.full(len(idx), np.nan, float) for a in AREAS}
    st     = {a: np.full(len(idx), np.nan, float) for a in AREAS}
    da     = {a: np.full(len(idx), np.nan, float) for a in AREAS}

    # fill RT tails
    for a in AREAS:
        try:
            vec = _load_rt_vector(a)
            rt[a][-len(vec):] = vec.astype(float)
        except Exception:
            pass
    # stitch DA by day
    # stitch DA by day
    for a in AREAS:
        try:
            mat, anchors = _load_da_matrix(a)  # (W,288)
            for k, day0 in enumerate(anchors):
                pos = idx.get_indexer([pd.Timestamp(day0).normalize()])[0]
                if pos >= 0 and pos + 288 <= len(idx):
                    da[a][pos:pos+288] = mat[k]
        except Exception as e:
            # log message somewhere if needed
            pass
    # stitch ST by hour
    for a in AREAS:
        try:
            st_loaded = _load_st_matrix(a)
            if st_loaded:
                mat, anchors = st_loaded
                for k, h0 in enumerate(anchors):
                    pos = idx.get_indexer([pd.Timestamp(h0).floor("H")])[0]
                    if pos >= 0 and pos + 12 <= len(idx):
                        st[a][pos:pos+12] = mat[k]
        except Exception:
            pass

    t = np.arange(len(idx))
    diurnal = 2.2 * np.sin(2*np.pi*(t % 288)/288 - np.pi/2)
    weekly  = 0.7 * np.sin(2*np.pi*(t % (288*7))/(288*7))
    total_load = np.clip(20.0 + diurnal + weekly, 5.0, None).astype(float)

    return idx, actual, rt, st, da, total_load

def _mode_for_step(tpos: int) -> str:
    sel = state.get("mode_selection", "auto")
    if sel in MODE_CONFIG:
        return sel
    bucket = (tpos // AUTO_MODE_PERIOD_STEPS) % 3
    return ["short", "medium", "long"][bucket]


def _per_area_load_target_kw(total_mw: float, tpos: int) -> dict:
    """Area-varying target load profile so each DC has dynamic demand over time."""
    total_kw = total_mw * 1000.0
    raw = {}
    for i, a in enumerate(AREAS):
        base = AREA_WEIGHTS.get(a, 1.0)
        # deterministic multi-frequency modulation per area
        mod = 1.0 + 0.20 * np.sin(2 * np.pi * (tpos % 288) / 288 + AREA_PHASE[a])
        mod += 0.10 * np.sin(2 * np.pi * (tpos % (288 * 7)) / (288 * 7) + 0.37 * i)
        raw[a] = max(0.2, base * mod)
    norm = sum(raw.values())
    return {a: total_kw * raw[a] / norm for a in AREAS}


def _floor_per_area_kw(total_mw: float, tpos: int) -> dict:
    target_kw = _per_area_load_target_kw(total_mw, tpos)
    return {a: NONSHIFTABLE_FRACTION * target_kw[a] for a in AREAS}

# ----------------------------
# Forecast stack (strict, no ACTUAL fallback)
# ----------------------------
def _price_for_horizon(a: str, tpos: int, mode: str) -> tuple[np.ndarray, str]:
    H = MODE_CONFIG[mode]["horizon"]
    out = np.empty(H, dtype=float)
    out[:] = np.nan
    src0 = "?"

    pri = MODE_CONFIG[mode]["priority"]

    for h in range(H):
        i = tpos + h
        cand = np.nan
        tag_used = None

        # keep horizon-aware behavior while allowing mode-specific source priority
        allowed = set(pri)
        if h >= 12 and "RT" in allowed:
            allowed.remove("RT")
        if h >= 288 and "ST" in allowed:
            allowed.remove("ST")

        for tg in pri:
            if tg not in allowed:
                continue
            if tg == "RT":
                v = state["rt"][a][i] if 0 <= i < len(state["rt"][a]) else np.nan
            elif tg == "ST":
                v = state["st12"][a][i] if 0 <= i < len(state["st12"][a]) else np.nan
            else:
                v = state["da288"][a][i] if 0 <= i < len(state["da288"][a]) else np.nan
            if np.isfinite(v):
                cand = float(v)
                tag_used = tg
                break

        # fallback: forward-fill then actual
        if not np.isfinite(cand):
            if h > 0 and np.isfinite(out[h - 1]):
                cand = out[h - 1]
                tag_used = "FWD"
            else:
                v_act = state["actual"][a][i] if 0 <= i < len(state["actual"][a]) else np.nan
                if np.isfinite(v_act):
                    cand = float(v_act)
                    tag_used = "ACT"
                else:
                    raise RuntimeError(f"No forecast available for {a} at index {i} (h={h})")

        out[h] = cand
        if h == 0:
            src0 = tag_used or "?"

    return out, src0

def _maybe_denorm(area: str, arr: np.ndarray, resdir: Path) -> np.ndarray:
    if not USE_DENO: 
        return arr
    # Try common scaler names
    for name in [f"{area}_scaler.json", "scaler.json", "target_scaler.json"]:
        js = resdir / name
        if js.exists():
            try:
                s = pd.read_json(js)
                mu = float(s["mean"].iloc[0])
                sd = float(s["std"].iloc[0])
                if sd > 0:
                    return arr * sd + mu
            except Exception:
                pass
    return arr

def _load_da_matrix(area: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    resdir = _latest_result_dir(DA_GLOB.format(area=area))
    if resdir is None:
        raise FileNotFoundError(f"DA results not found for {area}")

    pred = np.asarray(np.load(resdir / "pred.npy"))
    # Normalize to (W,288,C)
    if pred.ndim == 1 and pred.shape[0] == H_DA:
        mat = pred[np.newaxis, :, np.newaxis]
    elif pred.ndim == 2 and pred.shape[1] == H_DA:
        mat = pred[:, :, np.newaxis]
    elif pred.ndim == 3 and pred.shape[1] == H_DA:
        mat = pred
    else:
        raise ValueError(f"{area} DA pred.npy unexpected shape {pred.shape}")

    mat = _maybe_denorm(area, mat, resdir)  # de-norm if scaler provided

    # Channel pick
    if mat.shape[2] == 1:
        mat2 = mat[:, :, 0]
    else:
        # choose channel most correlated with actual
        s_actual = _load_actual_series(area).dropna()
        if len(s_actual) >= H_DA:
            # match daily means across windows
            y = np.array([s_actual[-H_DA:].mean()] * mat.shape[0])
            scores = []
            for c in range(mat.shape[2]):
                x = mat[:, :, c].mean(axis=1)
                scores.append(np.corrcoef(x, y)[0,1] if (np.std(x)>0 and np.std(y)>0) else -np.inf)
            best_c = int(np.nanargmax(scores))
            mat2 = mat[:, :, best_c]
        else:
            # fallback to last channel
            mat2 = mat[:, :, PRICE_CH_IDX]
    # build anchors from actual daily range (stable)
    s = _load_actual_series(area)
    day_starts = pd.date_range(s.index.min().normalize(), s.index.max().normalize(), freq="D")
    if len(day_starts) < mat2.shape[0]:
        day_starts = pd.date_range(day_starts.min(), periods=mat2.shape[0], freq="D")
    anchors = day_starts[-mat2.shape[0]:]
    return mat2, anchors


# ----------------------------
# MPC (one step apply)
# ----------------------------
def run_mpc_step(tpos: int) -> dict:
    mode = _mode_for_step(tpos)
    state["active_mode"] = mode
    H = min(MODE_CONFIG[mode]["horizon"], len(state["timeline"]) - tpos)
    dt = DT_HOURS

    model = pl.LpProblem("MPC_3layer", pl.LpMinimize)

    p_imp = {(a,h): pl.LpVariable(f"imp_{a}_{h}", lowBound=0, upBound=PIMP_MAX_KW[a]) for a in AREAS for h in range(H)}
    p_ch  = {(a,h): pl.LpVariable(f"ch_{a}_{h}",  lowBound=0, upBound=P_CH_MAX_KW)   for a in AREAS for h in range(H)}
    p_dis = {(a,h): pl.LpVariable(f"dis_{a}_{h}", lowBound=0, upBound=P_DIS_MAX_KW)  for a in AREAS for h in range(H)}
    p_exp = {h: pl.LpVariable(f"exp_{h}", lowBound=0, upBound=PEXP_MAX_KW) for h in range(H)}
    soc   = {(a,h): pl.LpVariable(f"soc_{a}_{h}", lowBound=E_MIN, upBound=E_MAX) for a in AREAS for h in range(H+1)}

    if USE_BINARIES_NO_SIM_CH_DIS:
        z_ch  = {(a,h): pl.LpVariable(f"zch_{a}_{h}", lowBound=0, upBound=1, cat="Binary") for a in AREAS for h in range(H)}
        z_dis = {(a,h): pl.LpVariable(f"zdis_{a}_{h}", lowBound=0, upBound=1, cat="Binary") for a in AREAS for h in range(H)}
        for a in AREAS:
            for h in range(H):
                model += p_ch[(a,h)]  <= P_CH_MAX_KW  * z_ch[(a,h)]
                model += p_dis[(a,h)] <= P_DIS_MAX_KW * z_dis[(a,h)]
                model += z_ch[(a,h)] + z_dis[(a,h)] <= 1.0

    # init SoC
    for a in AREAS:
        model += soc[(a,0)] == state["soc"][a]

    # hysteresis on first step imports
    v_pos = {a: pl.LpVariable(f"vpos_{a}", lowBound=0) for a in AREAS}
    v_neg = {a: pl.LpVariable(f"vneg_{a}", lowBound=0) for a in AREAS}
    for a in AREAS:
        model += (p_imp[(a,0)] - state["last_import"][a]) == (v_pos[a] - v_neg[a])

    # constraints over horizon
    for h in range(H):
        L_kw = state["total_load"][tpos+h] * 1000.0
        floor_kw = _floor_per_area_kw(state["total_load"][tpos+h], tpos+h)

        # battery dynamics
        for a in AREAS:
            model += soc[(a,h+1)] == soc[(a,h)] + dt*(EFF_CH*p_ch[(a,h)] - (1.0/EFF_DIS)*p_dis[(a,h)])

        # power balance
        model += (
            pl.lpSum(p_imp[(a,h)] for a in AREAS)
            + pl.lpSum(p_dis[(a,h)] for a in AREAS)
            - pl.lpSum(p_ch[(a,h)] for a in AREAS)
            == L_kw + p_exp[h]
        )

        # non-shiftable floors
        for a in AREAS:
            model += p_imp[(a,h)] >= floor_kw[a]

    # terminal SoC freedom (�30min)
    for a in AREAS:
        E0 = state["soc"][a]
        down = N30 * dt * (P_DIS_MAX_KW / EFF_DIS)
        up   = N30 * dt * (EFF_CH * P_CH_MAX_KW)
        model += soc[(a,H)] >= max(E_MIN, E0 - down)
        model += soc[(a,H)] <= min(E_MAX, E0 + up)

    # objective
    price_stack = {a: _price_for_horizon(a, tpos, mode)[0] for a in AREAS}
    terms = []
    for h in range(H):
        for a in AREAS:
            lam = float(price_stack[a][h])
            tau = float(TAU_FEE_PER_MWH[a])
            terms.append(p_imp[(a,h)] * (dt * (lam + tau) / 1000.0))
            terms.append((DEG_COST_PER_KWH * dt) * (p_ch[(a,h)] + p_dis[(a,h)]))
        if SELL_PRICE_PER_MWH > 0:
            terms.append(-(p_exp[h] * (dt * SELL_PRICE_PER_MWH / 1000.0)))
    for a in AREAS:
        terms.append(HYSTERESIS_PENALTY * (v_pos[a] + v_neg[a]))
    model += pl.lpSum(terms)

    model.solve(pl.PULP_CBC_CMD(msg=0))

    # apply h=0
    result = {"area_load": {}, "battery": {}, "power_balance": {}}
    imp_sum = ch_sum = dis_sum = 0.0
    for a in AREAS:
        imp_kw = float(p_imp[(a,0)].value() or 0.0)
        ch_kw  = float(p_ch[(a,0)].value() or 0.0)
        dis_kw = float(p_dis[(a,0)].value() or 0.0)
        result["area_load"][a] = imp_kw / 1000.0
        imp_sum += imp_kw; ch_sum += ch_kw; dis_sum += dis_kw
        state["soc"][a] = float(soc[(a,1)].value() or state["soc"][a])
        state["last_import"][a] = imp_kw

    exp_kw = float(p_exp[0].value() or 0.0)
    L_kw   = state["total_load"][tpos] * 1000.0
    residual = imp_sum + (dis_sum - ch_sum) - (L_kw + exp_kw)
    result["battery"] = {"charge_kw": ch_sum, "discharge_kw": dis_sum, "net_kw": dis_sum - ch_sum}
    result["power_balance"] = {
        "imports_kw": imp_sum,
        "battery_net_kw": dis_sum - ch_sum,
        "export_kw": exp_kw,
        "load_kw": L_kw,
        "residual_kw": residual,
    }

    # record forecast price and source used at h=0 for API
    step_forecast_0 = {}
    step_forecast_src = {}
    for a in AREAS:
        vec, src0 = _price_for_horizon(a, tpos, mode)
        step_forecast_0[a] = float(vec[0])
        step_forecast_src[a] = src0
    state["last_forecast_price"] = step_forecast_0
    state["last_forecast_src"]   = step_forecast_src
    prediction_window = {a: SOURCE_TO_WINDOW.get(step_forecast_src[a], "unknown") for a in AREAS}

    # costs (step)
    dt_energy = DT_HOURS / 1000.0
    step_cost_forecast = 0.0
    step_cost_equal_forecast = 0.0
    step_cost_actual = 0.0
    step_cost_equal_actual = 0.0

    for a in AREAS:
        lamF = float(step_forecast_0[a])
        lamA = float(state["actual"][a][tpos]) if np.isfinite(state["actual"][a][tpos]) else lamF
        tau  = float(TAU_FEE_PER_MWH[a])
        imp_kw = result["area_load"][a] * 1000.0
        step_cost_forecast += imp_kw * dt_energy * (lamF + tau)
        step_cost_actual   += imp_kw * dt_energy * (lamA + tau)

    deg_cost = (DEG_COST_PER_KWH * DT_HOURS) * (ch_sum + dis_sum)
    step_cost_forecast += deg_cost
    step_cost_actual   += deg_cost

    per_kw = L_kw / len(AREAS)
    for a in AREAS:
        lamF = float(step_forecast_0[a])
        lamA = float(state["actual"][a][tpos]) if np.isfinite(state["actual"][a][tpos]) else lamF
        tau  = float(TAU_FEE_PER_MWH[a])
        step_cost_equal_forecast += per_kw * dt_energy * (lamF + tau)
        step_cost_equal_actual   += per_kw * dt_energy * (lamA + tau)

    result.update({
        "step_cost_forecast": step_cost_forecast,
        "step_cost_equal_forecast": step_cost_equal_forecast,
        "step_cost_actual": step_cost_actual,
        "step_cost_equal_actual": step_cost_equal_actual,
        "areaPriceForecast": step_forecast_0,
        "areaPriceActual": {
            a: float(state["actual"][a][tpos]) if np.isfinite(state["actual"][a][tpos]) else float(step_forecast_0[a])
            for a in AREAS
        },
        "forecastSource": step_forecast_src,
        "predictionWindow": prediction_window,
        "optimizationMode": mode,
        "horizonSteps": H,
        "stepCosts": {
            "forecast_opt": step_cost_forecast,
            "forecast_equal": step_cost_equal_forecast,
            "forecast_savings": step_cost_equal_forecast - step_cost_forecast,
            "realized_opt": step_cost_actual,
            "realized_equal": step_cost_equal_actual,
            "realized_savings": step_cost_equal_actual - step_cost_actual,
            "degradation": deg_cost,
        },
    })
    return result


def _append_step_log(t: int, res: dict) -> None:
    ts = state["timeline"][t]
    rec = {
        "step": int(t),
        "timestamp": ts.isoformat(),
        "mode_selection": state.get("mode_selection", "auto"),
        "active_mode": state.get("active_mode", "short"),
        "total_load_mw": float(state["total_load"][t]),
        "cost_forecast_cum": float(state["opt_cost_forecast"]),
        "cost_equal_forecast_cum": float(state["equal_cost_forecast"]),
        "cost_actual_cum": float(state["opt_cost_actual"]),
        "cost_equal_actual_cum": float(state["equal_cost_actual"]),
        "step_costs": {
            "forecast_opt": float(res.get("step_cost_forecast", 0.0)),
            "forecast_equal": float(res.get("step_cost_equal_forecast", 0.0)),
            "actual_opt": float(res.get("step_cost_actual", 0.0)),
            "actual_equal": float(res.get("step_cost_equal_actual", 0.0)),
        },
        "area_load_mw": {a: float(res.get("area_load", {}).get(a, 0.0)) for a in AREAS},
        "area_price_forecast": {a: float(state.get("last_forecast_price", {}).get(a, np.nan)) for a in AREAS},
        "area_price_actual": {a: float(state["actual"][a][t]) if np.isfinite(state["actual"][a][t]) else np.nan for a in AREAS},
        "forecast_source": {a: state.get("last_forecast_src", {}).get(a, "?") for a in AREAS},
        "prediction_window": {
            a: SOURCE_TO_WINDOW.get(state.get("last_forecast_src", {}).get(a, "?"), "unknown")
            for a in AREAS
        },
        "soc_percent": {a: float(state["soc"][a]) / E_MAX * 100.0 for a in AREAS},
        "battery": res.get("battery", {}),
        "power_balance": res.get("power_balance", {}),
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ----------------------------
# API
# ----------------------------
@app.get("/api/health")
def health():
    try:
        idx = state.get("timeline", None)
        return JSONResponse({
            "status": state.get("status"),
            "message": state.get("message",""),
            "timeline_len": int(len(idx)) if idx is not None else 0,
            "idx_min": str(idx[0]) if idx is not None and len(idx) else None,
            "idx_max": str(idx[-1]) if idx is not None and len(idx) else None,
            "areas": AREAS,
            "actual_len": {a: int(len(state["actual"].get(a, []))) for a in AREAS},
            "rt_has": {a: bool(np.isfinite(state["rt"].get(a, np.array([]))).any()) for a in AREAS},
            "st_has": {a: bool(np.isfinite(state["st12"].get(a, np.array([]))).any()) for a in AREAS},
            "da_has": {a: bool(np.isfinite(state["da288"].get(a, np.array([]))).any()) for a in AREAS},
        })
    except Exception as e:
        return JSONResponse({"status":"error","message":f"{type(e).__name__}: {e}"})

@app.get("/api/simulation_data.json")
def get_live():
    if state["status"] != "running" and not state["history"]:
        return JSONResponse({"status": state["status"], "message": state.get("message","")})

    t = int(state["idx"]) if state["idx"] < len(state["timeline"]) else len(state["timeline"]) - 1
    last = state["history"][-1] if state["history"] else {"area_load": {a:0.0 for a in AREAS}, "battery": {}, "power_balance": {}}

    ts = state["timeline"][t]
    time_obj = {"date": ts.strftime("%Y-%m-%d"), "hhmm": ts.strftime("%H:%M")}

    price_forecast = {a: float(state.get("last_forecast_price", {}).get(a, np.nan)) for a in AREAS}
    price_actual   = {a: float(state["actual"][a][t]) if a in state["actual"] and t < len(state["actual"][a]) and np.isfinite(state["actual"][a][t]) else np.nan for a in AREAS}
    price_src      = {a: state.get("last_forecast_src", {}).get(a, "?") for a in AREAS}
    prediction_window = {a: SOURCE_TO_WINDOW.get(price_src[a], "unknown") for a in AREAS}

    W = 60
    payload = {
        "status": state["status"],
        "message": state.get("message",""),
        "step": t,
        "time": time_obj,
        "totalLoad": float(state["total_load"][t]) if state.get("total_load") is not None else 0.0,
        "totalCost": float(state["opt_cost_forecast"]),
        "totalSavings": float(state["equal_cost_forecast"] - state["opt_cost_forecast"]),
        "realizedCost": float(state["opt_cost_actual"]),
        "realizedSavings": float(state["equal_cost_actual"] - state["opt_cost_actual"]),
        "activeOptimizationMode": state.get("active_mode", "short"),
        "modeSelection": state.get("mode_selection", "auto"),
        "costComparison": {
            "withForecast_forecastBasis": float(state["opt_cost_forecast"]),
            "withoutForecast_forecastBasis": float(state["equal_cost_forecast"]),
            "withForecast_actualBasis": float(state["opt_cost_actual"]),
            "withoutForecast_actualBasis": float(state["equal_cost_actual"]),
        },
        "perAreaLoad": last["area_load"],
        "areaPriceForecast": price_forecast,
        "areaPriceActual":   price_actual,
        "forecastSource":    price_src,
        "predictionWindow":  prediction_window,
        "soc": {a: float(state["soc"][a]) / E_MAX * 100.0 for a in AREAS},
        "battery": last.get("battery", {"charge_kw":0,"discharge_kw":0,"net_kw":0}),
        "power_balance": last.get("power_balance", {}),
        "hist": {
            "steps": state["history_steps"][-W:],
            "total_load": state["history_total_load"][-W:],
            "total_cost_forecast": state["history_total_cost_forecast"][-W:],
            "total_savings_forecast": state["history_total_savings_forecast"][-W:],
            "total_cost_actual": state["history_total_cost_actual"][-W:],
            "total_savings_actual": state["history_total_savings_actual"][-W:],
            "soc": {a: state["history_soc"][a][-W:] for a in AREAS},
            "area_load": {a: state["history_area_load"][a][-W:] for a in AREAS},
            "area_price": {a: state["history_area_price"][a][-W:] for a in AREAS},
            "prediction_source": {a: state["history_prediction_source"][a][-W:] for a in AREAS},
            "mode": state["history_mode"][-W:],
        },
        "current": {
            "perAreaLoadMW": last["area_load"],
            "areaPriceForecast": price_forecast,
            "areaPriceActual": price_actual,
            "stepCosts": {
                "forecast_opt": float(last.get("step_cost_forecast", 0.0)),
                "forecast_equal": float(last.get("step_cost_equal_forecast", 0.0)),
                "forecast_savings": float(last.get("step_cost_equal_forecast", 0.0) - last.get("step_cost_forecast", 0.0)),
                "realized_opt": float(last.get("step_cost_actual", 0.0)),
                "realized_equal": float(last.get("step_cost_equal_actual", 0.0)),
                "realized_savings": float(last.get("step_cost_equal_actual", 0.0) - last.get("step_cost_actual", 0.0)),
                "degradation": float((DEG_COST_PER_KWH * DT_HOURS) * ((last.get("battery",{}).get("charge_kw",0.0)) + (last.get("battery",{}).get("discharge_kw",0.0)))),
            },
            "battery": last.get("battery", {}),
            "power_balance": last.get("power_balance", {}),
            "time": time_obj,
            "forecastSource": price_src,
            "predictionWindow": prediction_window,
            "optimizationMode": state.get("active_mode", "short"),
        },
        "previous": state["history"][-2] if len(state["history"]) >= 2 else None,
    }
    return JSONResponse(payload)




@app.get("/api/log_status")
def log_status():
    return JSONResponse({
        "logFile": str(LOG_FILE),
        "exists": LOG_FILE.exists(),
        "sizeBytes": int(LOG_FILE.stat().st_size) if LOG_FILE.exists() else 0,
    })

@app.get("/api/mode")
def get_mode():
    return JSONResponse({
        "modeSelection": state.get("mode_selection", "auto"),
        "activeMode": state.get("active_mode", "short"),
        "availableModes": ["auto", "short", "medium", "long"],
        "modeConfig": MODE_CONFIG,
    })


@app.post("/api/mode/{mode}")
def set_mode(mode: str):
    mode = mode.lower().strip()
    if mode not in {"auto", "short", "medium", "long"}:
        return JSONResponse({"status": "error", "message": f"Invalid mode '{mode}'"}, status_code=400)
    state["mode_selection"] = mode
    return JSONResponse({"status": "ok", "modeSelection": mode})

# ----------------------------
# Simulation loop (forever; 5s = 5min)
# ----------------------------
def simulate_loop():
    while True:
        try:
            idx, actual, rt, st, da, total_load = _align_series_to_timeline()
            state["timeline"]     = idx
            state["actual"]       = actual
            state["rt"]           = rt
            state["st12"]         = st
            state["da288"]        = da
            state["total_load"]   = total_load
            state["status"] = "running"
            state["message"] = ""

            while True:
                t = state["idx"]
                try:
                    res = run_mpc_step(t)
                except Exception as step_err:
                    state["status"] = "running"
                    state["message"] = f"step_error@{t}: {type(step_err).__name__}: {step_err}"
                    time.sleep(1)
                    state["idx"] = (t + 1) % len(state["timeline"])
                    continue

                state["history"].append(res)

                # accumulate costs
                state["opt_cost_forecast"]   += res["step_cost_forecast"]
                state["equal_cost_forecast"] += res["step_cost_equal_forecast"]
                state["opt_cost_actual"]     += res["step_cost_actual"]
                state["equal_cost_actual"]   += res["step_cost_equal_actual"]

                # history tails for charts
                state["history_steps"].append(t)
                state["history_total_load"].append(float(state["total_load"][t]))
                state["history_total_cost_forecast"].append(float(state["opt_cost_forecast"]))
                state["history_total_savings_forecast"].append(float(state["equal_cost_forecast"] - state["opt_cost_forecast"]))
                state["history_total_cost_actual"].append(float(state["opt_cost_actual"]))
                state["history_total_savings_actual"].append(float(state["equal_cost_actual"] - state["opt_cost_actual"]))
                state["history_mode"].append(state.get("active_mode", "short"))
                for a in AREAS:
                    state["history_soc"][a].append(float(state["soc"][a]) / E_MAX * 100.0)
                    state["history_area_load"][a].append(float(res["area_load"][a]))
                    state["history_area_price"][a].append(float(state["last_forecast_price"][a]))
                    state["history_prediction_source"][a].append(state["last_forecast_src"][a])

                # persist machine-consumable step record for offline analysis
                _append_step_log(t, res)

                # advance 1 step; loop forever
                state["idx"] = (t + 1) % len(state["timeline"])
                time.sleep(5)

        except Exception as e:
            state["status"] = "error"
            state["message"] = f"{type(e).__name__}: {e}; retrying in 5s"
            time.sleep(5)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    threading.Thread(target=simulate_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
