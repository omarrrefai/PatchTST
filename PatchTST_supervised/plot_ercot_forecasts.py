#!/usr/bin/env python3
# ercot_yearly_report.py
# Build per-timestamp predictions, per-year overlay plots, and metrics from:
#   - pred.npy (from PatchTST test/results)
#   - ercot_15min.csv (actuals)
#
# Outputs:
#   out_dir/predictions_long.csv  (date, region, actual, predicted)
#   out_dir/metrics_by_year.csv   (MAE, RMSE, MSE, MAPE by region & year + overall)
#   out_dir/plots/<REGION>.html   (Plotly, dropdown to choose year)

from pathlib import Path
import os, sys, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ----------------------------
# CONFIG — EDIT THESE PATHS
# ----------------------------
PRED_PATH = Path("/home/omaralrefai/dev/PatchTST/PatchTST_supervised/results/"
                 "ERCOT_PatchTST_96_M_all_PatchTST_ETTm1_ftM_sl1344_ll336_pl96_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_ercot_m_0/pred.npy")

CSV_PATH  = Path("/home/omaralrefai/dev/PatchTST/.dataset/ercot/ercot_15min.csv")

OUT_DIR   = Path("/home/omaralrefai/dev/PatchTST/PatchTST_supervised/ercot_report")
# Optional: how many channels to use when aligning the first window (speed/robustness)
ALIGN_USE_FIRST_K_CHANNELS = 6
# ----------------------------

def to_ntc(arr: np.ndarray) -> np.ndarray:
    """Ensure shape [N, T, C]. If [N, C, T], transpose to [N, T, C]."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array [N,T,C] or [N,C,T], got {arr.shape}")
    N, A, B = arr.shape
    if A <= 128 and B > A:  # likely [N, C, T]
        return np.transpose(arr, (0, 2, 1))
    return arr

def zscore(x: np.ndarray, axis=None, eps=1e-8):
    mu = np.nanmean(x, axis=axis, keepdims=True)
    std = np.nanstd(x, axis=axis, keepdims=True)
    return (x - mu) / (std + eps)

def best_alignment_offset(pred_first_sample: np.ndarray, series_2d: np.ndarray, use_cols: list) -> int:
    """
    pred_first_sample: [T, C]
    series_2d:         [L, C]
    use_cols:          indices of columns to compare
    Return s that minimizes SSE( z(series[s:s+T, use_cols]) - z(pred_first_sample[:, use_cols]) )
    """
    T, C = pred_first_sample.shape
    L, C2 = series_2d.shape
    if C2 != C:
        raise ValueError(f"Channel mismatch: pred C={C}, CSV C={C2}")
    max_start = L - T
    if max_start <= 0:
        raise ValueError("CSV series too short for the prediction horizon length.")

    p = zscore(pred_first_sample[:, use_cols], axis=0)
    best_s, best_score = 0, math.inf

    # Exhaustive search (single run, OK); stride=1 for exact alignment
    for s in range(max_start + 1):
        window = series_2d[s:s+T, :][:, use_cols]
        w = zscore(window, axis=0)
        diff = (w - p)
        score = np.nansum(diff * diff)
        if score < best_score:
            best_score = score
            best_s = s
    return best_s

def first_assignment_stitch(pred: np.ndarray, L: int, start_index: int) -> np.ndarray:
    """
    Stitch predictions into a single timeline using FIRST-AVAILABLE assignment.
    pred: [N, T, C], L: total csv length, start_index: alignment offset s
    Returns [L, C] array filled with NaN except where predicted.
    """
    N, T, C = pred.shape
    out = np.full((L, C), np.nan, dtype=float)
    for i in range(N):
        base = start_index + i
        if base >= L: break
        horizon_end = min(T, L - base)
        # Only assign where out is NaN (first-available)
        block = out[base:base+horizon_end]
        mask = np.isnan(block)
        # pred[i, :horizon_end, :] -> [horizon_end, C]
        block[mask] = pred[i, :horizon_end, :][mask]
        out[base:base+horizon_end] = block
    return out

def metrics(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-8):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):  # no overlap
        return dict(n=0, MAE=np.nan, MSE=np.nan, RMSE=np.nan, MAPE=np.nan)
    yt = y_true[mask].astype(float)
    yp = y_pred[mask].astype(float)
    mae = float(np.mean(np.abs(yp - yt)))
    mse = float(np.mean((yp - yt) ** 2))
    rmse = float(np.sqrt(mse))
    # Avoid div-by-zero for MAPE
    denom = np.maximum(np.abs(yt), eps)
    mape = float(np.mean(np.abs((yp - yt) / denom)) * 100.0)
    return dict(n=int(mask.sum()), MAE=mae, MSE=mse, RMSE=rmse, MAPE=mape)

def plot_region_years_html(region: str, dates: pd.Series, actual: pd.Series, predicted: pd.Series, out_html: Path):
    """One region -> dropdown to switch year; overlay actual vs predicted."""
    df = pd.DataFrame({"date": pd.to_datetime(dates), "actual": actual, "pred": predicted})
    df["year"] = df["date"].dt.year
    years = sorted(df["year"].dropna().unique().tolist())

    fig = go.Figure()
    # Add traces per year, hidden initially
    for y in years:
        d = df[df["year"] == y]
        fig.add_trace(go.Scatter(x=d["date"], y=d["actual"], mode="lines",
                                 name=f"{y} • Actual", line=dict(width=2), visible=False))
        fig.add_trace(go.Scatter(x=d["date"], y=d["pred"], mode="lines",
                                 name=f"{y} • Predicted", line=dict(dash="dash", width=2), visible=False))

    # Show first year by default
    if len(fig.data) >= 2:
        fig.data[0].visible = True
        fig.data[1].visible = True

    buttons = []
    for idx, y in enumerate(years):
        vis = [False] * (2 * len(years))
        vis[2*idx] = True
        vis[2*idx + 1] = True
        buttons.append(dict(
            label=str(y),
            method="restyle",
            args=[{"visible": vis}, [list(range(2 * len(years)))]]
        ))

    fig.update_layout(
        title=f"{region} — ERCOT SPP: Actual vs Predicted (per year)",
        xaxis_title="Time",
        yaxis_title="SPP ($/MWh)",
        updatemenus=[dict(type="dropdown", x=1.02, y=1.0, xanchor="left", yanchor="top",
                          showactive=True, direction="down", buttons=buttons)],
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=60, r=160, t=60, b=100),
        template="plotly_white",
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")

def main():
    # --- Load predictions
    if not PRED_PATH.exists():
        sys.exit(f"[error] pred file not found: {PRED_PATH}")
    pred = np.load(PRED_PATH, allow_pickle=True)  # [N,T,C] or [N,C,T]
    pred = to_ntc(pred)
    N, T, C = pred.shape
    print(f"[info] pred shape: N={N}, T={T}, C={C}")

    # --- Load actuals
    if not CSV_PATH.exists():
        sys.exit(f"[error] CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    regions = [c for c in df.columns if c != "date"]
    if len(regions) != C:
        print(f"[warn] channel mismatch: CSV has {len(regions)} columns (ex-date) but pred has C={C}.")
        # fallback: trim or pad
        regions = regions[:C] + [f"series_{i}" for i in range(len(regions), C)]
    actual_full = df[regions].values  # [L, C]
    dates_full  = df["date"].values   # [L]
    L = len(df)

    # --- Find alignment of the first pred sample into the CSV
    use_cols = list(range(min(C, ALIGN_USE_FIRST_K_CHANNELS)))
    s = best_alignment_offset(pred_first_sample=pred[0], series_2d=actual_full, use_cols=use_cols)
    print(f"[info] aligned first sample at CSV index: {s} (date={df['date'].iloc[s]})")

    # Quality check (optional): correlation of first horizon
    # corr = np.corrcoef(zscore(actual_full[s:s+T,:],0).ravel(), zscore(pred[0],0).ravel())[0,1]
    # print(f"[info] z-norm correlation of first sample: {corr:.3f}")

    # --- Stitch predictions into one timeline (first-available)
    pred_series = first_assignment_stitch(pred, L=L, start_index=s)  # [L,C]

    # --- Long-form CSV
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    long_rows = []
    for c_idx, reg in enumerate(regions):
        long_rows.append(pd.DataFrame({
            "date": df["date"],
            "region": reg,
            "actual": actual_full[:, c_idx],
            "predicted": pred_series[:, c_idx],
        }))
    long_df = pd.concat(long_rows, ignore_index=True)
    long_csv = out_dir / "predictions_long.csv"
    long_df.to_csv(long_csv, index=False)
    print(f"[done] wrote {long_csv}")

    # --- Metrics per region per year + overall
    long_df["year"] = pd.to_datetime(long_df["date"]).dt.year
    metrics_rows = []
    for reg, g in long_df.groupby("region"):
        # overall
        m_all = metrics(g["actual"].values, g["predicted"].values)
        m_all.update(dict(region=reg, year="ALL"))
        metrics_rows.append(m_all)
        # by year
        for y, gy in g.groupby("year"):
            m = metrics(gy["actual"].values, gy["predicted"].values)
            m.update(dict(region=reg, year=int(y)))
            metrics_rows.append(m)
    met_df = pd.DataFrame(metrics_rows, columns=["region","year","n","MAE","MSE","RMSE","MAPE"])
    met_csv = out_dir / "metrics_by_year.csv"
    met_df.to_csv(met_csv, index=False)
    print(f"[done] wrote {met_csv}")

    # --- Plot per region with year dropdown
    plots_dir = out_dir / "plots"
    for c_idx, reg in enumerate(regions):
        plot_region_years_html(
            region=reg,
            dates=df["date"],
            actual=pd.Series(actual_full[:, c_idx]),
            predicted=pd.Series(pred_series[:, c_idx]),
            out_html=plots_dir / f"{reg}_forecast.html"
        )
    print(f"[done] wrote HTML plots to {plots_dir}")

if __name__ == "__main__":
    main()
