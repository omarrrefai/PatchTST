#!/usr/bin/env python3
# canada_plotly_massive_study.py
# Interactive Plotly study for ALL areas:
# - Rebuild & denormalize predictions
# - Compare vs persistence (t-1)
# - Per-target interactive dashboards (HTML)
# - Global summary dashboard (HTML)
# No CLI args: all paths + targets are set below.

from pathlib import Path
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===================== USER CONFIG =====================
MASTER_CSV = Path("/home/omaralrefai/dev/PatchTST/.dataset/canada/canada_realtime_ENGY_2010_2025.csv")
DATA_ROOT  = MASTER_CSV.parent / "per_target_timeonly_clean"   # *_timeonly_clean.csv
RESULTS_ROOT = Path("results")

TARGETS = [
    "Manitoba", "Manitoba SK", "Michigan", "Minnesota", "New-York", "Ontario",
    "Quebec AT", "Quebec B5D.B31L", "Quebec D4Z", "Quebec D5A", "Quebec H4Z",
    "Quebec H9A", "Quebec P33C", "Quebec Q4C", "Quebec X2Y",
]

# Must match training
SEQ_LEN, LABEL_LEN, PRED_LEN = 576, 72, 1
SPLIT = (0.7, 0.1, 0.2)
CAL_SLICE = 500

# Output dirs
OUT_DIR   = RESULTS_ROOT / "plotly_study"
FIG_DIR   = OUT_DIR / "per_target"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
# =======================================================

# ----------------- Helpers -----------------
def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mape(a, b, eps=1.0): return float(np.mean(np.abs(a - b) / np.maximum(eps, np.abs(a))) * 100.0)

def latest_results_dir_for(target: str) -> Path:
    prefix = f"PTST_CAN_timeonly_t+1_{target.replace(' ', '_')}_PatchTST_custom_ftM_sl{SEQ_LEN}_ll{LABEL_LEN}_pl{PRED_LEN}_"
    cands = sorted([p for p in RESULTS_ROOT.glob(f"{prefix}*") if (p / "pred.npy").exists()],
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if cands: return cands[0]
    fb = sorted([p for p in RESULTS_ROOT.rglob("*") if (p / "pred.npy").exists() and target.replace(' ', '_') in p.name],
                key=lambda p: p.stat().st_mtime, reverse=True)
    if fb: return fb[0]
    raise FileNotFoundError(f"No results with pred.npy for {target}")

def load_series(per_target_csv: Path, target: str) -> pd.Series:
    df = pd.read_csv(per_target_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True).set_index("date")
    return df[target].astype(float)

def to_time_channels(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None: return None
    s = np.squeeze(arr)
    if s.ndim == 0: return s.reshape(-1, 1)
    if s.ndim == 1: return s.reshape(-1, 1)
    if s.ndim == 2: return s if s.shape[0] >= s.shape[1] else s.T
    if s.ndim == 3:
        t_ax = int(np.argmax(s.shape))
        s2 = np.moveaxis(s, t_ax, 0)  # time first
        return s2.reshape(s2.shape[0], -1)
    raise ValueError(f"Unsupported pred.npy ndim={s.ndim}")

def choose_col_by_corr(pred_WC: np.ndarray, y_true_raw: np.ndarray):
    W, C = pred_WC.shape
    L = min(W, y_true_raw.size)
    ywin = y_true_raw[:L]
    best_j, best_corr = 0, -np.inf
    for j in range(C):
        pj = pred_WC[:L, j].astype(float)
        if np.std(pj) < 1e-12 or np.std(ywin) < 1e-12:
            corr = -np.inf
        else:
            corr = float(np.corrcoef(pj, ywin)[0,1])
        if np.isfinite(corr) and corr > best_corr:
            best_corr, best_j = corr, j
    return pred_WC[:L, best_j].astype(float), best_j, best_corr

def affine_from_true(model_true: np.ndarray, actual_true: np.ndarray):
    L = min(model_true.size, actual_true.size)
    x = model_true[:L].reshape(-1, 1)
    y = actual_true[:L].reshape(-1, 1)
    X = np.hstack([x, np.ones_like(x)])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        a = float(beta[0, 0]); b = float(beta[1, 0])
        if not np.isfinite(a) or not np.isfinite(b): a, b = 1.0, 0.0
    except Exception:
        a, b = 1.0, 0.0
    return a, b

def pick_inverse(y_true_cal, y_pred_model_cal, train_mean, train_std, model_true_cal=None):
    best = ("identity", 1.0, 0.0, mae(y_true_cal, y_pred_model_cal))
    if np.isfinite(train_std) and train_std > 1e-12:
        y_std = y_pred_model_cal * train_std + train_mean
        m = mae(y_true_cal, y_std)
        if m < best[3]: best = ("std_inv", train_std, train_mean, m)
    if model_true_cal is not None and model_true_cal.size >= 10:
        a,b = affine_from_true(model_true_cal, y_true_cal)
        y_aff = y_pred_model_cal * a + b
        m = mae(y_true_cal, y_aff)
        if m < best[3]: best = ("affine", a, b, m)
    return best[0], best[1], best[2]

def build_persistence(y: pd.Series, idx: np.ndarray):
    out = np.empty(len(idx), dtype=float)
    for k, t in enumerate(idx):
        out[k] = float(y.iloc[t-1]) if t-1 >= 0 else float(y.iloc[0])
    return out

def compute_fluctuation_metrics(series: np.ndarray):
    diff = np.diff(series)
    madiff = np.abs(diff)
    mean_abs_change = float(madiff.mean())
    p95_abs_change  = float(np.percentile(madiff, 95))
    med_abs_change  = float(np.median(madiff))
    large_thr = 2.0 * med_abs_change if med_abs_change > 0 else 0.0
    frac_large = float((madiff > large_thr).mean()) if large_thr > 0 else 0.0
    # simple discrete peaks above threshold
    x = series
    thr = med_abs_change
    peaks = 0
    for i in range(1, len(x)-1):
        if (x[i] - x[i-1] >= thr) and (x[i] - x[i+1] >= thr):
            peaks += 1
    p2p = float(np.max(x) - np.min(x)) if len(x) else 0.0
    return {"mean_abs_change": mean_abs_change, "p95_abs_change": p95_abs_change,
            "frac_large_moves": frac_large, "peak_count": int(peaks), "peak_to_peak_range": p2p}

# ----------------- Plotly figure builders -----------------
def fig_timeseries(df_stream: pd.DataFrame, target: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_stream.ts, y=df_stream.y_true, name="Actual", mode="lines"))
    fig.add_trace(go.Scatter(x=df_stream.ts, y=df_stream.y_pred, name="PatchTST", mode="lines"))
    fig.add_trace(go.Scatter(x=df_stream.ts, y=df_stream.persist, name="Persistence (t-1)", mode="lines"))
    fig.update_layout(title=f"{target}  Actual vs PatchTST vs Persistence",
                      xaxis_title="Time", yaxis_title="ENGY ($/MWh)",
                      hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=True)
    return fig

def fig_daily_mae(daily: pd.DataFrame, target: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.date, y=daily.mae_model, name="PatchTST MAE", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=daily.date, y=daily.mae_persist, name="Persistence MAE", mode="lines+markers"))
    fig.update_layout(title=f"{target}  Daily MAE", xaxis_title="Date", yaxis_title="MAE ($/MWh)",
                      hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=True)
    return fig

def fig_improvement(daily: pd.DataFrame, target: str) -> go.Figure:
    # improvement = persistence - model (positive means model better)
    imp = daily["mae_persist"] - daily["mae_model"]
    colors = np.where(imp >= 0, "#2ca02c", "#d62728")
    fig = go.Figure(go.Bar(x=daily.date, y=imp, marker_color=colors))
    fig.update_layout(title=f"{target}  Daily MAE Improvement (Persist  Model)",
                      xaxis_title="Date", yaxis_title="�MAE ($/MWh)",
                      hovermode="x")
    fig.update_xaxes(rangeslider_visible=True)
    return fig

def fig_mae_rate_of_change(daily: pd.DataFrame, target: str) -> go.Figure:
    roc_model = daily["mae_model"].diff()
    roc_pers  = daily["mae_persist"].diff()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.date, y=roc_model, name="� MAE (PatchTST)", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=daily.date, y=roc_pers,  name="� MAE (Persistence)", mode="lines+markers"))
    fig.update_layout(title=f"{target}  Rate of Change of Daily MAE (day-over-day)",
                      xaxis_title="Date", yaxis_title="�MAE ($/MWh)",
                      hovermode="x unified")
    return fig

def fig_residual_hist(resid: np.ndarray, target: str) -> go.Figure:
    fig = px.histogram(x=resid, nbins=60, labels={"x": "Residual ($/MWh)"})
    fig.update_layout(title=f"{target}  Residuals (y_true  y_pred)", bargap=0.05)
    return fig

def fig_volatility(df_stream: pd.DataFrame, target: str, win=12) -> go.Figure:
    df = df_stream.copy()
    df["ret"] = df["y_true"].diff()
    df["vol_1h"] = df["ret"].rolling(win, min_periods=win//2).std()
    fig = go.Figure(go.Scatter(x=df.ts, y=df.vol_1h, mode="lines", name="Rolling 1h std"))
    fig.update_layout(title=f"{target}  Rolling 1h Std of Returns (volatility)",
                      xaxis_title="Time", yaxis_title="Std ($/MWh)", hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=True)
    return fig

def write_target_dashboard(target: str,
                           df_stream: pd.DataFrame,
                           daily: pd.DataFrame,
                           residuals: np.ndarray,
                           out_dir: Path):
    """Compose a multi-section HTML per target."""
    figs = []
    figs.append(fig_timeseries(df_stream, target))
    figs.append(fig_daily_mae(daily, target))
    figs.append(fig_improvement(daily, target))
    figs.append(fig_mae_rate_of_change(daily, target))
    figs.append(fig_residual_hist(residuals, target))
    figs.append(fig_volatility(df_stream, target))

    # Stack all figures one below another into a single HTML
    html_blocks = []
    for i, f in enumerate(figs, 1):
        html_blocks.append(f.to_html(full_html=False, include_plotlyjs=(i==1)))
    full_html = "<br>".join(html_blocks)
    out_path = out_dir / f"{target.replace(' ', '_')}_dashboard.html"
    out_path.write_text(full_html, encoding="utf-8")
    return out_path

# ----------------- Main study per target -----------------
def process_target(target: str):
    # Load actuals
    per_csv = DATA_ROOT / f"{target.replace(' ', '_')}_timeonly_clean.csv"
    if not per_csv.exists():
        raise FileNotFoundError(f"Missing per-target CSV: {per_csv}")
    y = load_series(per_csv, target)
    N = len(y)
    num_train = int(N * SPLIT[0]); num_val = int(N * SPLIT[1])
    train_mean = float(y.iloc[:num_train].mean())
    train_std  = float(y.iloc[:num_train].std(ddof=0)) if num_train > 1 else float(y.std(ddof=0))
    if not np.isfinite(train_std) or train_std < 1e-12: train_std = 1.0

    # Load predictions
    resdir = latest_results_dir_for(target)
    pred = np.load(resdir / "pred.npy")
    true_path = resdir / "true.npy"
    model_true = np.load(true_path) if true_path.exists() else None

    pred_WC = to_time_channels(pred)
    true_WC = to_time_channels(model_true) if model_true is not None else None
    W = pred_WC.shape[0]

    # Align to test window
    test_start = num_train + num_val
    if test_start + W + PRED_LEN > N:
        test_start = max(SEQ_LEN + LABEL_LEN, N - (W + PRED_LEN))
    idx = np.arange(test_start, min(test_start + W, N))
    ts  = pd.to_datetime(y.index[idx])
    y_true_full = y.iloc[idx].to_numpy(dtype=float)

    # Choose best pred column by correlation vs raw truth
    y_pred_model_full, picked_j, picked_corr = choose_col_by_corr(pred_WC[:len(idx), :], y_true_full)
    model_true_col_full = None
    if true_WC is not None and picked_j < true_WC.shape[1]:
        model_true_col_full = true_WC[:len(idx), picked_j].astype(float)

    # Inverse mapping on calibration slice
    Lcal = min(CAL_SLICE, y_true_full.size)
    inv_name, a_inv, b_inv = pick_inverse(
        y_true_full[:Lcal], y_pred_model_full[:Lcal], train_mean, train_std,
        model_true_col_full[:Lcal] if model_true_col_full is not None else None
    )

    # Denormalize predictions; build persistence
    y_pred_full = y_pred_model_full * a_inv + b_inv
    persist_full = build_persistence(y, idx)

    # Common stream dataframe
    L = min(len(y_true_full), len(y_pred_full), len(persist_full))
    df_stream = pd.DataFrame({
        "ts": ts[:L],
        "y_true": y_true_full[:L],
        "y_pred": y_pred_full[:L],
        "persist": persist_full[:L],
    })

    # Residuals & overall metrics
    residuals = df_stream["y_true"].values - df_stream["y_pred"].values
    m_mae  = mae(df_stream.y_true, df_stream.y_pred)
    p_mae  = mae(df_stream.y_true, df_stream.persist)
    m_rmse = rmse(df_stream.y_true, df_stream.y_pred)
    m_mape = mape(df_stream.y_true, df_stream.y_pred, 1.0)
    skill  = 1.0 - (m_mae / max(p_mae, 1e-12))
    corr   = float(np.corrcoef(df_stream.y_true, df_stream.y_pred)[0,1]) if df_stream.y_true.std()>1e-12 else 0.0

    # Daily MAE
    df_stream["date"] = pd.to_datetime(df_stream["ts"]).dt.normalize()
    g = df_stream.groupby("date", sort=True)
    daily = pd.DataFrame({
        "date": g["ts"].min().values,
        "mae_model": g.apply(lambda d: mae(d["y_true"].values, d["y_pred"].values)).values,
        "mae_persist": g.apply(lambda d: mae(d["y_true"].values, d["persist"].values)).values,
    }).sort_values("date")

    # Fluctuation metrics on actuals
    fluc = compute_fluctuation_metrics(df_stream["y_true"].values)

    # Save data CSVs
    df_stream.to_csv(FIG_DIR / f"{target.replace(' ', '_')}_stream.csv", index=False)
    daily.to_csv(FIG_DIR / f"{target.replace(' ', '_')}_daily_mae.csv", index=False)

    # Build and write per-target dashboard
    html_path = write_target_dashboard(target, df_stream, daily, residuals, FIG_DIR)

    # Summary row
    return {
        "target": target,
        "results_dir": str(resdir),
        "picked_pred_column": int(picked_j),
        "corr_pickstage": float(picked_corr),
        "inverse_mapping": inv_name, "inv_a": float(a_inv), "inv_b": float(b_inv),
        "overall_mae_model": m_mae, "overall_mae_persist": p_mae,
        "skill_vs_persistence": skill, "rmse_model": m_rmse, "mape_model_pct": m_mape,
        "fluc_mean_abs_change": fluc["mean_abs_change"],
        "fluc_p95_abs_change": fluc["p95_abs_change"],
        "fluc_frac_large_moves": fluc["frac_large_moves"],
        "fluc_peak_count": fluc["peak_count"],
        "fluc_peak_to_peak_range": fluc["peak_to_peak_range"],
        "dashboard_html": str(html_path),
    }

# ----------------- Global summary dashboard -----------------
def make_global_skill_bar(df: pd.DataFrame) -> go.Figure:
    df2 = df.sort_values("skill_vs_persistence", ascending=False)
    fig = go.Figure(go.Bar(x=df2["target"], y=df2["skill_vs_persistence"],
                           text=[f"{v:.3f}" for v in df2["skill_vs_persistence"]],
                           textposition="auto"))
    fig.update_layout(title="Skill vs Persistence (1  MAE_model/MAE_persist)",
                      xaxis_title="Target", yaxis_title="Skill",
                      xaxis_tickangle=-45)
    return fig

def make_global_table(df: pd.DataFrame) -> go.Figure:
    cols = ["target","overall_mae_model","overall_mae_persist","skill_vs_persistence",
            "rmse_model","mape_model_pct","picked_pred_column","corr_pickstage",
            "inverse_mapping","inv_a","inv_b",
            "fluc_mean_abs_change","fluc_p95_abs_change","fluc_frac_large_moves",
            "fluc_peak_count","fluc_peak_to_peak_range"]
    show = [c for c in cols if c in df.columns]
    header = dict(values=[c.replace("_"," ").title() for c in show], font=dict(size=12), fill_color="#f0f0f0")
    cells  = dict(values=[df[c] for c in show], font=dict(size=11), height=24)
    fig = go.Figure(data=[go.Table(header=header, cells=cells)])
    fig.update_layout(title="Per-Target Metrics (click links in index below to open dashboards)")
    return fig

def write_global_dashboard(df: pd.DataFrame, out_html: Path):
    # Compose bar + table + index of links
    figs_html = []
    figs_html.append(make_global_skill_bar(df).to_html(full_html=False, include_plotlyjs=True))
    figs_html.append("<br>")
    figs_html.append(make_global_table(df).to_html(full_html=False, include_plotlyjs=False))
    figs_html.append("<hr><h3>Per-target dashboards</h3><ul>")
    for _, row in df.iterrows():
        name = row["target"].replace(" ", "_")
        rel = Path(row["dashboard_html"]).name
        figs_html.append(f'<li><a href="per_target/{rel}" target="_blank">{row["target"]}</a></li>')
    figs_html.append("</ul>")
    out_html.write_text("".join(figs_html), encoding="utf-8")

# ----------------- Main -----------------
def main():
    rows = []
    for tgt in TARGETS:
        try:
            row = process_target(tgt)
            rows.append(row)
            print(f"[OK] {tgt}: skill={row['skill_vs_persistence']:.3f} | "
                  f"MAE={row['overall_mae_model']:.3f} vs Pers={row['overall_mae_persist']:.3f} | "
                  f"dashboard={row['dashboard_html']}")
        except Exception as e:
            print(f"[WARN] Failed for {tgt}: {e}")

    if not rows:
        print("No targets processed."); return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "summary_per_target.csv", index=False)

    # Copy per-target HTMLs already live under FIG_DIR; build a global index
    global_html = OUT_DIR / "index.html"
    write_global_dashboard(df, global_html)

    print("\n=== Plotly study complete ===")
    print("Global summary CSV:", OUT_DIR / "summary_per_target.csv")
    print("Global dashboard  :", global_html)
    print("Per-target pages  :", FIG_DIR)

if __name__ == "__main__":
    main()
