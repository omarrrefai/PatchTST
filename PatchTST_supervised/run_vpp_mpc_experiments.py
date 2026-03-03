#!/usr/bin/env python3
"""
run_vpp_mpc_experiments.py

Rolling MPC (MILP) experiments coupled with PatchTST forecasts.

You have:
- SHORT (t+1) PatchTST run folder: pred.npy contains one-step-ahead predictions (shape often N x C x 1)
- LONG (DA288) PatchTST run folder: pred.npy/true.npy contain multi-step forecasts/truth (shape often N x 288 x C)

This script:
1) Loads LONG pred/true, loads SHORT pred only (truth comes from LONG)
2) Auto-selects the target channel (price) by matching SHORT predictions to LONG truth at horizon step 0
3) Runs rolling MPC with controller variants:
   - oracle      : uses realized future prices (upper bound)
   - persistence : repeats current realized price across horizon
   - short       : first step from SHORT, remainder persistence
   - long        : full horizon from LONG
   - fused       : first step from SHORT, remainder from LONG
   - terminal    : like long, plus terminal SOC penalty (simple convex term)
4) Scores realized costs using realized price at each step (scientifically correct)
5) Saves CSV + consumable plots under /home/omaralrefai/dev/VPP_Forecast_Results

Dependencies:
  pip install pulp pandas numpy matplotlib
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pulp


# -----------------------------
# Fixed Paths (as provided)
# -----------------------------
SHORT_DIR = Path(
    "/home/omaralrefai/dev/PatchTST/PatchTST_supervised/results/"
    "PTST_CAN_features_t+1_new-york_PatchTST_custom_ftM_sl576_ll72_pl1_dm512_nh8_el3_dl2_df2048_fc1_ebfixed_dtTrue_test_0/"
)

LONG_DIR = Path(
    "/home/omaralrefai/dev/PatchTST/PatchTST_supervised/results/"
    "PTST_CAN_features_DA288_new-york_PatchTST_custom_ftM_sl576_ll72_pl288_dm512_nh8_el3_dl2_df2048_fc1_ebfixed_dtTrue_test_0"
)

OUT_ROOT = Path("/home/omaralrefai/dev/VPP_Forecast_Results")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Parameters / Config
# -----------------------------
AREA_NAME = "new-york"  # for naming outputs
DEFAULT_BASE_LOAD_MW = 50.0  # constant load profile (replace with real L_t if available)


@dataclass
class BatteryParams:
    E_min: float
    E_max: float
    P_ch_max: float
    P_dis_max: float
    eta_ch: float
    eta_dis: float
    C_deg: float  # $/MWh throughput


@dataclass
class GridParams:
    tau_import: float = 0.0        # $/MWh fee added to import price
    P_imp_max: float = 1e9         # MW
    P_exp_max: float = 0.0         # MW (0 disables export)
    sell_price_factor: float = 1.0 # sell price = factor * price


@dataclass
class MPCParams:
    dt_hours: float = 1 / 12        # 5 minutes
    horizon_steps: int = 72         # 6 hours; can set 288 for full day ahead
    terminal_window_steps: int = 6  # optional 30-min finish window
    solver_time_limit_s: Optional[int] = None


# -----------------------------
# Loading helpers
# -----------------------------
def _require_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(str(p))


def parse_pl_from_folder(folder: Path) -> Optional[int]:
    """
    Parse pred_len from folder name segment like '_pl288_'.
    Works for your LONG folder. For safety only.
    """
    m = re.search(r"_pl(\d+)_", folder.name)
    if not m:
        return None
    return int(m.group(1))


def to_nhc(a: np.ndarray) -> np.ndarray:
    """
    Normalize array to (N, H, C).
    Supports:
      - (N, H, C)
      - (N, C, H) -> transpose
      - (N, H)    -> add channel dim
      - (N,)      -> (N,1,1)
    """
    if a.ndim == 3:
        # assume (N,H,C) if H > C (common: 288 > 8)
        if a.shape[1] > a.shape[2]:
            return a
        # else treat as (N,C,H)
        return np.transpose(a, (0, 2, 1))

    if a.ndim == 2:
        return a[:, :, None]

    if a.ndim == 1:
        return a[:, None, None]

    raise ValueError(f"Unsupported shape for to_nhc: {a.shape}")


def load_long_run(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads LONG day-ahead run:
      pred.npy and true.npy are expected and must match after normalization.

    Returns:
      pred_nhc, true_nhc as float64 with shape (N, H, C)
    """
    _require_exists(run_dir)
    pred_path = run_dir / "pred.npy"
    true_path = run_dir / "true.npy"
    _require_exists(pred_path)
    _require_exists(true_path)

    pred = to_nhc(np.load(pred_path)).astype(np.float64)
    true = to_nhc(np.load(true_path)).astype(np.float64)

    if pred.shape != true.shape:
        raise ValueError(f"LONG pred/true mismatch: pred={pred.shape}, true={true.shape}")

    return pred, true


def load_short_pred_nc(run_dir: Path) -> np.ndarray:
    """
    Loads SHORT t+1 run pred.npy and returns (N, C) float64 predictions.
    Handles common shapes:
      - (N, C, 1) -> (N, C)
      - (N, 1, C) -> (N, C)
      - (N, C)    -> (N, C)
      - (N, 1)    -> (N, 1)
      - (N,)      -> (N, 1)
    """
    _require_exists(run_dir)
    pred_path = run_dir / "pred.npy"
    _require_exists(pred_path)

    pred = np.load(pred_path)

    if pred.ndim == 3:
        if pred.shape[2] == 1:
            return pred[:, :, 0].astype(np.float64)
        if pred.shape[1] == 1:
            return pred[:, 0, :].astype(np.float64)
        raise ValueError(f"Unsupported SHORT pred shape: {pred.shape}")

    if pred.ndim == 2:
        return pred.astype(np.float64)

    if pred.ndim == 1:
        return pred[:, None].astype(np.float64)

    raise ValueError(f"Unsupported SHORT pred shape: {pred.shape}")


def pick_target_channel(short_pred_nc: np.ndarray, long_true_nhc: np.ndarray, sample_n: int = 4000) -> int:
    """
    Auto-pick the channel that corresponds to the price series by matching SHORT(t+1) predictions
    against LONG truth at horizon index 0.

    short_pred_nc: (N_s, C_s)
    long_true_nhc : (N_l, H, C_l)

    Returns:
      best channel index in [0, min(C_s, C_l)-1]
    """
    N = min(short_pred_nc.shape[0], long_true_nhc.shape[0])
    C = min(short_pred_nc.shape[1], long_true_nhc.shape[2])
    n = min(sample_n, N)

    y = long_true_nhc[:n, 0, :C]    # (n, C)
    p = short_pred_nc[:n, :C]       # (n, C)

    maes = np.mean(np.abs(y - p), axis=0)
    return int(np.argmin(maes))


# -----------------------------
# MPC + MILP
# -----------------------------
def make_constant_load(N: int, base_mw: float) -> np.ndarray:
    return np.full(N, base_mw, dtype=np.float64)


def build_forecast_vector(
    k: int,
    H: int,
    variant: str,
    true_1step: np.ndarray,          # (N,)
    pred_short_1step: np.ndarray,    # (N,)
    pred_long_h: np.ndarray,         # (N, H_long)
    sell_price_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a length-H import forecast vector and sell vector used inside MILP.

    'true_1step' is realized series used for oracle/persistence and scoring.
    """
    lam = np.zeros(H, dtype=np.float64)

    if variant == "oracle":
        lam[:] = true_1step[k:k + H]
    elif variant == "persistence":
        lam[:] = true_1step[k]
    elif variant == "short":
        lam[0] = pred_short_1step[k]
        lam[1:] = true_1step[k]
    elif variant == "long":
        lam[:] = pred_long_h[k, :H]
    elif variant == "fused":
        lam[0] = pred_short_1step[k]
        lam[1:] = pred_long_h[k, 1:H]
    elif variant == "terminal":
        lam[:] = pred_long_h[k, :H]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    sell = sell_price_factor * lam
    return lam, sell


def solve_milp_horizon(
    lam_import: np.ndarray,
    lam_sell: np.ndarray,
    load: np.ndarray,
    E0: float,
    batt: BatteryParams,
    grid: GridParams,
    mpc: MPCParams,
    use_terminal_window: bool = True,
    add_terminal_penalty: bool = False,
    terminal_penalty_weight: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Solve single-zone MILP horizon (length H).
    Returns arrays for horizon actions; caller applies only first step.
    """
    H = len(lam_import)
    dt = mpc.dt_hours

    prob = pulp.LpProblem("VPP_MPC_MILP", pulp.LpMinimize)

    # Decision vars
    P_imp = pulp.LpVariable.dicts("P_imp", range(H), lowBound=0)
    P_exp = pulp.LpVariable.dicts("P_exp", range(H), lowBound=0, upBound=grid.P_exp_max)
    P_ch  = pulp.LpVariable.dicts("P_ch",  range(H), lowBound=0, upBound=batt.P_ch_max)
    P_dis = pulp.LpVariable.dicts("P_dis", range(H), lowBound=0, upBound=batt.P_dis_max)
    E     = pulp.LpVariable.dicts("E",     range(H), lowBound=batt.E_min, upBound=batt.E_max)

    z_ch  = pulp.LpVariable.dicts("z_ch",  range(H), cat="Binary")
    z_dis = pulp.LpVariable.dicts("z_dis", range(H), cat="Binary")

    # Objective
    obj_terms = []
    for t in range(H):
        energy = dt * ((lam_import[t] + grid.tau_import) * P_imp[t] - lam_sell[t] * P_exp[t])
        deg    = dt * batt.C_deg * (P_ch[t] + P_dis[t])
        obj_terms.append(energy + deg)

    # Optional terminal penalty (simple absolute deviation from mid SOC)
    if add_terminal_penalty and terminal_penalty_weight > 0.0:
        target = 0.5 * (batt.E_min + batt.E_max)
        dev_pos = pulp.LpVariable("dev_pos", lowBound=0)
        dev_neg = pulp.LpVariable("dev_neg", lowBound=0)
        prob += (E[H - 1] - target) == dev_pos - dev_neg
        obj_terms.append(terminal_penalty_weight * (dev_pos + dev_neg))

    prob += pulp.lpSum(obj_terms)

    # Constraints
    for t in range(H):
        # Power balance
        prob += P_imp[t] + P_dis[t] - P_ch[t] == load[t] + P_exp[t]

        # SOC dynamics
        if t == 0:
            prob += E[t] == E0 + dt * batt.eta_ch * P_ch[t] - dt * (1.0 / batt.eta_dis) * P_dis[t]
        else:
            prob += E[t] == E[t - 1] + dt * batt.eta_ch * P_ch[t] - dt * (1.0 / batt.eta_dis) * P_dis[t]

        # Import limit
        prob += P_imp[t] <= grid.P_imp_max

        # Mode exclusivity
        prob += z_ch[t] + z_dis[t] <= 1
        prob += P_ch[t]  <= batt.P_ch_max  * z_ch[t]
        prob += P_dis[t] <= batt.P_dis_max * z_dis[t]

    # Terminal window band (your Eq. terminal)
    if use_terminal_window:
        N30 = mpc.terminal_window_steps
        low = max(batt.E_min, E0 - N30 * dt * (batt.P_dis_max / batt.eta_dis))
        high = min(batt.E_max, E0 + N30 * dt * (batt.eta_ch * batt.P_ch_max))
        prob += E[H - 1] >= low
        prob += E[H - 1] <= high

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=mpc.solver_time_limit_s) \
        if mpc.solver_time_limit_s else pulp.PULP_CBC_CMD(msg=False)

    status = prob.solve(solver)
    st = pulp.LpStatus[status]
    if st not in ("Optimal", "Feasible"):
        raise RuntimeError(f"MILP failed: {st}")

    def v(d):
        return np.array([pulp.value(d[t]) for t in range(H)], dtype=np.float64)

    return {
        "P_imp": v(P_imp),
        "P_exp": v(P_exp),
        "P_ch":  v(P_ch),
        "P_dis": v(P_dis),
        "E":     v(E),
    }


def run_rolling_mpc(
    variant: str,
    true_prices_1step: np.ndarray,     # realized (N,)
    pred_short_1step: np.ndarray,      # (N,)
    pred_long_full: np.ndarray,        # (N, H_long)
    load_1step: np.ndarray,            # (N,)
    batt: BatteryParams,
    grid: GridParams,
    mpc: MPCParams,
    E0: float,
    add_terminal_penalty: bool = False,
    terminal_penalty_weight: float = 0.0,
) -> pd.DataFrame:
    """
    Rolling MPC simulation.
    Important: realized cost is computed using realized price at time k (true_prices_1step[k]).
    """
    N = len(true_prices_1step)
    H = mpc.horizon_steps
    dt = mpc.dt_hours

    if pred_long_full.shape[0] != N or len(pred_short_1step) != N or len(load_1step) != N:
        raise ValueError("Series length mismatch in run_rolling_mpc input.")

    if pred_long_full.shape[1] < H:
        raise ValueError(f"LONG pred horizon too short: pred_len={pred_long_full.shape[1]} < H={H}")

    max_k = N - H
    if max_k <= 1:
        raise ValueError(f"Need N > H. Got N={N}, H={H}")

    rows: List[dict] = []
    E = float(E0)

    for k in range(max_k):
        lam_f, lam_sell = build_forecast_vector(
            k=k,
            H=H,
            variant=variant,
            true_1step=true_prices_1step,
            pred_short_1step=pred_short_1step,
            pred_long_h=pred_long_full,
            sell_price_factor=grid.sell_price_factor,
        )

        load_h = load_1step[k:k + H]

        t0 = time.time()
        sol = solve_milp_horizon(
            lam_import=lam_f,
            lam_sell=lam_sell,
            load=load_h,
            E0=E,
            batt=batt,
            grid=grid,
            mpc=mpc,
            use_terminal_window=True,
            add_terminal_penalty=add_terminal_penalty,
            terminal_penalty_weight=terminal_penalty_weight,
        )
        solve_s = time.time() - t0

        # Apply first action
        P_imp0 = float(sol["P_imp"][0])
        P_exp0 = float(sol["P_exp"][0])
        P_ch0  = float(sol["P_ch"][0])
        P_dis0 = float(sol["P_dis"][0])

        # Update SOC
        E = E + dt * batt.eta_ch * P_ch0 - dt * (1.0 / batt.eta_dis) * P_dis0

        # Realized cost uses TRUE price at time k
        lam_true = float(true_prices_1step[k])
        sell_true = grid.sell_price_factor * lam_true

        energy_cost = dt * ((lam_true + grid.tau_import) * P_imp0 - sell_true * P_exp0)
        deg_cost    = dt * batt.C_deg * (P_ch0 + P_dis0)
        total_cost  = energy_cost + deg_cost

        rows.append({
            "k": k,
            "price_true": lam_true,
            "price_forecast_0": float(lam_f[0]),
            "P_imp": P_imp0,
            "P_exp": P_exp0,
            "P_ch": P_ch0,
            "P_dis": P_dis0,
            "E": E,
            "energy_cost": energy_cost,
            "deg_cost": deg_cost,
            "total_cost": total_cost,
            "solve_s": solve_s,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Plotting / Outputs
# -----------------------------
def plot_timeseries(df: pd.DataFrame, out_dir: Path, title_prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # SOC
    plt.figure(figsize=(12, 3.2))
    plt.plot(df["k"], df["E"])
    plt.title(f"{title_prefix}  SOC")
    plt.xlabel("Step (5-min)")
    plt.ylabel("E (MWh)")
    plt.tight_layout()
    plt.savefig(out_dir / "soc.png", dpi=220)
    plt.close()

    # Actions
    plt.figure(figsize=(12, 3.2))
    plt.plot(df["k"], df["P_imp"], label="P_imp")
    plt.plot(df["k"], df["P_ch"], label="P_ch")
    plt.plot(df["k"], df["P_dis"], label="P_dis")
    if df["P_exp"].abs().max() > 1e-9:
        plt.plot(df["k"], df["P_exp"], label="P_exp")
    plt.title(f"{title_prefix}  Actions")
    plt.xlabel("Step (5-min)")
    plt.ylabel("Power (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "actions.png", dpi=220)
    plt.close()

    # Cumulative cost
    plt.figure(figsize=(12, 3.2))
    plt.plot(df["k"], df["total_cost"].cumsum())
    plt.title(f"{title_prefix}  Cumulative Realized Cost")
    plt.xlabel("Step (5-min)")
    plt.ylabel("Cost ($)")
    plt.tight_layout()
    plt.savefig(out_dir / "cumulative_cost.png", dpi=220)
    plt.close()

    # Solver time
    plt.figure(figsize=(12, 3.2))
    plt.plot(df["k"], df["solve_s"])
    plt.title(f"{title_prefix}  MILP Solve Time per Step")
    plt.xlabel("Step (5-min)")
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.savefig(out_dir / "solve_time.png", dpi=220)
    plt.close()


def plot_summary_bar(summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 3.4))
    plt.bar(summary["variant"], summary["total_realized_cost_$"])
    plt.title("Total Realized Cost by Controller Variant")
    plt.xlabel("Variant")
    plt.ylabel("Cost ($)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # Load LONG
    pred_long_nhc, true_long_nhc = load_long_run(LONG_DIR)

    # Load SHORT pred only
    short_pred_nc = load_short_pred_nc(SHORT_DIR)

    # Choose target channel (price)
    chosen = pick_target_channel(short_pred_nc, true_long_nhc)
    print(f"[INFO] Chosen target channel = {chosen}")

    # Extract series for MPC
    # Truth used for scoring
    true_prices_1step = true_long_nhc[:, 0, chosen]      # (N_long,)
    # Short t+1 forecast
    pred_short_1step  = short_pred_nc[:, chosen]         # (N_short,)
    # Long horizon forecast
    pred_long_full    = pred_long_nhc[:, :, chosen]      # (N_long, H_long)

    # Align to common length
    N = min(len(true_prices_1step), len(pred_short_1step), pred_long_full.shape[0])
    true_prices_1step = true_prices_1step[:N]
    pred_short_1step  = pred_short_1step[:N]
    pred_long_full    = pred_long_full[:N, :]

    # Load profile
    load_1step = make_constant_load(N, DEFAULT_BASE_LOAD_MW)

    # Params (reasonable defaults; tune for your paper)
    batt = BatteryParams(
        E_min=5.0,
        E_max=25.0,
        P_ch_max=10.0,
        P_dis_max=10.0,
        eta_ch=0.95,
        eta_dis=0.95,
        C_deg=5.0,
    )
    grid = GridParams(
        tau_import=0.0,
        P_imp_max=1e6,
        P_exp_max=0.0,
        sell_price_factor=1.0,
    )
    mpc = MPCParams(
        dt_hours=1/12,
        horizon_steps=72,          # try 288 for DA MPC
        terminal_window_steps=6,
        solver_time_limit_s=None,
    )
    E0 = 0.5 * (batt.E_min + batt.E_max)

    # Variants (IEEE compare/contrast)
    variants = [
        ("oracle",      False, 0.0),
        ("persistence", False, 0.0),
        ("short",       False, 0.0),
        ("long",        False, 0.0),
        ("fused",       False, 0.0),
        ("terminal",    True,  50.0),  # terminal penalty weight (tune)
    ]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = OUT_ROOT / f"{AREA_NAME}_MPC_{stamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []

    for name, use_term, w in variants:
        print(f"[RUN] Variant: {name}")

        df = run_rolling_mpc(
            variant=name,
            true_prices_1step=true_prices_1step,
            pred_short_1step=pred_short_1step,
            pred_long_full=pred_long_full,
            load_1step=load_1step,
            batt=batt,
            grid=grid,
            mpc=mpc,
            E0=E0,
            add_terminal_penalty=use_term,
            terminal_penalty_weight=w,
        )

        out_dir = exp_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(out_dir / "trajectory.csv", index=False)
        plot_timeseries(df, out_dir, title_prefix=name.upper())

        total_cost = float(df["total_cost"].sum())
        avg_solve = float(df["solve_s"].mean())
        p95_solve = float(df["solve_s"].quantile(0.95))

        summary_rows.append({
            "variant": name,
            "total_realized_cost_$": total_cost,
            "avg_solve_s": avg_solve,
            "p95_solve_s": p95_solve,
        })

    summary = pd.DataFrame(summary_rows).sort_values("total_realized_cost_$")
    summary.to_csv(exp_dir / "summary.csv", index=False)
    plot_summary_bar(summary, exp_dir / "total_cost_comparison.png")

    print("\n[OK] Saved experiment outputs to:", exp_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()