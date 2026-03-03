#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def battery_throughput_mwh(df: pd.DataFrame, dt_hours: float) -> float:
    # throughput = integral(P_ch + P_dis) * dt
    return float(((df["P_ch"].abs() + df["P_dis"].abs()) * dt_hours).sum())


def summarize_variant(df: pd.DataFrame, dt_hours: float) -> dict:
    thr = battery_throughput_mwh(df, dt_hours)
    frac_ch = float((df["P_ch"] > 1e-6).mean())
    frac_dis = float((df["P_dis"] > 1e-6).mean())
    soc_min = float(df["E"].min())
    soc_max = float(df["E"].max())
    soc_std = float(df["E"].std())
    avg_price = float(df["price_true"].mean())
    # "alignment": average price during discharge - average price during charge
    discharge_prices = df.loc[df["P_dis"] > 1e-6, "price_true"]
    charge_prices = df.loc[df["P_ch"] > 1e-6, "price_true"]
    align = float(discharge_prices.mean() - charge_prices.mean()) if len(discharge_prices) and len(charge_prices) else np.nan

    return {
        "steps": int(len(df)),
        "total_cost_$": float(df["total_cost"].sum()),
        "energy_cost_$": float(df["energy_cost"].sum()),
        "deg_cost_$": float(df["deg_cost"].sum()),
        "battery_throughput_MWh": thr,
        "charge_frac": frac_ch,
        "discharge_frac": frac_dis,
        "soc_min": soc_min,
        "soc_max": soc_max,
        "soc_std": soc_std,
        "avg_price": avg_price,
        "price_alignment_discharge_minus_charge": align,
        "avg_solve_s": float(df["solve_s"].mean()),
        "p95_solve_s": float(df["solve_s"].quantile(0.95)),
    }


def pick_windows(df_ref: pd.DataFrame, steps_per_day: int = 288, window_days: int = 7):
    """
    Pick three windows:
      - median volatility week
      - highest volatility week
      - peak price week
    """
    w = window_days * steps_per_day
    n = len(df_ref)
    if n < w + 1:
        return [(0, min(n, w))]

    # compute rolling metrics on price_true
    price = df_ref["price_true"].to_numpy()
    # window start indices
    starts = np.arange(0, n - w, w)  # week blocks
    vols = []
    peaks = []
    for s in starts:
        seg = price[s:s+w]
        vols.append(np.std(seg))
        peaks.append(np.max(seg))

    vols = np.array(vols)
    peaks = np.array(peaks)

    med_idx = int(np.argsort(vols)[len(vols)//2])
    vol_idx = int(np.argmax(vols))
    peak_idx = int(np.argmax(peaks))

    def to_range(i):
        s = int(starts[i])
        return (s, s + w)

    return [to_range(med_idx), to_range(vol_idx), to_range(peak_idx)]


def plot_window(df: pd.DataFrame, out_path: Path, title: str, dt_hours: float):
    k = df["k"].to_numpy()
    price = df["price_true"].to_numpy()
    soc = df["E"].to_numpy()
    pch = df["P_ch"].to_numpy()
    pdis = df["P_dis"].to_numpy()

    plt.figure(figsize=(14, 3.2))
    plt.plot(k, price)
    plt.title(title + " — Realized Price")
    plt.xlabel("Step")
    plt.ylabel("$/MWh")
    plt.tight_layout()
    plt.savefig(out_path.with_name(out_path.stem + "_price.png"), dpi=250)
    plt.close()

    plt.figure(figsize=(14, 3.2))
    plt.plot(k, soc)
    plt.title(title + " — SOC")
    plt.xlabel("Step")
    plt.ylabel("MWh")
    plt.tight_layout()
    plt.savefig(out_path.with_name(out_path.stem + "_soc.png"), dpi=250)
    plt.close()

    plt.figure(figsize=(14, 3.2))
    plt.plot(k, pch, label="P_ch")
    plt.plot(k, pdis, label="P_dis")
    plt.title(title + " — Battery Power")
    plt.xlabel("Step")
    plt.ylabel("MW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.with_name(out_path.stem + "_battery_power.png"), dpi=250)
    plt.close()

    # cumulative cost over window
    plt.figure(figsize=(14, 3.2))
    plt.plot(k, df["total_cost"].cumsum().to_numpy())
    plt.title(title + " — Cumulative Cost (window)")
    plt.xlabel("Step")
    plt.ylabel("$")
    plt.tight_layout()
    plt.savefig(out_path.with_name(out_path.stem + "_cumcost.png"), dpi=250)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path, help="Experiment folder containing variant subfolders.")
    ap.add_argument("--dt-hours", type=float, default=1/12, help="Time step in hours (default 5 min).")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    dt = args.dt_hours
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    variant_dirs = [p for p in run_dir.iterdir() if p.is_dir() and (p / "trajectory.csv").exists()]
    if not variant_dirs:
        raise RuntimeError(f"No variant trajectories found in {run_dir}")

    # Summaries
    summaries = []
    data = {}
    for vd in sorted(variant_dirs, key=lambda p: p.name):
        df = pd.read_csv(vd / "trajectory.csv")
        data[vd.name] = df
        row = summarize_variant(df, dt)
        row["variant"] = vd.name
        summaries.append(row)

    summ = pd.DataFrame(summaries).sort_values("total_cost_$")
    summ.to_csv(run_dir / "diagnostics_summary.csv", index=False)
    print("\nSaved:", run_dir / "diagnostics_summary.csv")
    print(summ[["variant", "total_cost_$", "battery_throughput_MWh", "charge_frac", "discharge_frac",
                "soc_min", "soc_max", "price_alignment_discharge_minus_charge"]].to_string(index=False))

    # Readable window plots based on oracle price stream (or any)
    ref_variant = "oracle" if "oracle" in data else list(data.keys())[0]
    windows = pick_windows(data[ref_variant])

    out_plots = run_dir / "readable_windows"
    out_plots.mkdir(exist_ok=True)

    for wi, (s, e) in enumerate(windows, start=1):
        for vname, df in data.items():
            seg = df.iloc[s:e].copy()
            title = f"{vname.upper()} | Window {wi} | steps {s}..{e}"
            plot_window(seg, out_plots / f"{vname}_w{wi}.png", title, dt)

    print("Saved readable window plots to:", out_plots)


if __name__ == "__main__":
    main()