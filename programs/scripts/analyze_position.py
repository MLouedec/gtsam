#!/usr/bin/env python3
"""Position-error analysis for ifac_wc_2026 runs.

Reads the per-IMU-tick state stack (time.csv, pos.csv, vel.csv + _std)
plus the per-tick truth CSV, interps truth onto the estimator time, and
reports:

    * RMSE per axis (N, E, D), horizontal sqrt(N^2+E^2), and 3D.
    * Per-interval breakdown (default edges match the MATLAB pipeline:
      RTK / SEG.1 / SEG.2 / SEG.3).
    * Error line + +/-3 sigma envelope plot (pos only, 3x1 grid).
    * Horizontal and 3D error magnitude plots over time.

Usage:
    python analyze_position.py --prefix <dir+prefix> --truth <truth.csv>
      [--intervals 0,200,300,415,600 --labels RTK,SEG.1,SEG.2,SEG.3]

Compare multiple runs in one call:
    python analyze_position.py --prefix path1_ --prefix path2_ \
        --label "no handover" --label "angle-range" --truth ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_stack(prefix: Path) -> pd.DataFrame:
    """Load ifac_wc_2026 per-tick pos/vel + std CSVs."""
    def _xyz(name):
        df = pd.read_csv(prefix.with_name(prefix.name + f"{name}.csv"))
        return df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()

    t = pd.read_csv(prefix.with_name(prefix.name + "time.csv"))["t"].to_numpy()
    px, py, pz = _xyz("pos")
    spx, spy, spz = _xyz("pos_std")
    return pd.DataFrame({
        "t": t,
        "p_n": px, "p_e": py, "p_d": pz,
        "s_p_n": spx, "s_p_e": spy, "s_p_d": spz,
    })


def interp_truth(truth: pd.DataFrame, t_est: np.ndarray) -> pd.DataFrame:
    out = {"t": t_est}
    t_tru = truth["t"].to_numpy()
    for col in ("p_n", "p_e", "p_d"):
        out[col] = np.interp(t_est, t_tru, truth[col].to_numpy())
    return pd.DataFrame(out)


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def interval_rmse_table(t, err_n, err_e, err_d, edges, labels):
    """Return list of (label, n_samples, rmse_n, rmse_e, rmse_d, rmse_h, rmse_3d)."""
    rows = []
    for i, lab in enumerate(labels):
        lo, hi = edges[i], edges[i + 1]
        mask = (t >= lo) & (t < hi)
        if mask.sum() == 0:
            continue
        rows.append((
            lab, int(mask.sum()),
            rmse(err_n[mask]), rmse(err_e[mask]), rmse(err_d[mask]),
            rmse(np.hypot(err_n[mask], err_e[mask])),
            rmse(np.sqrt(err_n[mask]**2 + err_e[mask]**2 + err_d[mask]**2)),
        ))
    return rows


def print_rmse(label: str, t, err_n, err_e, err_d, edges, labels):
    print(f"\n=== {label} ===")
    print(f"{'Interval':<8}  {'N':<6}  {'RMSE N':>8}  {'RMSE E':>8}  "
          f"{'RMSE D':>8}  {'RMSE H':>8}  {'RMSE 3D':>8}")
    print("-" * 72)
    # Per interval.
    for row in interval_rmse_table(t, err_n, err_e, err_d, edges, labels):
        lab, n, rn, re_, rd, rh, r3 = row
        print(f"{lab:<8}  {n:<6d}  {rn:8.3f}  {re_:8.3f}  "
              f"{rd:8.3f}  {rh:8.3f}  {r3:8.3f}")
    # Overall.
    all_r = (rmse(err_n), rmse(err_e), rmse(err_d),
             rmse(np.hypot(err_n, err_e)),
             rmse(np.sqrt(err_n**2 + err_e**2 + err_d**2)))
    print("-" * 72)
    print(f"{'ALL':<8}  {len(t):<6d}  {all_r[0]:8.3f}  {all_r[1]:8.3f}  "
          f"{all_r[2]:8.3f}  {all_r[3]:8.3f}  {all_r[4]:8.3f}")


def plot_errors(runs, out_path: Path, edges, labels):
    """2-panel plot: per-axis error+3sigma (3 rows) and magnitude (1 row)."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    names = ["N [m]", "E [m]", "D [m]"]
    keys = ["p_n", "p_e", "p_d"]
    skeys = ["s_p_n", "s_p_e", "s_p_d"]

    # Shaded interval backgrounds (draw on top row then replicate via axvspan).
    colours = ["#d6ecd3", "#fad7e0", "#d3e3f7", "#faeec6"]
    for ax in axes:
        for i, lab in enumerate(labels):
            ax.axvspan(edges[i], edges[i + 1],
                       color=colours[i % len(colours)], alpha=0.25, lw=0)

    for r, (label, est, tru, colour) in enumerate(runs):
        t = est["t"].to_numpy()
        for i, (k, sk, nm) in enumerate(zip(keys, skeys, names)):
            err = est[k].to_numpy() - tru[k].to_numpy()
            s3 = 3.0 * est[sk].to_numpy()
            axes[i].plot(t, err, color=colour, lw=0.8, label=label)
            axes[i].plot(t, s3, color=colour, lw=0.6, ls="--", alpha=0.6)
            axes[i].plot(t, -s3, color=colour, lw=0.6, ls="--", alpha=0.6)
            if r == 0:
                axes[i].set_ylabel(nm)
                axes[i].grid(True, alpha=0.3)
        # Magnitude panel.
        err_h = np.hypot(est["p_n"].to_numpy() - tru["p_n"].to_numpy(),
                         est["p_e"].to_numpy() - tru["p_e"].to_numpy())
        axes[3].plot(t, err_h, color=colour, lw=0.8, label=f"{label} (h)")
        err_3d = np.sqrt(err_h**2 +
                         (est["p_d"].to_numpy() - tru["p_d"].to_numpy())**2)
        axes[3].plot(t, err_3d, color=colour, lw=0.8, ls=":",
                     label=f"{label} (3D)")

    axes[3].set_ylabel("|err| [m]")
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlabel("t [s]")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[3].legend(fontsize=7, loc="upper right", ncol=2)

    # Label intervals on top axis.
    for i, lab in enumerate(labels):
        axes[0].text((edges[i] + edges[i+1]) / 2,
                     axes[0].get_ylim()[1] * 0.9, lab,
                     ha="center", fontsize=9, color="grey")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=Path, action="append", required=True,
                    help="Dir+prefix for an ifac_wc_2026 run (repeat for multi-compare).")
    ap.add_argument("--label", action="append", default=None,
                    help="Label for each --prefix. Defaults to prefix name.")
    ap.add_argument("--truth", type=Path, required=True,
                    help="df_flat_truth.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG (default: <first-prefix>_position.png)")
    ap.add_argument("--intervals", default="0,200,300,415,600",
                    help="Comma-separated interval edges in seconds")
    ap.add_argument("--labels", default="RTK,SEG.1,SEG.2,SEG.3",
                    help="Comma-separated interval labels")
    args = ap.parse_args()

    edges = [float(x) for x in args.intervals.split(",")]
    labels = args.labels.split(",")
    if len(labels) != len(edges) - 1:
        raise SystemExit("number of --labels must be len(--intervals) - 1")

    truth = pd.read_csv(args.truth)
    missing = {"t", "p_n", "p_e", "p_d"} - set(truth.columns)
    if missing:
        raise SystemExit(f"truth CSV missing columns: {missing}")

    default_labels = [p.name.rstrip("_") for p in args.prefix]
    run_labels = args.label or default_labels
    if len(run_labels) != len(args.prefix):
        raise SystemExit("--label count must match --prefix count")

    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    runs = []
    for i, (prefix, lab) in enumerate(zip(args.prefix, run_labels)):
        est = load_stack(prefix)
        tru = interp_truth(truth, est["t"].to_numpy())
        err_n = est["p_n"].to_numpy() - tru["p_n"].to_numpy()
        err_e = est["p_e"].to_numpy() - tru["p_e"].to_numpy()
        err_d = est["p_d"].to_numpy() - tru["p_d"].to_numpy()
        print_rmse(lab, est["t"].to_numpy(),
                   err_n, err_e, err_d, edges, labels)
        runs.append((lab, est, tru, palette[i % len(palette)]))

    out = args.out or (args.prefix[0].parent /
                       f"{args.prefix[0].name}position.png")
    plot_errors(runs, out, edges, labels)


if __name__ == "__main__":
    main()
