#!/usr/bin/env python3
"""Plot pos/vel/att errors with 3-sigma bounds when no MATLAB GT column dump
was generated (e.g. `--handover none` runs).

Reads the per-IMU-tick CSV stack produced by ifac_wc_2026
(`<prefix>time.csv`, `pos.csv`, `vel.csv`, `att.csv`, `pos_std.csv`,
`vel_std.csv`, `att_std.csv`) and the per-tick truth CSV
(`df_flat_truth.csv` with columns `t,p_n,p_e,p_d,v_n,v_e,v_d,roll,pitch,yaw`),
interpolates truth onto the estimator time, and plots a 3x3 grid of
error lines with +/-3 sigma envelopes.

Usage:
    python plot_3sigma_from_truth.py \\
        --prefix _build/programs/ \\
        --truth  ~/ws/ntnu/parnav/parnav-scripts/post_processing/df_flat_truth.csv

Attitude errors are smallest-signed-angle of (est - truth), plotted in
degrees along with their 3-sigma envelope (also converted to degrees).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ssa(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def load_stack(prefix: Path):
    """Load ifac_wc_2026 per-tick CSV stack into a single DataFrame."""
    def _xyz(name):
        df = pd.read_csv(prefix.with_name(prefix.name + f"{name}.csv"))
        return df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()

    t = pd.read_csv(prefix.with_name(prefix.name + "time.csv"))["t"].to_numpy()
    px, py, pz = _xyz("pos")
    vx, vy, vz = _xyz("vel")
    r, p, y = _xyz("att")
    spx, spy, spz = _xyz("pos_std")
    svx, svy, svz = _xyz("vel_std")
    sr, sp, sy = _xyz("att_std")
    return pd.DataFrame({
        "t": t,
        "p_n": px, "p_e": py, "p_d": pz,
        "v_n": vx, "v_e": vy, "v_d": vz,
        "roll": r, "pitch": p, "yaw": y,
        "s_p_n": spx, "s_p_e": spy, "s_p_d": spz,
        "s_v_n": svx, "s_v_e": svy, "s_v_d": svz,
        "s_roll": sr, "s_pitch": sp, "s_yaw": sy,
    })


def interp_truth(truth: pd.DataFrame, t_est: np.ndarray) -> pd.DataFrame:
    """Linear-interp every non-time column of truth onto t_est."""
    out = {"t": t_est}
    t_tru = truth["t"].to_numpy()
    for col in truth.columns:
        if col == "t":
            continue
        # Unwrap yaw before interpolating to dodge +/-pi wraparounds.
        v = truth[col].to_numpy()
        if col == "yaw":
            v = np.unwrap(v)
        out[col] = np.interp(t_est, t_tru, v)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=Path,
                    default=Path("_build/programs/"),
                    help="Directory+prefix for ifac_wc_2026 outputs. "
                         "Trailing slash or '_prefix' as used in --output-prefix.")
    ap.add_argument("--truth", type=Path, required=True,
                    help="Path to df_flat_truth.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG (defaults to <prefix>3sigma_truth.png).")
    ap.add_argument("--title", default="Error + 3-sigma vs truth")
    args = ap.parse_args()

    # Normalise prefix: allow either a directory (trailing /) or a dir+prefix.
    pre_str = str(args.prefix)
    if pre_str.endswith("/") or args.prefix.is_dir():
        prefix = args.prefix / ""  # keeps directory, name part becomes empty
    else:
        prefix = args.prefix

    est = load_stack(prefix)
    tru_raw = pd.read_csv(args.truth)
    required = {"t", "p_n", "p_e", "p_d", "v_n", "v_e", "v_d",
                "roll", "pitch", "yaw"}
    missing = required - set(tru_raw.columns)
    if missing:
        raise SystemExit(f"truth CSV missing columns: {missing}")

    tru = interp_truth(tru_raw, est["t"].to_numpy())

    # Errors.
    err = pd.DataFrame({"t": est["t"]})
    for c in ("p_n", "p_e", "p_d", "v_n", "v_e", "v_d"):
        err[c] = est[c].to_numpy() - tru[c].to_numpy()
    for c in ("roll", "pitch", "yaw"):
        err[c] = ssa(est[c].to_numpy() - tru[c].to_numpy())

    # Plot grid: rows = pos/vel/att, cols = N/E/D (or R/P/Y).
    rows = [
        ("Position", ["p_n", "p_e", "p_d"],
         ["s_p_n", "s_p_e", "s_p_d"], ["N [m]", "E [m]", "D [m]"], False),
        ("Velocity", ["v_n", "v_e", "v_d"],
         ["s_v_n", "s_v_e", "s_v_d"], ["vN [m/s]", "vE [m/s]", "vD [m/s]"], False),
        ("Attitude", ["roll", "pitch", "yaw"],
         ["s_roll", "s_pitch", "s_yaw"],
         ["Roll [deg]", "Pitch [deg]", "Yaw [deg]"], True),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
    t = est["t"].to_numpy()
    for r, (row_name, cols, sig_cols, labels, to_deg) in enumerate(rows):
        for c, (col, sig_col, lab) in enumerate(zip(cols, sig_cols, labels)):
            e = err[col].to_numpy()
            s = est[sig_col].to_numpy()  # 1-sigma from the binary
            s3 = 3.0 * s
            if to_deg:
                e = np.degrees(e)
                s3 = np.degrees(s3)
            ax = axes[r, c]
            ax.plot(t, e, color="tab:blue", lw=0.8, label="error")
            ax.plot(t,  s3, color="tab:red", lw=0.8, ls="--",
                    label=r"$\pm 3\sigma$")
            ax.plot(t, -s3, color="tab:red", lw=0.8, ls="--")
            ax.fill_between(t, -s3, s3, color="tab:red", alpha=0.08)
            ax.set_ylabel(lab, fontsize=9)
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 2:
                ax.legend(fontsize=8, loc="upper right")
        # Row title on left axis only.
        axes[r, 0].set_title(row_name, fontsize=10, loc="left")

    for ax in axes[-1, :]:
        ax.set_xlabel("t [s]")

    fig.suptitle(args.title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or (prefix.parent / f"{prefix.name}3sigma_truth.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()

    # RMSE summary.
    print("\nRMSE:")
    for row_name, cols, _, labels, to_deg in rows:
        for col, lab in zip(cols, labels):
            e = err[col].to_numpy()
            if to_deg:
                e = np.degrees(e)
            rmse = float(np.sqrt(np.mean(e**2)))
            print(f"  {row_name:8s} {lab:14s}: {rmse:8.4f}")


if __name__ == "__main__":
    main()
