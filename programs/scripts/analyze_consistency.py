#!/usr/bin/env python3
"""Filter-consistency analysis for run_bledar / SimulationFixedLagSmoother CSVs.

Overconfidence shows up as "small sigma, large error". This quantifies it with
the position NEES (Normalized Estimation Error Squared):

    NEES(t) = sum_k (pos_err_k / sigma_pos_k)^2 ,   sigma_pos_k = sig3_pos_k / 3

For a consistent filter the time-average ANEES ~= 3 (position dof), inside the
chi-square confidence band for the effective sample count. ANEES >> 3 means the
covariance is too small (overconfident); << 3 means conservative.

Only POSITION is used: run_bledar has RTK truth for position but writes vel/att/
bias error+sigma as zero, so position is the only consistency anchor.

Usage:
  python analyze_consistency.py <results.csv> [more.csv ...]
      [--decimate-s 1.0]   # spacing between ~independent samples for the band
      [--conf 0.95]        # two-sided confidence level for the band
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

AXES = ("n", "e", "d")


def chi2_quantile(k: float, p: float) -> float:
    """Chi-square quantile via Wilson-Hilferty (no scipy needed)."""
    z = NormalDist().inv_cdf(p)
    return k * (1.0 - 2.0 / (9.0 * k) + z * math.sqrt(2.0 / (9.0 * k))) ** 3


def analyze(path: Path, decimate_s: float, conf: float,
            sigma_floor: float) -> dict:
    df = pd.read_csv(path)
    t = df["t"].to_numpy()
    err = df[[f"pos_err_{a}" for a in AXES]].to_numpy()
    sig = df[[f"sig3_pos_{a}" for a in AXES]].to_numpy() / 3.0

    # Drop rows whose reported sigma is implausibly small (anchor/reset
    # artifacts: a handful of rows with sigma ~ 0 otherwise blow the mean NEES
    # to millions and swamp the true behaviour).
    finite = np.isfinite(err).all(1) & np.isfinite(sig).all(1)
    ok = finite & (sig > sigma_floor).all(1)
    n_dropped = int(finite.sum() - ok.sum())
    err, sig, t = err[ok], sig[ok], t[ok]
    if len(err) == 0:
        raise ValueError(f"{path}: no rows with sigma > {sigma_floor}")

    z2 = (err / sig) ** 2               # per-axis normalized error^2 (~1 each)
    nees = z2.sum(1)                     # per-time NEES (dof = 3)
    anees = float(nees.mean())          # mean (outlier-sensitive)
    nees_median = float(np.median(nees))  # robust; chi2_3 median ~ 2.366
    # 3-sigma coverage per axis: fraction of |err| <= 3 sigma (target 99.73%).
    coverage = (np.abs(err) <= 3.0 * sig).mean(0) * 100.0

    # Effective independent-sample count for the band: decimate by time so the
    # 2 kHz correlation doesn't produce an absurdly tight band.
    if decimate_s > 0 and len(t) > 1:
        keep = [0]
        for i in range(1, len(t)):
            if t[i] - t[keep[-1]] >= decimate_s:
                keep.append(i)
        n_eff = len(keep)
    else:
        n_eff = len(t)
    lo = chi2_quantile(3 * n_eff, (1 - conf) / 2) / n_eff
    hi = chi2_quantile(3 * n_eff, (1 + conf) / 2) / n_eff

    # Robust verdict from 3-sigma coverage + median NEES (the mean is skewed by
    # a heavy tail, e.g. occasional vertical excursions).
    worst_cov = float(coverage.min())
    if worst_cov < 99.0:
        verdict = f"OVERCONFIDENT (worst-axis 3-sigma coverage {worst_cov:.1f}%)"
    elif nees_median < lo:
        verdict = "conservative (sigma too large)"
    else:
        verdict = "consistent"

    rmse_axis = np.sqrt((err ** 2).mean(0))
    rmse_horiz = float(np.sqrt((err[:, :2] ** 2).sum(1).mean()))
    rmse_3d = float(np.sqrt((err ** 2).sum(1).mean()))
    return {
        "path": path, "anees": anees, "nees_median": nees_median,
        "coverage": coverage, "band": (lo, hi), "verdict": verdict,
        "per_axis_z2": z2.mean(0), "n": len(err), "n_eff": n_eff,
        "n_dropped": n_dropped, "median_sigma": np.median(sig, 0),
        "rmse_axis": rmse_axis, "rmse_horiz": rmse_horiz, "rmse_3d": rmse_3d,
        "inflation": math.sqrt(anees / 3.0),
    }


def report(r: dict) -> None:
    print(f"\n=== {r['path'].name} ===")
    print(f"  rows: {r['n']}  (effective independent: {r['n_eff']}; "
          f"dropped {r['n_dropped']} sigma~0 rows)")
    ms = r["median_sigma"]
    print(f"  median position sigma [m]:  N={ms[0]:.3f}  E={ms[1]:.3f}  "
          f"D={ms[2]:.3f}")
    print(f"  position RMSE [m]:  N={r['rmse_axis'][0]:.3f}  "
          f"E={r['rmse_axis'][1]:.3f}  D={r['rmse_axis'][2]:.3f}  "
          f"| horiz={r['rmse_horiz']:.3f}  3D={r['rmse_3d']:.3f}")
    z2 = r["per_axis_z2"]
    print(f"  per-axis norm.err^2 (target ~1):  N={z2[0]:.2f}  "
          f"E={z2[1]:.2f}  D={z2[2]:.2f}")
    cov = r["coverage"]
    print(f"  3-sigma coverage [%] (target 99.7):  N={cov[0]:.1f}  "
          f"E={cov[1]:.1f}  D={cov[2]:.1f}")
    lo, hi = r["band"]
    print(f"  NEES  median={r['nees_median']:.2f} (target ~2.37)   "
          f"mean={r['anees']:.2f} (band [{lo:.2f},{hi:.2f}]; outlier-sensitive)")
    print(f"  VERDICT: {r['verdict']}")
    # One-line machine-readable summary for sweep drivers.
    print(f"  RESULT {r['path'].name} nees_med={r['nees_median']:.4f} "
          f"cov_min={cov.min():.2f} rmse3d={r['rmse_3d']:.4f} "
          f"rmse_horiz={r['rmse_horiz']:.4f} verdict={r['verdict'].split()[0]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, nargs="+")
    ap.add_argument("--decimate-s", type=float, default=1.0,
                    help="Spacing [s] between ~independent samples for the "
                         "confidence band (default 1.0).")
    ap.add_argument("--conf", type=float, default=0.95,
                    help="Two-sided confidence level for the band.")
    ap.add_argument("--sigma-floor", type=float, default=1e-3,
                    help="Drop rows with reported 1-sigma below this [m] "
                         "(anchor/reset artifacts). Default 1e-3.")
    args = ap.parse_args()
    for p in args.csv:
        try:
            report(analyze(p, args.decimate_s, args.conf, args.sigma_floor))
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"\n=== {p} ===\n  SKIP: {e}")


if __name__ == "__main__":
    main()
