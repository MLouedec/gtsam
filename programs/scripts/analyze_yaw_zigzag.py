#!/usr/bin/env python3
"""Analyze yaw zigzag in ifac_wc_2026 debug log.

Joins the per-update debug log (from `ifac_wc_2026 --debug-log`) with the
per-IMU-tick truth CSV (`df_flat_truth.csv`) and produces:

  1. yaw error vs time, with rover-tick markers
  2. FFT of yaw error (dominant frequency)
  3. gyro-bias trajectory (drift vs oscillation sanity check)
  4. time-shift search: best lag tau that minimises the Unit3 residual
     between measured baseline and truth-predicted baseline
  5. innovation-vs-yaw-rate scatter (correlation with yaw rate => timing)

Usage:
    python analyze_yaw_zigzag.py \
        [--debug PATH] [--truth PATH] [--out-dir PATH]

Defaults look in ~/ws/ntnu/parnav/parnav-scripts/post_processing/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_DIR = Path.home() / "ws/ntnu/parnav/parnav-scripts/post_processing"


def rpy_to_R(roll, pitch, yaw):
    """NED Z-Y-X (yaw-pitch-roll) rotation matrix, matching GTSAM Rot3::Ypr.

    Broadcasting: roll/pitch/yaw are 1-D arrays of length N -> returns (N,3,3).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    N = len(roll)
    Rz = np.zeros((N, 3, 3))
    Ry = np.zeros((N, 3, 3))
    Rx = np.zeros((N, 3, 3))
    Rz[:, 0, 0] = cy; Rz[:, 0, 1] = -sy; Rz[:, 1, 0] = sy; Rz[:, 1, 1] = cy; Rz[:, 2, 2] = 1
    Ry[:, 0, 0] = cp; Ry[:, 0, 2] = sp; Ry[:, 1, 1] = 1; Ry[:, 2, 0] = -sp; Ry[:, 2, 2] = cp
    Rx[:, 0, 0] = 1; Rx[:, 1, 1] = cr; Rx[:, 1, 2] = -sr; Rx[:, 2, 1] = sr; Rx[:, 2, 2] = cr
    return Rz @ Ry @ Rx  # Z-Y-X


def unit3_error_vec(a, b):
    """Approximate 2D Unit3 errorVector (angle on tangent plane at a).

    For small angles this is ~the angular error vector. Full norm is the
    arc length; we just need a scalar 'angular mismatch', so use arccos of
    dot product for the total angular error.
    """
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.clip((a * b).sum(axis=-1), -1.0, 1.0)
    return np.arccos(dot)  # radians


def ssa(x):
    """Smallest signed angle in [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def load(args):
    dbg = pd.read_csv(args.debug)
    tru = pd.read_csv(args.truth)
    print(f"debug:  {len(dbg)} rows, t=[{dbg.t.iloc[0]:.3f}, {dbg.t.iloc[-1]:.3f}]")
    print(f"truth:  {len(tru)} rows, t=[{tru.t.iloc[0]:.3f}, {tru.t.iloc[-1]:.3f}]")
    # Align via nearest-neighbour on time.
    merged = pd.merge_asof(
        dbg.sort_values("t"),
        tru.sort_values("t"),
        on="t",
        direction="nearest",
        suffixes=("", "_truth"),
    )
    return merged, tru


def yaw_rate_truth(merged):
    """Truth yaw rate at debug-row times (deg/s)."""
    yaw = merged["yaw"].to_numpy()
    t = merged["t"].to_numpy()
    return np.degrees(np.gradient(np.unwrap(yaw), t))


def maneuver_mask(merged, max_yaw_rate_deg_s):
    """True for samples within a low-|yaw_rate| window (between maneuvers)."""
    if max_yaw_rate_deg_s is None or max_yaw_rate_deg_s <= 0:
        return np.ones(len(merged), dtype=bool)
    return np.abs(yaw_rate_truth(merged)) <= max_yaw_rate_deg_s


def plot_yaw_error(merged, out_dir, mask=None):
    rpy = merged[["roll", "pitch", "yaw"]].to_numpy()
    yaw_err = ssa(merged["yaw_post"].to_numpy() - rpy[:, 2])
    rover = merged["rover_tick"].to_numpy().astype(bool)
    t = merged["t"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, np.degrees(yaw_err), lw=0.8, label="yaw_post − yaw_truth")
    ax.scatter(t[rover], np.degrees(yaw_err[rover]), s=4, c="r", label="rover tick")
    if mask is not None and not mask.all():
        ax.scatter(t[~mask], np.degrees(yaw_err[~mask]), s=4, c="grey",
                   label="masked (high-rate)", alpha=0.5)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("yaw error [deg]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "yaw_error.png", dpi=140)
    plt.close(fig)

    rmse_all = np.sqrt(np.mean(yaw_err**2))
    print(f"Yaw RMSE (all):           {np.degrees(rmse_all):.3f} deg  "
          f"({len(yaw_err)} samples)")
    if mask is not None and not mask.all():
        rmse_masked = np.sqrt(np.mean(yaw_err[mask]**2))
        dropped = (~mask).sum()
        print(f"Yaw RMSE (between-mvr):   {np.degrees(rmse_masked):.3f} deg  "
              f"({mask.sum()} samples, {dropped} masked)")
    return yaw_err, t


def plot_yaw_fft(yaw_err, t, out_dir):
    # Assume near-uniform sampling.
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    y = yaw_err - yaw_err.mean()
    Y = np.fft.rfft(y * np.hanning(len(y)))
    f = np.fft.rfftfreq(len(y), dt)
    mag = np.abs(Y) / len(y)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogy(f, mag)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("|FFT(yaw_err)|")
    ax.set_xlim(0, min(fs / 2, 20.0))
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "yaw_error_fft.png", dpi=140)
    plt.close(fig)

    # Top 3 peaks.
    idx = np.argsort(mag)[-5:][::-1]
    print("Top-5 yaw-error FFT peaks [Hz]:", f[idx])


def plot_bias(merged, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for col, ls in zip(["bg_x", "bg_y", "bg_z"], ["-", "-", "-"]):
        axes[0].plot(merged.t, merged[col], lw=0.6, label=col)
    axes[0].set_ylabel("gyro bias [rad/s]")
    axes[0].grid(True); axes[0].legend()
    for col in ["ba_x", "ba_y", "ba_z"]:
        axes[1].plot(merged.t, merged[col], lw=0.6, label=col)
    axes[1].set_ylabel("accel bias [m/s^2]")
    axes[1].set_xlabel("t [s]")
    axes[1].grid(True); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bias.png", dpi=140)
    plt.close(fig)


def time_shift_search(dbg, tru, out_dir, max_ticks=500):
    """Best lag tau (IMU samples) minimising truth-predicted baseline error.

    For each rover-tick row in dbg, look up truth rotation at t + tau*dt
    and predict nZ = R_truth * baseline_body_hat. Compare to the measured
    z_gnss_comp direction via angular error.
    """
    rover_rows = dbg[dbg.rover_tick == 1].copy()
    if len(rover_rows) == 0:
        print("time-shift: no rover ticks in debug log")
        return

    # Median baseline_body (constant across rows; take any).
    bb = rover_rows[["baseline_body_x", "baseline_body_y", "baseline_body_z"]].iloc[0].to_numpy()
    bb_hat = bb / np.linalg.norm(bb)

    # Measured NED baseline (unit vectors) on rover ticks.
    z = rover_rows[["z_gnss_comp_x", "z_gnss_comp_y", "z_gnss_comp_z"]].to_numpy()
    z_hat = z / np.linalg.norm(z, axis=1, keepdims=True)

    # Truth time base.
    tru_t = tru["t"].to_numpy()
    tru_rpy = tru[["roll", "pitch", "yaw"]].to_numpy()
    dt_truth = np.median(np.diff(tru_t))
    fs = 1.0 / dt_truth

    taus_sec = np.arange(-max_ticks, max_ticks + 1) * dt_truth
    mean_err = np.zeros_like(taus_sec)

    t_rover = rover_rows["t"].to_numpy()
    for k, tau in enumerate(taus_sec):
        # Nearest-neighbour lookup at shifted time.
        idx = np.searchsorted(tru_t, t_rover + tau)
        idx = np.clip(idx, 0, len(tru_t) - 1)
        rpy_k = tru_rpy[idx]
        R_k = rpy_to_R(rpy_k[:, 0], rpy_k[:, 1], rpy_k[:, 2])
        n_pred = R_k @ bb_hat
        err = unit3_error_vec(n_pred, z_hat)
        mean_err[k] = np.mean(err)

    tau_star = taus_sec[np.argmin(mean_err)]
    tau_ticks = int(round(tau_star * fs))
    print(f"time-shift search: τ* = {tau_star*1000:.2f} ms "
          f"(~{tau_ticks} truth ticks); "
          f"min mean err = {np.degrees(mean_err.min()):.3f} deg "
          f"vs τ=0: {np.degrees(mean_err[len(mean_err)//2]):.3f} deg")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(taus_sec * 1000, np.degrees(mean_err))
    ax.axvline(0, color="gray", lw=0.5)
    ax.axvline(tau_star * 1000, color="r", lw=0.8, label=f"τ*={tau_star*1000:.1f} ms")
    ax.set_xlabel("time shift τ [ms]")
    ax.set_ylabel("mean angular residual [deg]")
    ax.grid(True); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "time_shift_search.png", dpi=140)
    plt.close(fig)


def plot_innovation_vs_yaw_rate(merged, out_dir):
    rover = merged[merged.rover_tick == 1].copy()
    if len(rover) < 3:
        return
    t = rover["t"].to_numpy()
    yaw_truth = rover["yaw"].to_numpy()
    # Unwrap for differentiation.
    yaw_unwrap = np.unwrap(yaw_truth)
    yaw_rate = np.gradient(yaw_unwrap, t)
    inn = rover["att_innov_norm"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(np.degrees(yaw_rate), np.degrees(inn), s=3, alpha=0.4)
    ax.set_xlabel("yaw rate (truth) [deg/s]")
    ax.set_ylabel("|attitude innovation| [deg]")
    ax.grid(True)
    corr = np.corrcoef(yaw_rate, inn)[0, 1]
    ax.set_title(f"corr(yaw_rate, |innov|) = {corr:+.3f}")
    fig.tight_layout()
    fig.savefig(out_dir / "innovation_vs_yaw_rate.png", dpi=140)
    plt.close(fig)
    print(f"corr(yaw_rate, |innov|) = {corr:+.3f}   "
          "(|>0.3| => likely timing/latency bug)")


def plot_structural_diagnostics(merged, out_dir, mask=None):
    """Separate structural error sources from filter tuning.

    (1) Gyro scale-factor error:
        yaw_err vs cumulative |truth_yaw_rate|*dt. A linear slope means a
        fraction of every degree of yaw motion is mistracked -- constant
        gyro bias can't absorb it.
    (2) Axis misalignment:
        corr(yaw_err, truth_roll_rate) and corr(yaw_err, truth_pitch_rate).
        Nonzero => gyro X/Y axes bleed into Z.
    (3) Compass baseline geometry:
        mean yaw_err over low-rate samples (static offset) and stddev of
        |baseline_body|. A non-zero mean means a yaw-offset lever-arm
        miscalibration; a large relative stddev means lever-arm length
        itself is wrong.
    """
    if mask is None:
        mask = np.ones(len(merged), dtype=bool)
    m = merged.loc[mask].reset_index(drop=True)
    if len(m) < 10:
        print("structural diagnostics: not enough masked samples, skipping")
        return

    t = m["t"].to_numpy()
    rpy = m[["roll", "pitch", "yaw"]].to_numpy()
    roll_rate = np.gradient(np.unwrap(rpy[:, 0]), t)
    pitch_rate = np.gradient(np.unwrap(rpy[:, 1]), t)
    yaw_rate = np.gradient(np.unwrap(rpy[:, 2]), t)
    yaw_err = ssa(m["yaw_post"].to_numpy() - rpy[:, 2])

    # (1) Scale factor: yaw_err vs cumulative |yaw-rate|*dt (total yaw path).
    dt = np.gradient(t)
    cum_yaw = np.cumsum(np.abs(yaw_rate) * dt)
    # Robust slope: linear fit yaw_err = k * cum_yaw + c.
    A = np.vstack([cum_yaw, np.ones_like(cum_yaw)]).T
    k, c = np.linalg.lstsq(A, yaw_err, rcond=None)[0]
    sf_ppm = k * 1e6  # rad per rad of path = ppm of scale error
    print(f"structural (1) gyro scale factor: slope = {sf_ppm:.0f} ppm "
          f"(yaw_err per rad of cumulative |yaw-rate|)")

    # (2) Axis misalignment: correlations.
    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        s = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / s) if s > 0 else 0.0
    c_rr = corr(roll_rate, yaw_err)
    c_pr = corr(pitch_rate, yaw_err)
    # Least-squares decomposition: yaw_err ~ a*roll_rate + b*pitch_rate + c.
    X = np.vstack([roll_rate, pitch_rate, np.ones_like(yaw_err)]).T
    coefs, *_ = np.linalg.lstsq(X, yaw_err, rcond=None)
    print(f"structural (2) axis misalignment: "
          f"corr(roll_rate, yaw_err)={c_rr:+.3f}  "
          f"corr(pitch_rate, yaw_err)={c_pr:+.3f}")
    print(f"                                 "
          f"yaw_err ~ {coefs[0]:+.4f}*roll_rate {coefs[1]:+.4f}*pitch_rate "
          f"+ {coefs[2]:+.4e}  (coef [rad per rad/s])")

    # (3) Compass baseline geometry.
    bb = merged[["baseline_body_x", "baseline_body_y", "baseline_body_z"]].to_numpy()
    bb_norm = np.linalg.norm(bb, axis=1)
    bb_norm = bb_norm[bb_norm > 1e-9]
    mean_bb = bb_norm.mean() if len(bb_norm) else 0.0
    std_bb = bb_norm.std() if len(bb_norm) else 0.0
    # Static subset: very low yaw-rate.
    static = np.abs(yaw_rate) < np.deg2rad(5.0)
    if static.sum() > 5:
        offset = np.degrees(yaw_err[static].mean())
        print(f"structural (3) compass baseline: |bb|={mean_bb:.4f} m "
              f"(std {std_bb:.2e}, {std_bb/mean_bb*100 if mean_bb>0 else 0:.2f}%)  "
              f"static yaw offset = {offset:+.3f} deg "
              f"({static.sum()} samples, |yaw_rate|<5 deg/s)")
    else:
        print(f"structural (3) compass baseline: |bb|={mean_bb:.4f} m "
              f"(std {std_bb:.2e}); not enough low-rate samples for offset")

    # Plots.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.scatter(cum_yaw, np.degrees(yaw_err), s=3, alpha=0.4)
    xs = np.array([cum_yaw.min(), cum_yaw.max()])
    ax.plot(xs, np.degrees(k * xs + c), "r-", lw=1,
            label=f"slope = {sf_ppm:.0f} ppm")
    ax.set_xlabel("cumulative |yaw-rate|·dt [rad]")
    ax.set_ylabel("yaw error [deg]")
    ax.set_title("(1) Gyro scale factor")
    ax.grid(True); ax.legend()

    ax = axes[0, 1]
    ax.scatter(np.degrees(roll_rate), np.degrees(yaw_err), s=3, alpha=0.4)
    ax.set_xlabel("truth roll rate [deg/s]")
    ax.set_ylabel("yaw error [deg]")
    ax.set_title(f"(2a) Roll→Yaw  corr={c_rr:+.3f}")
    ax.grid(True)

    ax = axes[1, 0]
    ax.scatter(np.degrees(pitch_rate), np.degrees(yaw_err), s=3, alpha=0.4)
    ax.set_xlabel("truth pitch rate [deg/s]")
    ax.set_ylabel("yaw error [deg]")
    ax.set_title(f"(2b) Pitch→Yaw  corr={c_pr:+.3f}")
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(merged.t, bb_norm if len(bb_norm) == len(merged)
            else np.linalg.norm(bb, axis=1), lw=0.6)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("|baseline_body| [m]")
    ax.set_title("(3) Baseline length sanity")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_dir / "structural_diagnostics.png", dpi=140)
    plt.close(fig)


def plot_early_window(merged, out_dir, window_s=10.0):
    """Zoom on the first `window_s` seconds of the debug log.

    Overlays propagated and smoothed yaw plus truth yaw, with rover ticks
    marked. Useful for diagnosing early-convergence problems introduced
    by, e.g., compass-lag re-timing that drops the first rover samples.
    """
    t0 = merged["t"].iloc[0]
    mask = merged["t"] <= t0 + window_s
    m = merged[mask]
    if len(m) < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(m.t, np.degrees(np.unwrap(m.yaw)),      label="yaw_truth", lw=1.2)
    ax.plot(m.t, np.degrees(np.unwrap(m.yaw_prop)), label="yaw_prop (propagated)", lw=0.8)
    ax.plot(m.t, np.degrees(np.unwrap(m.yaw_post)), label="yaw_post (smoothed)", lw=0.8)
    rover = m.rover_tick.to_numpy().astype(bool)
    if rover.any():
        ax.scatter(m.t[rover], np.degrees(m.yaw_post[rover]),
                   s=8, c="r", label="rover tick", zorder=3)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("yaw [deg]")
    ax.set_title(f"Early-window zoom (first {window_s:.0f} s)")
    ax.grid(True); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "yaw_early_window.png", dpi=140)
    plt.close(fig)


def lag_loss_report(data_csv, lag_ticks):
    """Count rover ticks lost to a given --compass-lag-ticks value.

    Reads the raw df_flat_data.csv (which has gnss_rover_meas_idx). With
    shifted[i] = original[i - lag], samples whose i-lag is out of [0,N-1]
    are dropped. For positive lag: the first `lag` samples of the shifted
    column are zero; any rover firings in original[i] with i in [N-lag,N)
    fall off the end.
    """
    if not data_csv.exists():
        print(f"lag-loss: data CSV not found at {data_csv}, skipping")
        return
    df = pd.read_csv(data_csv, usecols=["gnss_rover_meas_idx"])
    rover = (df["gnss_rover_meas_idx"].to_numpy() != 0)
    N = len(rover)
    total = int(rover.sum())
    if lag_ticks > 0:
        # original[i] with i >= N - lag falls off.
        lost = int(rover[N - lag_ticks:].sum()) if lag_ticks < N else total
        head_gap = lag_ticks
        print(f"lag-loss: lag=+{lag_ticks} ticks => {lost}/{total} rover ticks "
              f"lost from tail; first {head_gap} ticks of shifted column are 0")
    elif lag_ticks < 0:
        lag = -lag_ticks
        lost = int(rover[:lag].sum()) if lag < N else total
        print(f"lag-loss: lag={lag_ticks} ticks => {lost}/{total} rover ticks "
              f"lost from head; last {lag} ticks of shifted column are 0")
    else:
        print("lag-loss: lag=0, no samples dropped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", type=Path,
                    default=DEFAULT_DIR / "ifac_wc_2026_se3_cb_none_none_debug.csv")
    ap.add_argument("--truth", type=Path,
                    default=DEFAULT_DIR / "df_flat_truth.csv")
    ap.add_argument("--data", type=Path,
                    default=DEFAULT_DIR / "df_flat_data.csv",
                    help="raw CSV — used for the lag-loss report")
    ap.add_argument("--out-dir", type=Path,
                    default=DEFAULT_DIR / "yaw_zigzag_analysis")
    ap.add_argument("--lag-ticks", type=int, default=0,
                    help="lag value applied during the run being analysed")
    ap.add_argument("--early-window-s", type=float, default=10.0)
    ap.add_argument("--yaw-rate-mask-deg-s", type=float, default=100.0,
                    help="mask samples where |yaw_rate_truth| exceeds this "
                         "(deg/s) when reporting RMSE / FFT / τ*. 0 disables.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged, tru = load(args)

    mask = maneuver_mask(merged, args.yaw_rate_mask_deg_s)
    if not mask.all():
        print(f"Maneuver mask: keeping {mask.sum()}/{len(mask)} samples "
              f"(|yaw_rate| <= {args.yaw_rate_mask_deg_s} deg/s)")

    yaw_err, t = plot_yaw_error(merged, args.out_dir, mask)
    # FFT and τ* on the masked subset for between-maneuver characterisation.
    plot_yaw_fft(yaw_err[mask], t[mask], args.out_dir)
    plot_bias(merged, args.out_dir)
    time_shift_search(merged[mask].reset_index(drop=True), tru, args.out_dir)
    plot_innovation_vs_yaw_rate(merged[mask].reset_index(drop=True), args.out_dir)
    plot_structural_diagnostics(merged, args.out_dir, mask)
    plot_early_window(merged, args.out_dir, args.early_window_s)
    lag_loss_report(args.data, args.lag_ticks)

    print(f"\nPlots written to {args.out_dir}")


if __name__ == "__main__":
    main()
