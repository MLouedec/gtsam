#!/usr/bin/env python3
"""IMU characterisation for the IFAC WC 2026 dataset.

Reads df_flat_data.csv and produces:
  1. Raw gyro/accel spectra on a static window (pre-takeoff) and a cruise
     window. Peaks well above the static floor are mechanical vibration.
  2. Overlapping Allan variance on the static window, per gyro and accel
     axis. Extracts:
        - ARW  (angle random walk, rad/sqrt(s))                -- slope -1/2
        - RRW  (rate random walk, rad/s/sqrt(s))               -- slope +1/2
        - Bias instability (rad/s)                              -- flat min
     These map directly to the sigmas consumed by GTSAM's PreintegrationParams.
  3. Gauss-Markov bias correlation time tau, two ways (cross-checked):
        - Allan-deviation bump:   tau_c = T_peak / 1.89
        - Bias autocorrelation:   exp(-t/tau) fit at lag > 0
     These feed T_acc / T_ars in run_bledar (currently 3600 s, a placeholder).
     A long static window is required: a first-order GM tau of T seconds needs
     a record of at least a few * T. The report prints the max resolvable tau
     for the chosen window and warns when an estimate is near that ceiling.
  4. Optional bias-observability view (if --debug is supplied): overlays
     bg_z(t), truth yaw-rate, and yaw-error on a shared axis.

Static window is auto-detected as the earliest contiguous window with
low gyro magnitude. Override with --static-start / --static-end (seconds).

Usage:
  python characterise_imu.py \
      --data ~/ws/ntnu/parnav/parnav-scripts/post_processing/df_flat_data.csv \
      [--debug <debug.csv>] [--truth <truth.csv>] \
      [--out-dir <dir>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_DIR = Path.home() / "ws/ntnu/parnav/parnav-scripts/post_processing"


# ---------------------------------------------------------------------------
# Static-window auto-detection
# ---------------------------------------------------------------------------

def find_static_window(t, w_m, min_s=20.0, win_s=1.0):
    """Return (t_start, t_end, rms) of the LONGEST contiguous static window.

    Computes |w_m| RMS in short win_s blocks, thresholds at 3x the quietest
    block, and returns the longest contiguous run below it. This adapts to
    however much static data exists -- a fixed-length lowest-RMS search can
    straddle takeoff and silently mix in motion. `min_s` is used only to warn
    when the detected static run is shorter than that. Override with
    --static-start / --static-end.
    """
    dt = np.median(np.diff(t))
    nb = max(1, int(round(win_s / dt)))
    mag = np.linalg.norm(w_m, axis=0)
    nblk = len(mag) // nb
    rms = np.sqrt(np.mean(mag[:nblk * nb].reshape(nblk, nb) ** 2, axis=1))
    below = rms < 3.0 * rms.min()

    best = (0, 0)
    i = 0
    while i < nblk:
        if below[i]:
            j = i
            while j + 1 < nblk and below[j + 1]:
                j += 1
            if (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j + 1
        else:
            i += 1
    i0 = best[0] * nb
    i1 = min((best[1] + 1) * nb - 1, len(mag) - 1)
    seg_rms = float(np.sqrt(np.mean(mag[i0:i1 + 1] ** 2)))
    if (t[i1] - t[i0]) < min_s:
        print(f"  WARNING: longest static run is only {t[i1] - t[i0]:.1f}s "
              f"(< min_s={min_s}); GM tau above ~{(t[i1]-t[i0])/4/1.89:.1f}s "
              f"is unresolvable.")
    return t[i0], t[i1], seg_rms


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------

def one_sided_psd(x, fs):
    """Simple one-sided PSD with Hann window. Returns (f, Pxx)."""
    x = x - x.mean()
    win = np.hanning(len(x))
    X = np.fft.rfft(x * win)
    # Normalisation: PSD in units^2 / Hz.
    scale = 2.0 / (fs * (win ** 2).sum())
    Pxx = (np.abs(X) ** 2) * scale
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return f, Pxx


def plot_spectra(t, w_m, f_m, static_t, cruise_t, out_dir):
    dt = np.median(np.diff(t))
    fs = 1.0 / dt

    def slice_window(arr, window):
        mask = (t >= window[0]) & (t <= window[1])
        return arr[:, mask]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    names_g = ["gyro_x", "gyro_y", "gyro_z"]
    names_a = ["accel_x", "accel_y", "accel_z"]

    for col, label in enumerate(names_g):
        ax = axes[0, col]
        for win_t, lab, ls in [(static_t, "static", "-"),
                               (cruise_t, "cruise", "-")]:
            w = slice_window(w_m, win_t)
            if w.shape[1] < 64:
                continue
            f, P = one_sided_psd(w[col], fs)
            ax.loglog(f, np.sqrt(P), ls, label=lab, lw=0.7)
        ax.set_title(label)
        ax.grid(True, which="both", lw=0.3)
        if col == 0:
            ax.set_ylabel(r"gyro ASD [rad/s/$\sqrt{Hz}$]")
        ax.legend(fontsize=8)

    for col, label in enumerate(names_a):
        ax = axes[1, col]
        for win_t, lab in [(static_t, "static"), (cruise_t, "cruise")]:
            w = slice_window(f_m, win_t)
            if w.shape[1] < 64:
                continue
            f, P = one_sided_psd(w[col], fs)
            ax.loglog(f, np.sqrt(P), label=lab, lw=0.7)
        ax.set_title(label)
        ax.set_xlabel("f [Hz]")
        ax.grid(True, which="both", lw=0.3)
        if col == 0:
            ax.set_ylabel(r"accel ASD [m/s$^2$/$\sqrt{Hz}$]")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "imu_spectra.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Allan variance
# ---------------------------------------------------------------------------

def allan_variance(x, fs, n_tau=60):
    """Overlapping Allan variance for a 1-D signal x sampled at fs Hz.

    Returns (tau, adev) with adev = sigma_y(tau) in the same units as x.
    """
    N = len(x)
    # log-spaced m (samples per cluster)
    m_max = N // 4
    ms = np.unique(np.logspace(0, np.log10(m_max), n_tau).astype(int))
    taus = ms / fs
    adev = np.empty_like(taus, dtype=float)

    # Cumulative theta (integrated signal).
    theta = np.cumsum(x) / fs
    for i, m in enumerate(ms):
        # Overlapping sample mean differences.
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        var = np.mean(d ** 2) / (2 * (m / fs) ** 2)
        adev[i] = np.sqrt(var)
    return taus, adev


def fit_allan_params(tau, adev):
    """Crude ARW / RRW / bias-instability estimates.

    ARW = adev(1s) (assumes slope -1/2 near tau=1s).
    RRW = adev(tau_max) * sqrt(3/tau_max)   (slope +1/2 region).
    BI  = min(adev) * sqrt(2 / (2*ln(2)/pi)) ≈ min(adev) / 0.664.
    """
    # Nearest tau to 1s for ARW.
    i1 = int(np.argmin(np.abs(tau - 1.0)))
    arw = adev[i1] * np.sqrt(tau[i1])  # rad/sqrt(s): adev*sqrt(tau) is flat for WN
    # Actually: for angle random walk N, adev(tau) = N / sqrt(tau).
    # So N = adev * sqrt(tau). Take it at tau=1.
    # RRW: adev(tau) = K * sqrt(tau/3) => K = adev * sqrt(3/tau) at large tau
    i_max = len(tau) - 1
    rrw = adev[i_max] * np.sqrt(3.0 / tau[i_max])
    # Bias instability (flat floor): min(adev)*pi/(2*ln(2)) factor.
    bi = adev.min() / 0.664
    return arw, rrw, bi


# ---------------------------------------------------------------------------
# Gauss-Markov bias correlation time
# ---------------------------------------------------------------------------

def tau_from_allan_peak(tau, adev):
    """First-order Gauss-Markov correlation time from the Allan-deviation bump.

    A 1st-order GM process peaks in the Allan variance at averaging time
    T_peak = 1.89 * tau_c, so tau_c = T_peak / 1.89. The bump sits at tau
    LONGER than the bias-instability floor (the AD minimum); the search starts
    there so the short-tau white-noise roll-off is not mistaken for the peak.

    Returns (tau_c, T_peak, resolved). resolved=False if the peak lands on the
    largest averaging time (AD still rising -> no GM bump inside the window: the
    record is too short, or the bias is closer to a random walk).
    """
    i_floor = int(np.argmin(adev))
    i_peak = i_floor + int(np.argmax(adev[i_floor:]))
    resolved = i_floor < i_peak < len(tau) - 1
    return tau[i_peak] / 1.89, tau[i_peak], resolved


def tau_from_autocorr(x, fs, ac_floor=0.1):
    """GM correlation time from the bias autocorrelation.

    White measurement noise is uncorrelated, so it only inflates lag 0; the
    normalized autocorrelation at lag >= 1 sample reflects the slow bias. Fit
    exp(-t/tau) (log-linear) over the lags where the AC is still well above the
    noise (AC > ac_floor), excluding lag 0.

    Returns tau [s], or nan if no decaying correlation is found.
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    max_lag = max(8, n // 5)
    ac = np.correlate(x, x, mode="full")[n - 1: n - 1 + max_lag]
    if ac[0] <= 0:
        return float("nan")
    ac = ac / ac[0]
    lags = np.arange(len(ac)) / fs
    good = np.arange(1, len(ac))
    good = good[ac[good] > ac_floor]
    if len(good) < 5:
        return float("nan")
    slope, _ = np.polyfit(lags[good], np.log(ac[good]), 1)
    return (-1.0 / slope) if slope < 0 else float("nan")


def plot_allan(w_m_static, f_m_static, fs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    params = {"gyro": [], "accel": []}
    gm = {"gyro": [], "accel": []}  # (axis, tau_peak, resolved, tau_autocorr)
    tau_grid_max = 0.0

    for col, (name, unit) in enumerate([
        ("gyro",  r"$\sigma_y$ [rad/s]"),
        ("accel", r"$\sigma_y$ [m/s$^2$]"),
    ]):
        ax = axes[col]
        sig = w_m_static if name == "gyro" else f_m_static
        for axis_i, axis_n in enumerate(["x", "y", "z"]):
            tau, adev = allan_variance(sig[axis_i], fs)
            tau_grid_max = max(tau_grid_max, tau[-1])
            ax.loglog(tau, adev, label=f"{name}_{axis_n}", lw=0.8)
            arw, rrw, bi = fit_allan_params(tau, adev)
            params[name].append((axis_n, arw, rrw, bi))
            tau_pk, T_pk, resolved = tau_from_allan_peak(tau, adev)
            tau_ac = tau_from_autocorr(sig[axis_i], fs)
            gm[name].append((axis_n, tau_pk, resolved, tau_ac))
            if resolved:
                ax.axvline(T_pk, color="0.6", lw=0.5, ls=":")
        ax.set_xlabel(r"$\tau$ [s]")
        ax.set_ylabel(unit)
        ax.set_title(f"Allan deviation — {name}")
        ax.grid(True, which="both", lw=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "allan_deviation.png", dpi=140)
    plt.close(fig)

    # Console report.
    print("\n=== Allan-variance-derived IMU params ===")
    print(f"{'axis':<8}{'ARW':>14}{'RRW':>16}{'BiasInstab':>18}")
    for name in ("gyro", "accel"):
        for axis_n, arw, rrw, bi in params[name]:
            print(f"{name}_{axis_n:<6}{arw:>14.4e}{rrw:>16.4e}{bi:>18.4e}")
    print("\nMap to GTSAM PreintegrationParams:")
    print("  gyroscopeCovariance      ~ (ARW_gyro)^2   per axis")
    print("  biasOmegaCovariance      ~ (RRW_gyro)^2   per axis")
    print("  accelerometerCovariance  ~ (ARW_accel)^2  per axis")
    print("  biasAccCovariance        ~ (RRW_accel)^2  per axis")

    # Gauss-Markov correlation time.
    tau_ceiling = tau_grid_max / 1.89
    print("\n=== Gauss-Markov bias correlation time tau ===")
    print(f"{'axis':<8}{'tau_AllanPeak[s]':>18}{'resolved':>10}"
          f"{'tau_autocorr[s]':>18}")
    med = {}
    for name in ("gyro", "accel"):
        resolved_taus = []
        for axis_n, tau_pk, resolved, tau_ac in gm[name]:
            flag = "yes" if resolved else "NO(ceil)"
            pk = f"{tau_pk:>18.2f}" if resolved else f"{'>'+f'{tau_pk:.1f}':>18}"
            ac = f"{tau_ac:>18.2f}" if np.isfinite(tau_ac) else f"{'--':>18}"
            print(f"{name}_{axis_n:<6}{pk}{flag:>10}{ac}")
            if resolved:
                resolved_taus.append(tau_pk)
            if np.isfinite(tau_ac):
                resolved_taus.append(tau_ac)
        med[name] = float(np.median(resolved_taus)) if resolved_taus else \
            float("nan")

    print(f"\nMax resolvable tau from this window ~ tau_max/1.89 = "
          f"{tau_ceiling:.1f} s")
    print("Suggested run_bledar time constants (median of finite estimates):")
    print(f"  T_ars (gyro bias)  ~ {med['gyro']:.1f} s")
    print(f"  T_acc (accel bias) ~ {med['accel']:.1f} s")
    for name, label in (("gyro", "T_ars"), ("accel", "T_acc")):
        if np.isfinite(med[name]) and med[name] > 0.5 * tau_ceiling:
            print(f"  WARNING: {label} estimate ({med[name]:.1f}s) is near the "
                  f"{tau_ceiling:.1f}s ceiling -> extend the static window "
                  f"(--static-min-s / --static-end) for a reliable value.")
    print("  (run_bledar currently hardcodes T_acc = T_ars = 3600 s.)")
    return params


# ---------------------------------------------------------------------------
# Bias observability
# ---------------------------------------------------------------------------

def plot_bias_observability(debug_path, truth_path, out_dir):
    if not debug_path.exists():
        return
    dbg = pd.read_csv(debug_path)
    tru = pd.read_csv(truth_path) if truth_path.exists() else None
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(dbg.t, dbg.bg_z, lw=0.7, label="bg_z [rad/s]")
    axes[0].grid(True); axes[0].legend(); axes[0].set_ylabel("gyro-z bias")

    if tru is not None:
        # Nearest-neighbour truth yaw at debug times.
        idx = np.searchsorted(tru.t.to_numpy(), dbg.t.to_numpy())
        idx = np.clip(idx, 0, len(tru) - 1)
        yaw_tru = tru.yaw.to_numpy()[idx]
        yaw_tru_unwrap = np.unwrap(yaw_tru)
        yaw_rate = np.gradient(yaw_tru_unwrap, dbg.t.to_numpy())
        axes[1].plot(dbg.t, np.degrees(yaw_rate), lw=0.7, color="tab:orange")
        axes[1].set_ylabel("truth yaw rate [deg/s]")
        axes[1].grid(True)

        yaw_err = (dbg.yaw_post.to_numpy() - yaw_tru + np.pi) % (2 * np.pi) - np.pi
        axes[2].plot(dbg.t, np.degrees(yaw_err), lw=0.7, color="tab:red")
        axes[2].set_ylabel("yaw error [deg]")
        axes[2].grid(True)

        # Cross-correlation bg_z vs yaw-rate (lag 0).
        bg = dbg.bg_z.to_numpy()
        bg_c = (bg - bg.mean()) / (bg.std() + 1e-12)
        yr_c = (yaw_rate - yaw_rate.mean()) / (yaw_rate.std() + 1e-12)
        corr = float(np.mean(bg_c * yr_c))
        axes[0].set_title(f"corr(bg_z, yaw_rate_truth) = {corr:+.3f}   "
                          "(|>0.3| => vibration rectification / observability issue)")

    axes[-1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(out_dir / "bias_observability.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=DEFAULT_DIR / "df_flat_data.csv")
    ap.add_argument("--debug", type=Path,
                    default=DEFAULT_DIR /
                    "ifac_wc_2026_se3_cb_no_handover_no_robust_debug.csv")
    ap.add_argument("--truth", type=Path,
                    default=DEFAULT_DIR / "df_flat_truth.csv")
    ap.add_argument("--out-dir", type=Path,
                    default=DEFAULT_DIR / "imu_characterisation")
    ap.add_argument("--static-start", type=float, default=None,
                    help="Override static-window start [s].")
    ap.add_argument("--static-end", type=float, default=None,
                    help="Override static-window end [s].")
    ap.add_argument("--static-min-s", type=float, default=60.0,
                    help="Auto static-window length [s]. Longer is better for "
                         "the GM tau estimate (needs a few * tau of data).")
    ap.add_argument("--cruise-start", type=float, default=None)
    ap.add_argument("--cruise-end", type=float, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data)
    # Accept both schemas: df_flat (t, w_m_*, f_m_*) and the BLEDAR export
    # (time, gyro_*, acc_*).
    df = df.rename(columns={
        "time": "t",
        "gyro_x": "w_m_1", "gyro_y": "w_m_2", "gyro_z": "w_m_3",
        "acc_x": "f_m_1", "acc_y": "f_m_2", "acc_z": "f_m_3",
    })
    t = df.t.to_numpy()
    w_m = df[["w_m_1", "w_m_2", "w_m_3"]].to_numpy().T
    f_m = df[["f_m_1", "f_m_2", "f_m_3"]].to_numpy().T
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    print(f"  N={len(t)}, fs≈{fs:.1f} Hz, span=[{t[0]:.3f},{t[-1]:.3f}] s")

    # Static window.
    if args.static_start is not None and args.static_end is not None:
        static_t = (args.static_start, args.static_end)
        print(f"Static window (user): [{static_t[0]:.2f},{static_t[1]:.2f}] s")
    else:
        s0, s1, rms = find_static_window(t, w_m, min_s=args.static_min_s)
        static_t = (s0, s1)
        print(f"Static window (auto): [{s0:.2f},{s1:.2f}] s  "
              f"|w| rms={rms:.4e} rad/s")

    # Cruise window: default = 60s chunk from the middle of the flight.
    if args.cruise_start is not None and args.cruise_end is not None:
        cruise_t = (args.cruise_start, args.cruise_end)
    else:
        mid = 0.5 * (t[0] + t[-1])
        cruise_t = (mid - 30.0, mid + 30.0)
    print(f"Cruise window: [{cruise_t[0]:.2f},{cruise_t[1]:.2f}] s")

    plot_spectra(t, w_m, f_m, static_t, cruise_t, args.out_dir)

    # Allan on static window only.
    mask = (t >= static_t[0]) & (t <= static_t[1])
    w_s = w_m[:, mask]
    f_s = f_m[:, mask]
    plot_allan(w_s, f_s, fs, args.out_dir)

    plot_bias_observability(args.debug, args.truth, args.out_dir)

    print(f"\nPlots written to {args.out_dir}")


if __name__ == "__main__":
    main()
