#!/usr/bin/env python3
"""Save 3-sigma error plots for the RMSE matrix (no display).

Two kinds of 5x3 figure (5 states x 3 axes) are written, each as PNG + PDF +
SVG:
  * per-case : one figure per manifest row (that run's error + /-3sigma band),
               named <mode>_<asmp>_<pose>_<bias>_<method>.
  * grouped  : one figure per (mode, assumption) overlaying every combo,
               named group_<mode>_<asmp>.

Y-limits are zoomed so the interesting part is easy to read (--margin scales
the bound): --ylim half (default) zooms to the peak estimation error after the
midpoint (all of the second half stays visible), err to the end estimation
error, end to the steady-state +/-3sigma, max to the peak +/-3sigma.

A LaTeX appendix snippet (grouped figures only) is written to
<out_dir>/figures_appendix.tex: one full-width figure* per page, referencing
the PDFs via <tex-relpath>/<name>.pdf. \\input it once and every rerun refreshes
the figures in the paper.

Usage:
  plot_rmse_matrix_sigma.py <manifest.tsv> <out_dir>
      [--ylim end|max] [--margin F] [--tex-relpath figures_debug]
"""
import argparse
import os
import re
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")  # headless: save only, never display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# state -> (err cols, sig3 cols, axis labels)
STATES = OrderedDict([
    ("Att",  (["att_err_roll", "att_err_pitch", "att_err_yaw"],
              ["sig3_roll", "sig3_pitch", "sig3_yaw"], ["roll", "pitch", "yaw"])),
    ("Pos",  (["pos_err_n", "pos_err_e", "pos_err_d"],
              ["sig3_pos_n", "sig3_pos_e", "sig3_pos_d"], ["N", "E", "D"])),
    ("Vel",  (["vel_err_n", "vel_err_e", "vel_err_d"],
              ["sig3_vel_n", "sig3_vel_e", "sig3_vel_d"], ["N", "E", "D"])),
    ("AccB", (["acc_bias_err_x", "acc_bias_err_y", "acc_bias_err_z"],
              ["sig3_ab_x", "sig3_ab_y", "sig3_ab_z"], ["x", "y", "z"])),
    ("GyrB", (["gyro_bias_err_x", "gyro_bias_err_y", "gyro_bias_err_z"],
              ["sig3_gb_x", "sig3_gb_y", "sig3_gb_z"], ["x", "y", "z"])),
])

FORMATS = ("png", "pdf", "svg")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def bound(err, s3, ylim_mode):
    """Per-subplot y-limit magnitude.

    half: peak estimation error after the midpoint (max |error| over the second
          half of the run) -- keeps everything past halfway visible. [default]
    err : estimation error at the end (max |error| over the final 5% of the run).
    end : steady-state uncertainty (|+/-3sigma| at the last sample).
    max : peak uncertainty (max |+/-3sigma| over the whole run).
    """
    if ylim_mode in ("half", "err"):
        e = np.asarray(err, float)
        e = e[np.isfinite(e)]
        if e.size == 0:
            return None
        tail = e[e.size // 2:] if ylim_mode == "half" \
            else e[-max(1, int(0.05 * e.size)):]
        return float(np.max(np.abs(tail)))
    s3 = np.asarray(s3, float)
    s3 = s3[np.isfinite(s3)]
    if s3.size == 0:
        return None
    return float(abs(s3[-1])) if ylim_mode == "end" else float(np.max(np.abs(s3)))


def apply_ylim(ax, b, margin):
    if b is not None and b > 0 and np.isfinite(b):
        ax.set_ylim(-b * margin, b * margin)


def savefig_all(fig, outdir, name):
    for ext in FORMATS:
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"),
                    dpi=120 if ext == "png" else None)
    print(f"saved {os.path.join(outdir, name)}.{{{','.join(FORMATS)}}}")


def plot_case(mode, asmp, pose, bias, method, csv, outdir, ylim_mode, margin):
    """One 5x3 figure for a single run: its error + /-3sigma envelope."""
    try:
        df = pd.read_csv(csv)
    except Exception:
        print(f"skip (no csv): {csv}")
        return
    t = df["t"].to_numpy()
    fig, axes = plt.subplots(5, 3, figsize=(14, 16), sharex=True)
    for si, (state, (ec, sc, axl)) in enumerate(STATES.items()):
        for ai in range(3):
            ax = axes[si, ai]
            ax.plot(t, df[ec[ai]], color="tab:blue", lw=0.8, label="error")
            s3 = df[sc[ai]].to_numpy()
            ax.plot(t, s3, color="tab:red", lw=0.6, ls="--", label="+/-3sigma")
            ax.plot(t, -s3, color="tab:red", lw=0.6, ls="--")
            apply_ylim(ax, bound(df[ec[ai]].to_numpy(), s3, ylim_mode), margin)
            ax.set_title(f"{state} {axl[ai]}", fontsize=9)
            ax.grid(True, lw=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("t [s]")
    name = f"{slug(mode)}_{slug(asmp)}_{slug(pose)}_{slug(bias)}_{slug(method)}"
    fig.suptitle(f"{mode} / {asmp} / {pose}-{bias.upper()}-{method}"
                 " — error +/- 3sigma", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=15,
               bbox_to_anchor=(0.5, 0.965))
    savefig_all(fig, outdir, name)
    plt.close(fig)


def plot_group(mode, asmp, runs, outdir, ylim_mode, margin):
    """One 5x3 figure per (mode, assumption) overlaying every combo's error.

    Returns the figure basename if written, else None.
    """
    fig, axes = plt.subplots(5, 3, figsize=(14, 16), sharex=True)
    cmap = plt.cm.tab10
    # Per-subplot y-bound = max over runs of each run's end/max +/-3sigma.
    ybound = [[0.0] * 3 for _ in STATES]
    any_data = False
    for ri, (label, csv) in enumerate(runs):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        any_data = True
        t = df["t"].to_numpy()
        col = cmap(ri % 10)
        for si, (state, (ec, sc, axl)) in enumerate(STATES.items()):
            for ai in range(3):
                ax = axes[si, ai]
                ax.plot(t, df[ec[ai]], color=col, lw=0.7,
                        label=label if (si == 0 and ai == 0) else None)
                s3 = df[sc[ai]].to_numpy()
                ax.plot(t, s3, color=col, lw=0.5, ls="--", alpha=0.35)
                ax.plot(t, -s3, color=col, lw=0.5, ls="--", alpha=0.35)
                b = bound(df[ec[ai]].to_numpy(), s3, ylim_mode)
                if b is not None:
                    ybound[si][ai] = max(ybound[si][ai], b)
                if ri == 0:
                    ax.set_title(f"{state} {axl[ai]}", fontsize=9)
                    ax.grid(True, lw=0.3)
    if not any_data:
        plt.close(fig)
        return None
    for si in range(len(STATES)):
        for ai in range(3):
            apply_ylim(axes[si][ai], ybound[si][ai], margin)
    for ax in axes[-1]:
        ax.set_xlabel("t [s]")
    fig.suptitle(f"{mode} / {asmp} — error +/- 3sigma (all combos)", y=0.995)
    h, l = axes[0, 0].get_legend_handles_labels()
    ncol = min(len(l), 5)
    top = 0.90 if len(l) > 5 else 0.93  # extra room when the legend wraps
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.legend(h, l, loc="upper center", ncol=ncol, fontsize=12,
               bbox_to_anchor=(0.5, 0.975))
    name = f"group_{slug(mode)}_{slug(asmp)}"
    savefig_all(fig, outdir, name)
    plt.close(fig)
    return name


def write_appendix(outdir, relpath, entries):
    """Grouped-figures-only LaTeX appendix: one full-width figure* per page."""
    lines = [
        "% Auto-generated by plot_rmse_matrix_sigma.py -- DO NOT EDIT BY HAND.",
        "% Regenerated on every matrix run; \\input this once in your appendix.",
        "",
    ]
    for name, mode, asmp in entries:
        lines += [
            r"\begin{figure*}[p]",
            r"  \centering",
            rf"  \includegraphics[width=\textwidth]{{{relpath}/{name}.pdf}}",
            rf"  \caption{{Error and $\pm3\sigma$ bounds -- {mode}, {asmp} "
            r"(all combinations).}",
            rf"  \label{{fig:{name}}}",
            r"\end{figure*}",
            r"\clearpage",
            "",
        ]
    path = os.path.join(outdir, "figures_appendix.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote appendix snippet {path} ({len(entries)} figures)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("outdir")
    ap.add_argument("--ylim", choices=("half", "err", "end", "max"),
                    default="half",
                    help="y-limit bound: peak estimation error after the "
                         "midpoint (half, default), end estimation error (err), "
                         "steady-state uncertainty (end), or peak uncertainty "
                         "(max)")
    ap.add_argument("--margin", type=float, default=1.25,
                    help="y-limit = bound * margin (default 1.25)")
    ap.add_argument("--tex-relpath", default="figures_debug",
                    help="path prefix used inside the appendix \\includegraphics")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = [l.rstrip("\n").split("\t")
            for l in open(args.manifest) if l.strip()]

    groups = OrderedDict()
    for mode, asmp, pose, bias, method, csv in rows:
        plot_case(mode, asmp, pose, bias, method, csv, args.outdir,
                  args.ylim, args.margin)
        groups.setdefault((mode, asmp), []).append(
            (f"{pose}-{bias.upper()}-{method}", csv))

    appendix = []
    for (mode, asmp), runs in groups.items():
        name = plot_group(mode, asmp, runs, args.outdir, args.ylim, args.margin)
        if name:
            appendix.append((name, mode, asmp))

    write_appendix(args.outdir, args.tex_relpath, appendix)


if __name__ == "__main__":
    main()
