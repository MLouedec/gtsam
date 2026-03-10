#!/usr/bin/env python3
"""Plot estimation errors with 3-sigma bounds in a 5x3 grid.

Compares up to four (preintegrator, bias) variants on the same axes:
  - SE3  + Constant Bias    -> gtsam_fork_test_cb.csv
  - SE3  + Gauss-Markov     -> gtsam_fork_test_gm.csv
  - SE23 + Constant Bias    -> gtsam_fork_test_cb_se23.csv
  - SE23 + Gauss-Markov     -> gtsam_fork_test_gm_se23.csv

Any subset that exists on disk will be plotted.

Modes:
  default        error line + +/-3 sigma envelope per dataset
  --mode sigma1  1-sigma (|sigma|) curve over time per dataset, no error line
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

# One color per (preint, bias) case. Error lines are drawn solid; sigma lines
# are drawn dashed — same dash pattern across all cases, colors distinguish.
#
# (tag, filename_suffix, label, color)
# Earlier entries draw last (on top).
PLOT_ORDER = [
    ("se23_gm", "gm_se23", "SE23 + GM", "tab:blue"),
    ("se23_cb", "cb_se23", "SE23 + CB", "tab:green"),
    ("se3_gm",  "gm",      "SE3 + GM",  "tab:red"),
    ("se3_cb",  "cb",      "SE3 + CB",  "tab:purple"),
]

# Shared line styles.
ERR_LINESTYLE = "solid"
SIG_DASHES = (4, 2)

RESULTS_DIR = "/Users/ghms/ws/ntnu/parnav_ins_simulator/results"
# Default output locations (the Overleaf IEEE TAES paper project).
OVERLEAF_DIR = "~/Dropbox/Apper/Overleaf/IEEE TAES Journal"
FIG_DIR = OVERLEAF_DIR + "/figures"
TABLE_DIR = OVERLEAF_DIR + "/tables"

# RMSE substate structure: (group_label, unit, convert_to_deg,
#                           [(component_label, error_col), ...]).
RMSE_STRUCTURE = [
    ("Position", "m", False,
     [("N", "pos_err_n"), ("E", "pos_err_e"), ("D", "pos_err_d")]),
    ("Velocity", "m/s", False,
     [("N", "vel_err_n"), ("E", "vel_err_e"), ("D", "vel_err_d")]),
    ("Attitude", "deg", True,
     [("roll", "att_err_roll"), ("pitch", "att_err_pitch"),
      ("yaw", "att_err_yaw")]),
    ("Acc bias", "m/s^2", False,
     [("x", "acc_bias_err_x"), ("y", "acc_bias_err_y"),
      ("z", "acc_bias_err_z")]),
    ("Gyro bias", "deg/s", True,
     [("x", "gyro_bias_err_x"), ("y", "gyro_bias_err_y"),
      ("z", "gyro_bias_err_z")]),
]


def _rmse_rows(detail):
    """Build the substate row list for a given detail level. Each row is
    (label, component_label, [error_cols], to_deg). There is no separate unit
    column: in 'all' mode the unit sits in `label` on the row below the group
    name; in 'group'/'axis' it is appended inline to the group name."""
    rows = []
    for grp, unit, to_deg, comps in RMSE_STRUCTURE:
        norm_cols = [c for _, c in comps]
        if detail == "group":
            rows.append((f"{grp} [{unit}]", "", norm_cols, to_deg))
        elif detail == "axis":
            for clabel, col in comps:
                rows.append((f"{grp} [{unit}]", clabel, [col], to_deg))
        else:  # all: norm row, then unit row + per-axis rows
            rows.append((grp, "Norm", norm_cols, to_deg))
            for i, (clabel, col) in enumerate(comps):
                rows.append((f"[{unit}]" if i == 0 else "", clabel, [col],
                             to_deg))
    return rows


# ============================================================================


def _substate_rmse(df, cols, to_deg):
    """RMSE of a substate. For multi-column groups this is the RMS of the
    per-sample vector magnitude: sqrt(mean(sum_axes err^2))."""
    sq = np.zeros(len(df))
    for c in cols:
        e = df[c].values
        if to_deg:
            e = np.degrees(e)
        sq = sq + e * e
    return float(np.sqrt(np.nanmean(sq)))


def _downsample(df, max_points):
    """Stride-subsample a dataframe to at most max_points rows (for plotting
    only). 0 or a small frame is returned unchanged."""
    if max_points and len(df) > max_points:
        stride = int(np.ceil(len(df) / max_points))
        return df.iloc[::stride]
    return df


def _mc_average(dfs):
    """Combine N Monte Carlo run dataframes into one whose error columns hold
    the per-sample RMS across runs, sqrt(mean_runs err^2). Squaring this in
    _substate_rmse then averaging over time yields the pooled Monte Carlo RMSE
    sqrt(mean over runs & time of err^2). Non-error columns come from run 1."""
    n = min(len(d) for d in dfs)
    out = dfs[0].iloc[:n].copy()
    err_cols = [c for c in out.columns if "err" in c]
    for c in err_cols:
        stacked = np.stack([d[c].values[:n].astype(float) for d in dfs])
        out[c] = np.sqrt(np.nanmean(stacked ** 2, axis=0))
    return out


# Baseline and the four canonical (preint, bias) combinations. Each %impr
# column compares one variant against the SE3 + CB baseline.
RMSE_BASELINE = "SE3 + CB"
RMSE_COLS = ["SE3 + CB", "SE3 + GM", "SE23 + CB", "SE23 + GM"]
IMPR_COLS = [
    ("%impr of GM", "SE3 + GM"),
    ("%impr of SE23", "SE23 + CB"),
    ("%impr of SE23+GM", "SE23 + GM"),
]
# Short sub-headers used under the centered "% Improvement" banner (the
# "impr. (%)" wording lives in the banner, so the columns just name the
# variant being compared to the SE3+CB baseline).
IMPR_SHORT = ["GM", "SE23", "SE23+GM"]
DIFF_SHORT = "SE3->SE23 GM"


def print_rmse_table(datasets, detail):
    """Print an RMSE table: one row per substate.

    Columns: the four RMSE values (SE3/SE23 x CB/GM), three %impr columns
    (each variant vs the SE3+CB baseline; green = better, red = worse), and
    a final column naming the lowest-RMSE combination for that substate.
    """
    substates = _rmse_rows(detail)

    use_color = sys.stdout.isatty()
    GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"

    def sign_color(cell, impr):
        if not use_color or impr == 0:
            return cell
        return f"{GREEN if impr > 0 else RED}{cell}{RESET}"

    by_label = {ds[0]: ds[1] for ds in datasets}

    grp_w = max([len("Substate")] + [len(s[0]) for s in substates]) + 2
    comp_w = max([0] + [len(s[1]) for s in substates])
    if comp_w:
        comp_w += 2
    col_w = max(12, max(len(c) for c in RMSE_COLS) + 2)
    pct_headers = IMPR_SHORT + [DIFF_SHORT]
    impr_w = max(10, max(len(h) for h in pct_headers) + 2)
    sep = " | "  # vertical divider between RMSE and % blocks

    left_w = grp_w + (comp_w if comp_w else 0)
    rmse_w = col_w * len(RMSE_COLS)
    pct_w = impr_w * len(pct_headers)

    # Banner row: centre "RMSE" and "% Improvement" over their blocks.
    banner = " " * left_w + "RMSE".center(rmse_w) + sep \
        + "% Improvement".center(pct_w)
    # Column sub-header row.
    header = "Substate".ljust(grp_w) + ("Comp".ljust(comp_w) if comp_w else "")
    header += "".join(c.rjust(col_w) for c in RMSE_COLS) + sep
    header += "".join(h.rjust(impr_w) for h in pct_headers)
    header += "  " + "Lowest RMSE"
    print(f"\n=== RMSE ({detail}) ===")
    print(banner)
    print(header)
    print("-" * len(header))

    for idx, (label, comp, cols, to_deg) in enumerate(substates):
        # Blank separator before each new group in 'all' mode.
        if detail == "all" and comp == "Norm" and idx:
            print()
        rmse = {lbl: _substate_rmse(by_label[lbl], cols, to_deg)
                for lbl in RMSE_COLS if lbl in by_label}
        row = label.ljust(grp_w) + (comp.ljust(comp_w) if comp_w else "")

        for lbl in RMSE_COLS:
            row += (f"{rmse[lbl]:>{col_w}.2f}" if lbl in rmse
                    else "n/a".rjust(col_w))
        row += sep

        base = rmse.get(RMSE_BASELINE)
        for _h, var in IMPR_COLS:
            if base and var in rmse:
                impr = (base - rmse[var]) / base * 100.0
                row += sign_color(f"{impr:>{impr_w}.2f}", impr)
            else:
                row += "n/a".rjust(impr_w)

        # %diff of SE23+GM relative to SE3+GM (isolates SE23 within GM).
        gm3, gm23 = rmse.get("SE3 + GM"), rmse.get("SE23 + GM")
        if gm3 and gm23 is not None:
            diff = (gm3 - gm23) / gm3 * 100.0
            row += sign_color(f"{diff:>{impr_w}.2f}", diff)
        else:
            row += "n/a".rjust(impr_w)

        best = min(rmse, key=rmse.get) if rmse else "n/a"
        row += "  " + best
        print(row)


# ---------------------------------------------------------------------------
# Norm-only tables (group magnitudes) + LaTeX export
# ---------------------------------------------------------------------------

# LaTeX (IEEE TAES) header/label formatting.
LATEX_COMBO = {
    "SE3 + CB": "SE(3)+CB",
    "SE3 + GM": "SE(3)+GM",
    "SE23 + CB": r"SE$_2$(3)+CB",
    "SE23 + GM": r"SE$_2$(3)+GM",
}
# Short sub-headers under the "% Improvement" banner (no "impr. (%)" text).
LATEX_IMPR_SHORT = ["GM", r"SE$_2$(3)", r"SE$_2$(3)+GM"]
LATEX_DIFF_SHORT = r"SE(3)$\rightarrow$SE$_2$(3) GM"


def _norm_rmse(datasets):
    """Group-norm RMSE per combo: list of (group, unit, {combo: rmse})."""
    by_label = {ds[0]: ds[1] for ds in datasets}
    out = []
    for grp, unit, to_deg, comps in RMSE_STRUCTURE:
        cols = [c for _, c in comps]
        vals = {lbl: _substate_rmse(by_label[lbl], cols, to_deg)
                for lbl in RMSE_COLS if lbl in by_label}
        out.append((grp, unit, vals))
    return out


def _pct_cells(vals):
    """Return (impr:[(header,value|None)], diff|None, best_label) for a row."""
    base = vals.get(RMSE_BASELINE)
    impr = []
    for h, var in IMPR_COLS:
        impr.append((h, (base - vals[var]) / base * 100.0
                     if base and var in vals else None))
    gm3, gm23 = vals.get("SE3 + GM"), vals.get("SE23 + GM")
    diff = (gm3 - gm23) / gm3 * 100.0 if gm3 and gm23 is not None else None
    best = min(vals, key=vals.get) if vals else "n/a"
    return impr, diff, best


def print_norm_rmse_table(datasets):
    """Table 2: norm-only RMSE values (no percentages), with an RMSE banner."""
    rows = _norm_rmse(datasets)
    name_w = max([len("Substate")]
                 + [len(f"{g} [{u}]") for g, u, _ in rows]) + 2
    col_w = max(12, max(len(c) for c in RMSE_COLS) + 2)
    rmse_w = col_w * len(RMSE_COLS)
    banner = " " * name_w + "RMSE".center(rmse_w)
    header = "Substate".ljust(name_w) + "".join(c.rjust(col_w) for c in RMSE_COLS)
    print("\n=== RMSE (norm only) ===")
    print(banner)
    print(header)
    print("-" * len(header))
    for grp, unit, vals in rows:
        row = f"{grp} [{unit}]".ljust(name_w)
        for lbl in RMSE_COLS:
            row += (f"{vals[lbl]:>{col_w}.2f}" if lbl in vals
                    else "n/a".rjust(col_w))
        print(row)


def print_norm_pct_table(datasets):
    """Table 3: norm-only percentage differences (colorized) + Lowest RMSE,
    with a centered '% Improvement' banner over the comparison columns."""
    rows = _norm_rmse(datasets)
    use_color = sys.stdout.isatty()
    GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"

    def sc(cell, v):
        if not use_color or v == 0:
            return cell
        return f"{GREEN if v > 0 else RED}{cell}{RESET}"

    pct_headers = IMPR_SHORT + [DIFF_SHORT]
    name_w = max([len("Substate")]
                 + [len(f"{g} [{u}]") for g, u, _ in rows]) + 2
    impr_w = max(10, max(len(h) for h in pct_headers) + 2)
    pct_w = impr_w * len(pct_headers)
    banner = " " * name_w + "% Improvement".center(pct_w)
    header = "Substate".ljust(name_w)
    header += "".join(h.rjust(impr_w) for h in pct_headers)
    header += "  " + "Lowest RMSE"
    print("\n=== RMSE %-differences (norm only) ===")
    print(banner)
    print(header)
    print("-" * len(header))
    for grp, unit, vals in rows:
        impr, diff, best = _pct_cells(vals)
        row = f"{grp} [{unit}]".ljust(name_w)
        for _h, v in impr:
            row += sc(f"{v:>{impr_w}.2f}", v) if v is not None \
                else "n/a".rjust(impr_w)
        row += sc(f"{diff:>{impr_w}.2f}", diff) if diff is not None \
            else "n/a".rjust(impr_w)
        row += "  " + best
        print(row)


def _latex_num(x):
    return f"{x:.2f}" if x is not None else "--"


def _latex_pct(v, color=False):
    if v is None:
        return "--"
    s = f"{v:.2f}"
    if color and v != 0:
        return rf"\textcolor{{{'green!55!black' if v > 0 else 'red'}}}{{{s}}}"
    return s


def _latex_unit(u):
    return u.replace("^2", r"$^2$")


def _latex_table(caption, label, colspec, header_cells, body_rows,
                 wide=False, banner=None):
    env = "table*" if wide else "table"
    lines = [rf"\begin{{{env}}}[!t]",
             r"\renewcommand{\arraystretch}{1.2}",
             rf"\caption{{{caption}}}",
             rf"\label{{{label}}}",
             r"\centering",
             rf"\begin{{tabular}}{{{colspec}}}",
             r"\toprule"]
    if banner:
        lines += banner  # multicolumn banner row + \cmidrule lines
    lines += [" & ".join(header_cells) + r" \\", r"\midrule"]
    lines += body_rows
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    return "\n".join(lines)


def _mc_note(n_runs):
    """Caption suffix noting the Monte Carlo averaging (empty for a single run)."""
    return (rf" Averaged over $N={n_runs}$ Monte Carlo runs." if n_runs > 1
            else "")


def latex_full_table(datasets, detail, n_runs=1):
    """LaTeX for Table 1 (the full console table, matching --rmse detail).
    A centered RMSE / % Improvement banner sits above the columns, with a
    vertical rule separating the two blocks; % cells are colorized."""
    substates = _rmse_rows(detail)
    by_label = {ds[0]: ds[1] for ds in datasets}
    has_comp = any(s[1] for s in substates)
    n_left = 1 + (1 if has_comp else 0)
    n_rmse = len(RMSE_COLS)
    n_pct = len(IMPR_COLS) + 1
    colspec = "l" * n_left + "c" * n_rmse + "|" + "c" * n_pct + "l"

    # Banner: RMSE over the value columns, % Improvement over the comparisons.
    r0, r1 = n_left + 1, n_left + n_rmse
    p0, p1 = r1 + 1, r1 + n_pct
    banner_cells = [""] * n_left
    banner_cells.append(rf"\multicolumn{{{n_rmse}}}{{c}}{{RMSE}}")
    banner_cells.append(
        rf"\multicolumn{{{n_pct}}}{{|c}}{{\% Improvement}}")
    banner_cells.append("")  # Lowest RMSE column
    banner = [" & ".join(banner_cells) + r" \\",
              rf"\cmidrule(lr){{{r0}-{r1}}} \cmidrule(lr){{{p0}-{p1}}}"]

    header = ["Substate"] + (["Comp."] if has_comp else [])
    header += [LATEX_COMBO[c] for c in RMSE_COLS]
    header += LATEX_IMPR_SHORT + [LATEX_DIFF_SHORT, "Lowest RMSE"]
    body = []
    for i, (label, comp, cols, to_deg) in enumerate(substates):
        if detail == "all" and comp == "Norm" and i:
            body.append(r"\addlinespace")
        rmse = {lbl: _substate_rmse(by_label[lbl], cols, to_deg)
                for lbl in RMSE_COLS if lbl in by_label}
        cells = [_latex_unit(label)] + ([comp] if has_comp else [])
        cells += [_latex_num(rmse.get(c)) for c in RMSE_COLS]
        impr, diff, best = _pct_cells(rmse)
        cells += [_latex_pct(v, color=True) for _h, v in impr]
        cells += [_latex_pct(diff, color=True), LATEX_COMBO.get(best, best)]
        body.append(" & ".join(cells) + r" \\")
    return _latex_table(
        f"Estimation RMSE and relative improvement ({detail}).{_mc_note(n_runs)}",
        f"tab:rmse_{detail}", colspec, header, body, wide=True, banner=banner)


def latex_norm_rmse_table(datasets, n_runs=1):
    """LaTeX for Table 2 (norm-only RMSE) with a centered RMSE banner."""
    rows = _norm_rmse(datasets)
    n = len(RMSE_COLS)
    colspec = "l" + "c" * n
    banner = ["" + rf" & \multicolumn{{{n}}}{{c}}{{RMSE}} \\",
              rf"\cmidrule(lr){{2-{1 + n}}}"]
    header = ["Substate"] + [LATEX_COMBO[c] for c in RMSE_COLS]
    body = []
    for grp, unit, vals in rows:
        cells = [f"{grp} [{_latex_unit(unit)}]"]
        cells += [_latex_num(vals.get(c)) for c in RMSE_COLS]
        body.append(" & ".join(cells) + r" \\")
    return _latex_table(
        f"Estimation RMSE (vector-norm per substate).{_mc_note(n_runs)}",
        "tab:rmse_norm", colspec, header, body, banner=banner)


def latex_norm_pct_table(datasets, n_runs=1):
    """LaTeX for Table 3 (norm-only percentages, colorized) + Lowest RMSE,
    under a centered % Improvement banner."""
    rows = _norm_rmse(datasets)
    n = len(IMPR_COLS) + 1
    colspec = "l" + "c" * n + "l"
    banner = ["" + rf" & \multicolumn{{{n}}}{{c}}{{\% Improvement}} & \\",
              rf"\cmidrule(lr){{2-{1 + n}}}"]
    header = ["Substate"] + LATEX_IMPR_SHORT + [LATEX_DIFF_SHORT, "Lowest RMSE"]
    body = []
    for grp, unit, vals in rows:
        impr, diff, best = _pct_cells(vals)
        cells = [f"{grp} [{_latex_unit(unit)}]"]
        cells += [_latex_pct(v, color=True) for _h, v in impr]
        cells += [_latex_pct(diff, color=True), LATEX_COMBO.get(best, best)]
        body.append(" & ".join(cells) + r" \\")
    return _latex_table(
        "Relative RMSE differences (vector-norm per substate). "
        r"Green: improvement over the SE(3)+CB baseline; red: regression."
        + _mc_note(n_runs),
        "tab:rmse_norm_pct", colspec, header, body, banner=banner)


def write_latex_tables(datasets, detail, out_dir, n_runs=1):
    """Write the three LaTeX tables to out_dir; return the directory path."""
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    files = {
        "table1_full.tex": latex_full_table(datasets, detail, n_runs),
        "table2_norm_rmse.tex": latex_norm_rmse_table(datasets, n_runs),
        "table3_norm_pct.tex": latex_norm_pct_table(datasets, n_runs),
    }
    note = ("% Requires \\usepackage{booktabs} and \\usepackage{xcolor}.\n")
    for name, body in files.items():
        with open(os.path.join(out_dir, name), "w") as fh:
            fh.write(note + body + "\n")
    return out_dir


def tag_from_path(path: str) -> str:
    """Infer a PLOT_ORDER tag from an explicit CSV path."""
    base = os.path.basename(path)
    if "_se23" in base:
        return "se23_gm" if "_gm" in base else "se23_cb"
    return "se3_gm" if "_gm" in base else "se3_cb"


def main():
    parser = argparse.ArgumentParser(description="Plot estimation errors / sigmas")
    parser.add_argument(
        "--dir",
        default=RESULTS_DIR,
        help="Directory containing gtsam_fork_test_{cb,gm}[_se23].csv",
    )
    parser.add_argument(
        "--csv",
        nargs="*",
        help="Explicit CSV file path(s). Overrides --dir auto-detection.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix appended before .csv when auto-detecting "
             "(e.g. '_none_10s'). Empty matches the default filenames.",
    )
    parser.add_argument(
        "--ylim",
        choices=("auto", "sigma", "error"),
        default="auto",
        help="Per-subplot y-limit strategy: "
             "'auto' = matplotlib default; "
             "'sigma' = max |3 sigma| across datasets, +/-10%% padding "
             "(error lines may clip out when they diverge); "
             "'error' = max |error| across datasets, +/-10%% padding.",
    )
    parser.add_argument(
        "--mode",
        choices=("error3sigma", "sigma1"),
        default="error3sigma",
        help="'error3sigma' (default) plots error line with +/-3 sigma "
             "envelope; 'sigma1' plots the 1-sigma curve over time only.",
    )
    parser.add_argument(
        "--rmse",
        choices=("group", "axis", "all", "none"),
        default="group",
        help="Print an RMSE table: 'group' = 5 substates "
             "(position/velocity/attitude/acc bias/gyro bias magnitudes); "
             "'axis' = 15 per-axis substates (N/E/D, roll/pitch/yaw, ...); "
             "'all' = each group's norm followed by its per-axis rows; "
             "'none' = skip.",
    )
    parser.add_argument(
        "--fig-dir",
        default=FIG_DIR,
        help=f"Directory to write the figures (PNGs). Default: {FIG_DIR}",
    )
    parser.add_argument(
        "--scenario",
        default="",
        help="If set, figures and tables are nested under <scenario> "
             "(e.g. sim/dead_reckoning, bledar_gnss).",
    )
    parser.add_argument(
        "--result-prefix",
        default="gtsam_fork_test_",
        help="Result CSV filename prefix (default gtsam_fork_test_; use "
             "bledar_ for the multirotor runs).",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=20000,
        help="Downsample each series to at most this many points for the "
             "figures (RMSE tables always use the full data). Keeps very "
             "high-rate datasets renderable. 0 disables.",
    )
    parser.add_argument(
        "--latex-dir",
        default=TABLE_DIR,
        help="Directory to write the three IEEE-TAES-style LaTeX tables "
             f"(full, norm-RMSE, norm-%%-diff). Default: {TABLE_DIR}. "
             "Skipped when --rmse none.",
    )
    parser.add_argument(
        "--box-yscale",
        choices=("log", "linear"),
        default="log",
        help="Y-axis scale for the box plot (default: log).",
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=1,
        help="Number of Monte Carlo runs. The plots use run 01; the RMSE "
             "tables average the squared error across all runs (reads "
             "gtsam_fork_test_*_runNN.csv).",
    )
    args = parser.parse_args()

    # Figures are written to fig_dir (created if missing); CSVs are read from
    # --dir. Tables go to --latex-dir (created by write_latex_tables). When a
    # scenario is given, both nest under sim/<scenario> (e.g. dead_reckoning).
    args.fig_dir = os.path.expanduser(args.fig_dir)
    args.latex_dir = os.path.expanduser(args.latex_dir)
    if args.scenario:
        args.fig_dir = os.path.join(args.fig_dir, args.scenario)
        args.latex_dir = os.path.join(args.latex_dir, args.scenario)
    os.makedirs(args.fig_dir, exist_ok=True)

    # Index PLOT_ORDER by tag for quick lookup.
    by_tag = {e[0]: e for e in PLOT_ORDER}

    # Build list of (label, dataframe, color). `datasets` holds the first run
    # (used for every plot); `mc_datasets` holds the Monte-Carlo-averaged
    # error columns (used for the RMSE tables).
    datasets = []
    mc_datasets = []
    mc_all = []  # (label, [all run dataframes], color) -> pooled box plot

    def _run_path(fname_suffix, run):
        pre = args.result_prefix
        if args.mc_runs > 1:
            return os.path.join(
                args.dir, f"{pre}{fname_suffix}_run{run:02d}.csv")
        return os.path.join(
            args.dir, f"{pre}{fname_suffix}{args.suffix}.csv")

    if args.csv:
        for path in args.csv:
            entry = by_tag.get(tag_from_path(path), PLOT_ORDER[0])
            df = pd.read_csv(path)
            datasets.append((entry[2], df, entry[3]))
            mc_datasets.append((entry[2], df, entry[3]))
            mc_all.append((entry[2], [df], entry[3]))
    else:
        for tag, fname_suffix, label, color in reversed(PLOT_ORDER):
            first = _run_path(fname_suffix, 1)
            if not os.path.exists(first):
                continue
            first_df = pd.read_csv(first)
            datasets.append((label, first_df, color))
            # Collect all available runs for the RMSE average.
            run_dfs = [first_df]
            for r in range(2, args.mc_runs + 1):
                p = _run_path(fname_suffix, r)
                if os.path.exists(p):
                    run_dfs.append(pd.read_csv(p))
            mc_datasets.append(
                (label, _mc_average(run_dfs) if len(run_dfs) > 1 else first_df,
                 color))
            mc_all.append((label, run_dfs, color))

    if not datasets:
        print(f"No result files found in {args.dir}")
        return

    if args.rmse != "none":
        if args.mc_runs > 1:
            print(f"(RMSE averaged over up to {args.mc_runs} Monte Carlo runs)")
        print_rmse_table(mc_datasets, args.rmse)   # Table 1: full
        print_norm_rmse_table(mc_datasets)         # Table 2: norm RMSE
        print_norm_pct_table(mc_datasets)          # Table 3: norm %-diff
        out_dir = write_latex_tables(mc_datasets, args.rmse, args.latex_dir,
                                     args.mc_runs)
        print(f"\nLaTeX tables written to {out_dir}")

    # Downsample the first-run dataframes for the time-series figures (keeps
    # very high-rate datasets renderable; RMSE tables use the full data).
    plot_datasets = [(lbl, _downsample(df, args.max_plot_points), c)
                     for lbl, df, c in datasets]

    # Combined figures (all available combinations overlaid).
    plot_main_grid(plot_datasets, args)
    plot_lie_group_axes(plot_datasets, args)
    plot_lie_group_norm(plot_datasets, args)

    # One figure per Lie-group block, plus a box plot (pooled over all runs).
    plot_lie_group_by_group(plot_datasets, args)
    plot_box(mc_all, args)

    plt.show()


def plot_main_grid(datasets, args, suffix=""):
    """The 5x3 per-axis error + 3-sigma grid (position, velocity, attitude,
    accelerometer bias, gyroscope bias), overlaying the given datasets. Saved
    as 3sigma_comparison<suffix>.png (or 1sigma_comparison<suffix>.png)."""
    rad2deg = np.degrees

    # Define the 5x3 grid: (error_col, sig3_col, label, unit, convert_to_deg)
    grid = [
        # Row 1: Position
        ("pos_err_n", "sig3_pos_n", "North position", "m", False),
        ("pos_err_e", "sig3_pos_e", "East position", "m", False),
        ("pos_err_d", "sig3_pos_d", "Down position", "m", False),
        # Row 2: Velocity
        ("vel_err_n", "sig3_vel_n", "North velocity", "m/s", False),
        ("vel_err_e", "sig3_vel_e", "East velocity", "m/s", False),
        ("vel_err_d", "sig3_vel_d", "Down velocity", "m/s", False),
        # Row 3: Attitude (rad -> deg)
        ("att_err_roll", "sig3_roll", "Roll", "deg", True),
        ("att_err_pitch", "sig3_pitch", "Pitch", "deg", True),
        ("att_err_yaw", "sig3_yaw", "Yaw", "deg", True),
        # Row 4: Accelerometer bias
        ("acc_bias_err_x", "sig3_ab_x", "Acc bias x", "m/s^2", False),
        ("acc_bias_err_y", "sig3_ab_y", "Acc bias y", "m/s^2", False),
        ("acc_bias_err_z", "sig3_ab_z", "Acc bias z", "m/s^2", False),
        # Row 5: Gyroscope bias (rad/s -> deg/s)
        ("gyro_bias_err_x", "sig3_gb_x", "Gyro bias x", "deg/s", True),
        ("gyro_bias_err_y", "sig3_gb_y", "Gyro bias y", "deg/s", True),
        ("gyro_bias_err_z", "sig3_gb_z", "Gyro bias z", "deg/s", True),
    ]

    n_datasets = len(datasets)
    title = " vs ".join(d[0] for d in datasets) if n_datasets > 1 else datasets[0][0]
    fig, axes = plt.subplots(5, 3, figsize=(16, 14), sharex=True)

    mode_title = ("Estimation errors with 3-sigma bounds"
                  if args.mode == "error3sigma"
                  else "1-sigma over time")
    fig.suptitle(f"{mode_title} — {title}", fontsize=14)

    # Reduce sigma fill alpha when many datasets overlap.
    fill_alpha = 0.06 if n_datasets <= 2 else 0.04

    for i, (err_col, sig_col, label, unit, to_deg) in enumerate(grid):
        row, col = divmod(i, 3)
        ax = axes[row, col]

        max_sig = 0.0
        max_err = 0.0
        for ds in datasets:
            ds_label, df, color = ds
            n = len(df)
            # Use the saved timestamp column; fall back to 10 Hz for old CSVs.
            t = df["t"].values if "t" in df.columns else np.arange(n) * 0.1

            err = df[err_col].values
            sig3 = df[sig_col].values

            if to_deg:
                err = rad2deg(err)
                sig3 = rad2deg(sig3)

            max_sig = max(max_sig, float(np.nanmax(np.abs(sig3))))
            max_err = max(max_err, float(np.nanmax(np.abs(err))))

            slabel = f" ({ds_label})" if n_datasets > 1 else ""

            if args.mode == "error3sigma":
                ax.plot(t, err, color=color, linewidth=0.9,
                        linestyle=ERR_LINESTYLE, alpha=0.9,
                        label=f"Error{slabel}")
                ax.plot(t, sig3, color=color, dashes=SIG_DASHES,
                        linewidth=0.9, alpha=0.9,
                        label=f"+3sigma{slabel}")
                ax.plot(t, -sig3, color=color, dashes=SIG_DASHES,
                        linewidth=0.9, alpha=0.9,
                        label=f"-3sigma{slabel}")
                ax.fill_between(t, -sig3, sig3, color=color, alpha=fill_alpha)
            else:  # sigma1
                sig1 = np.abs(sig3) / 3.0
                ax.plot(t, sig1, color=color, dashes=SIG_DASHES,
                        linewidth=1.0, alpha=0.95,
                        label=f"1sigma{slabel}")

        # Per-subplot y-limits.
        if args.mode == "error3sigma" and args.ylim in ("sigma", "error"):
            ref = max_sig if args.ylim == "sigma" else max_err
            if ref > 0 and np.isfinite(ref):
                pad = 1.1 * ref
                ax.set_ylim(-pad, pad)
        elif args.mode == "sigma1":
            # Sigma is non-negative; don't force symmetric limits.
            if args.ylim in ("sigma", "error") and max_sig > 0:
                ax.set_ylim(0.0, 1.1 * max_sig / 3.0)

        ax.set_ylabel(f"{label} [{unit}]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Build a compact shared legend at the figure level.
    from matplotlib.lines import Line2D
    handles = []
    for ds in datasets:
        ds_label, _, color = ds
        if args.mode == "error3sigma":
            handles.append(Line2D([0], [0], color=color, linewidth=1.4,
                                  linestyle=ERR_LINESTYLE,
                                  label=f"Error ({ds_label})"))
            handles.append(Line2D([0], [0], color=color, dashes=SIG_DASHES,
                                  linewidth=1.4,
                                  label=f"+/-3sigma ({ds_label})"))
        else:
            handles.append(Line2D([0], [0], color=color, dashes=SIG_DASHES,
                                  linewidth=1.4,
                                  label=f"1sigma ({ds_label})"))
    fig.legend(handles=handles, loc="upper right", fontsize=7,
               ncol=min(n_datasets, 4), framealpha=0.9)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    base = "3sigma_comparison" if args.mode == "error3sigma" else "1sigma_comparison"
    out_name = os.path.join(args.fig_dir, f"{base}{suffix}.png")
    plt.savefig(out_name, dpi=150)
    print(f"Saved to {out_name}")


# SE_2(3) Lie-group blocks used by the norm / per-group / box figures.
# (name, err_cols, sig_cols, unit, to_deg)
LIE_GROUPS = [
    ("Position", ["pos_err_n", "pos_err_e", "pos_err_d"],
     ["sig3_pos_n", "sig3_pos_e", "sig3_pos_d"], "m", False),
    ("Velocity", ["vel_err_n", "vel_err_e", "vel_err_d"],
     ["sig3_vel_n", "sig3_vel_e", "sig3_vel_d"], "m/s", False),
    ("Attitude", ["att_err_roll", "att_err_pitch", "att_err_yaw"],
     ["sig3_roll", "sig3_pitch", "sig3_yaw"], "deg", True),
]


def _group_mag(df, cols, to_deg):
    """Per-sample vector-norm of the given columns (deg-converted if needed)."""
    sq = np.zeros(len(df))
    for c in cols:
        v = df[c].values.astype(float)
        if to_deg:
            v = np.degrees(v)
        sq = sq + v * v
    return np.sqrt(sq)


def plot_lie_group_axes(datasets, args, suffix=""):
    """Per-axis figure: error + 3-sigma for the SE_2(3) Lie-group state only —
    attitude, velocity, position (R, v, p). Biases are excluded (not part of
    the group manifold). 3x3 grid, overlaying whatever datasets are present."""
    rad2deg = np.degrees
    grid = [
        ("pos_err_n", "sig3_pos_n", "North position", "m", False),
        ("pos_err_e", "sig3_pos_e", "East position", "m", False),
        ("pos_err_d", "sig3_pos_d", "Down position", "m", False),
        ("vel_err_n", "sig3_vel_n", "North velocity", "m/s", False),
        ("vel_err_e", "sig3_vel_e", "East velocity", "m/s", False),
        ("vel_err_d", "sig3_vel_d", "Down velocity", "m/s", False),
        ("att_err_roll", "sig3_roll", "Roll", "deg", True),
        ("att_err_pitch", "sig3_pitch", "Pitch", "deg", True),
        ("att_err_yaw", "sig3_yaw", "Yaw", "deg", True),
    ]
    n_datasets = len(datasets)
    title = " vs ".join(d[0] for d in datasets) if n_datasets > 1 \
        else datasets[0][0]
    fig, axes = plt.subplots(3, 3, figsize=(16, 9), sharex=True)
    mode_title = ("SE_2(3) Lie-group error with 3-sigma bounds"
                  if args.mode == "error3sigma"
                  else "SE_2(3) Lie-group 1-sigma over time")
    fig.suptitle(f"{mode_title} — {title}", fontsize=14)
    fill_alpha = 0.06 if n_datasets <= 2 else 0.04

    for i, (err_col, sig_col, label, unit, to_deg) in enumerate(grid):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        max_sig = 0.0
        max_err = 0.0
        for ds in datasets:
            ds_label, df, color = ds
            n = len(df)
            t = df["t"].values if "t" in df.columns else np.arange(n) * 0.1
            err = df[err_col].values
            sig3 = df[sig_col].values
            if to_deg:
                err = rad2deg(err)
                sig3 = rad2deg(sig3)
            max_sig = max(max_sig, float(np.nanmax(np.abs(sig3))))
            max_err = max(max_err, float(np.nanmax(np.abs(err))))
            slabel = f" ({ds_label})" if n_datasets > 1 else ""
            if args.mode == "error3sigma":
                ax.plot(t, err, color=color, linewidth=0.9,
                        linestyle=ERR_LINESTYLE, alpha=0.9,
                        label=f"Error{slabel}")
                ax.plot(t, sig3, color=color, dashes=SIG_DASHES,
                        linewidth=0.9, alpha=0.9)
                ax.plot(t, -sig3, color=color, dashes=SIG_DASHES,
                        linewidth=0.9, alpha=0.9)
                ax.fill_between(t, -sig3, sig3, color=color, alpha=fill_alpha)
            else:  # sigma1
                sig1 = np.abs(sig3) / 3.0
                ax.plot(t, sig1, color=color, dashes=SIG_DASHES,
                        linewidth=1.0, alpha=0.95)
        if args.mode == "error3sigma" and args.ylim in ("sigma", "error"):
            ref = max_sig if args.ylim == "sigma" else max_err
            if ref > 0 and np.isfinite(ref):
                pad = 1.1 * ref
                ax.set_ylim(-pad, pad)
        elif args.mode == "sigma1":
            if args.ylim in ("sigma", "error") and max_sig > 0:
                ax.set_ylim(0.0, 1.1 * max_sig / 3.0)
        ax.set_ylabel(f"{label} [{unit}]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    handles = []
    for ds in datasets:
        ds_label, _, color = ds
        if args.mode == "error3sigma":
            handles.append(Line2D([0], [0], color=color, linewidth=1.4,
                                  linestyle=ERR_LINESTYLE,
                                  label=f"Error ({ds_label})"))
            handles.append(Line2D([0], [0], color=color, dashes=SIG_DASHES,
                                  linewidth=1.4,
                                  label=f"+/-3sigma ({ds_label})"))
        else:
            handles.append(Line2D([0], [0], color=color, dashes=SIG_DASHES,
                                  linewidth=1.4, label=f"1sigma ({ds_label})"))
    fig.legend(handles=handles, loc="upper right", fontsize=7,
               ncol=min(n_datasets, 4), framealpha=0.9)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_name = os.path.join(args.fig_dir, f"lie_group_error_3sigma{suffix}.png")
    plt.savefig(out_name, dpi=150)
    print(f"Saved to {out_name}")


def plot_lie_group_norm(datasets, args, suffix=""):
    """3x1 norm view of the SE_2(3) Lie-group state. For position, velocity,
    and attitude, each subplot plots — for all four (preint, bias)
    combinations — the estimation-error norm ||e(t)|| (solid) and its 3-sigma
    magnitude (dashed) on a shared axis. The 3-sigma magnitude is
    sqrt(sum_axes (3 sigma_axis)^2) = 3*sqrt(trace(P_block)), i.e. the norm of
    the per-axis 3-sigma. Biases are excluded (not part of the manifold)."""
    n_datasets = len(datasets)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(r"Lie group error norm with 3$\sigma$ magnitude",
                 fontsize=13)
    fill_alpha = 0.05 if n_datasets <= 2 else 0.03

    for row, (name, err_cols, sig_cols, unit, to_deg) in enumerate(LIE_GROUPS):
        ax = axes[row]
        for ds_label, df, color in datasets:
            n = len(df)
            t = df["t"].values if "t" in df.columns else np.arange(n) * 0.1
            e_norm = _group_mag(df, err_cols, to_deg)      # ||e||
            sig3_mag = _group_mag(df, sig_cols, to_deg)    # ||3 sigma||
            ax.plot(t, e_norm, color=color, linewidth=1.1,
                    linestyle="solid", alpha=0.9)
            ax.plot(t, sig3_mag, color=color, dashes=SIG_DASHES,
                    linewidth=1.1, alpha=0.9)
            ax.fill_between(t, 0.0, sig3_mag, color=color, alpha=fill_alpha)
        ax.set_ylabel(f"{name} [{unit}]", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("Time [s]", fontsize=10)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, linewidth=1.6, label=lbl)
               for lbl, _, c in datasets]
    handles.append(Line2D([0], [0], color="k", linewidth=1.6,
                          linestyle="solid", label="error norm"))
    handles.append(Line2D([0], [0], color="k", dashes=SIG_DASHES,
                          linewidth=1.6, label="3-sigma magnitude"))
    fig.legend(handles=handles, loc="upper right", fontsize=8,
               ncol=min(n_datasets + 2, 3), framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_name = os.path.join(args.fig_dir, f"lie_group_norm{suffix}.png")
    plt.savefig(out_name, dpi=150)
    print(f"Saved to {out_name}")


def plot_lie_group_by_group(datasets, args):
    """One figure per Lie-group block (position, velocity, attitude): the
    error norm (solid) and 3-sigma magnitude (dashed) for all combinations,
    so each block can be inspected on its own. Saved as lie_group_<block>.png."""
    from matplotlib.lines import Line2D
    n_datasets = len(datasets)
    fill_alpha = 0.05 if n_datasets <= 2 else 0.03
    for name, err_cols, sig_cols, unit, to_deg in LIE_GROUPS:
        fig, ax = plt.subplots(figsize=(11, 5))
        for ds_label, df, color in datasets:
            n = len(df)
            t = df["t"].values if "t" in df.columns else np.arange(n) * 0.1
            ax.plot(t, _group_mag(df, err_cols, to_deg), color=color,
                    linewidth=1.3, linestyle="solid", alpha=0.9)
            sig3_mag = _group_mag(df, sig_cols, to_deg)
            ax.plot(t, sig3_mag, color=color, dashes=SIG_DASHES,
                    linewidth=1.3, alpha=0.9)
            ax.fill_between(t, 0.0, sig3_mag, color=color, alpha=fill_alpha)
        ax.set_title(rf"{name} error norm with 3$\sigma$ magnitude")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{name} [{unit}]")
        ax.grid(True, alpha=0.3)
        handles = [Line2D([0], [0], color=c, linewidth=1.6, label=lbl)
                   for lbl, _, c in datasets]
        handles.append(Line2D([0], [0], color="k", linewidth=1.6,
                              linestyle="solid", label="error norm"))
        handles.append(Line2D([0], [0], color="k", dashes=SIG_DASHES,
                              linewidth=1.6, label="3-sigma magnitude"))
        ax.legend(handles=handles, fontsize=8, ncol=2, framealpha=0.9)
        plt.tight_layout()
        out_name = os.path.join(args.fig_dir, f"lie_group_{name.lower()}.png")
        plt.savefig(out_name, dpi=150)
        print(f"Saved to {out_name}")


def plot_box(combos, args):
    """Box plot comparing the four combinations: one panel per Lie-group block
    (position, velocity, attitude), each showing the distribution of the
    normalised error ||e|| / 3-sigma as a box per combination. Samples are
    POOLED over all Monte Carlo runs (concatenated over runs and time), so the
    spread reflects both time- and run-to-run variability. Outliers are shown;
    the y-axis follows --box-yscale; the red line at 1 marks the 3-sigma bound
    (values below 1 are within 3-sigma).

    `combos` is a list of (label, [run dataframes], color)."""
    n_runs = max((len(dfs) for _lbl, dfs, _c in combos), default=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, err_cols, sig_cols, unit, to_deg) in zip(axes, LIE_GROUPS):
        data = []
        for _lbl, dfs, _c in combos:
            samples = []
            for df in dfs:
                e = _group_mag(df, err_cols, to_deg)
                s = _group_mag(df, sig_cols, to_deg)  # 3-sigma magnitude
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = e / s
                samples.append(r[np.isfinite(r) & (r > 0)])
            pooled = np.concatenate(samples) if samples else np.array([])
            cap = args.max_plot_points
            if cap and len(pooled) > cap:  # bound flier count for the renderer
                pooled = pooled[np.linspace(0, len(pooled) - 1, cap).astype(int)]
            # An all-zero error block (e.g. no truth for that state) filters to
            # empty; use NaN so the box is simply omitted instead of erroring.
            data.append(pooled if pooled.size else np.array([np.nan]))
        colors = [c for _lbl, _d, c in combos]
        labels = [lbl for lbl, _d, _c in combos]
        bp = ax.boxplot(data, patch_artist=True, showfliers=True,
                        flierprops=dict(marker=".", markersize=3, alpha=0.3))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        for med in bp["medians"]:
            med.set_color("black")
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_yscale(args.box_yscale)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(name)
        ax.set_ylabel(r"$\|e\| / 3\sigma$", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3, which="both")
    pooled = f" pooled over N={n_runs} runs" if n_runs > 1 else ""
    fig.suptitle(r"Normalised error $\|e\|/3\sigma$ by combination"
                 + pooled + r" (red line = 3$\sigma$ bound)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_name = os.path.join(args.fig_dir, "lie_group_boxplot.png")
    plt.savefig(out_name, dpi=150)
    print(f"Saved to {out_name}")


if __name__ == "__main__":
    main()
