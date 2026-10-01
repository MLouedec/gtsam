#!/usr/bin/env python3
"""Build the 5-state RMSE matrix table from a run manifest.

Reads a TSV manifest (mode, group, pose, bias, method, csv) written by
run_rmse_matrix.sh, computes the 3D-norm RMSE per state, and prints one
Excel-pasteable block per mode with two column groups side by side: the SE3
legacy increment as GTSAM's native method (gtsam:) vs the piecewise
constant-body-IMU (full:). IMU is generated as the exact inverse of
ConstantBodyImu, so the 'full' group reconstructs to machine precision (~0 in
the DR block); SE23 rows are identical across groups (flag ignored there).

RMSE(state) = sqrt(mean_t sum_axes err^2). Units: Att [rad], Pos [m], Vel
[m/s], AccBias [m/s^2], GyrBias [rad/s].
"""
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

STATES = OrderedDict([
    ("Att",  ["att_err_roll", "att_err_pitch", "att_err_yaw"]),
    ("Pos",  ["pos_err_n", "pos_err_e", "pos_err_d"]),
    ("Vel",  ["vel_err_n", "vel_err_e", "vel_err_d"]),
    ("AccB", ["acc_bias_err_x", "acc_bias_err_y", "acc_bias_err_z"]),
    ("GyrB", ["gyro_bias_err_x", "gyro_bias_err_y", "gyro_bias_err_z"]),
])


def rmse(csv):
    try:
        df = pd.read_csv(csv)
    except Exception:
        return {k: float("nan") for k in STATES}
    return {k: float(np.sqrt((df[c].to_numpy() ** 2).sum(1).mean()))
            for k, c in STATES.items()}


def main():
    prec = int(os.environ.get("PREC", "3"))   # decimals in the RMSE cells
    rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
    data = OrderedDict()   # mode -> [(pose,bias,method)] preserving order
    cell = {}              # (mode,pose,bias,method,group) -> rmse dict
    seen = []              # column groups present, in first-seen order
    for mode, asmp, pose, bias, method, csv in rows:
        data.setdefault(mode, [])
        if (pose, bias, method) not in data[mode]:
            data[mode].append((pose, bias, method))
        if asmp not in seen:
            seen.append(asmp)
        cell[(mode, pose, bias, method, asmp)] = rmse(csv)

    # Only print the groups that actually ran; stable order gtsam, full, rest.
    order = [g for g in ("gtsam", "full") if g in seen]
    groups = order + [g for g in seen if g not in order]

    st = list(STATES)
    sep = "  "
    header = ["Pose", "Bias", "Method"] + [s for _ in groups for s in st]

    # Gather all rows (per mode) so column widths align across the whole table.
    sections = []  # (mode, [row-of-strings])
    for mode, combos in data.items():
        body = []
        for pose, bias, method in combos:
            row = [pose, bias.upper(), method]
            for grp in groups:
                c = cell.get((mode, pose, bias, method, grp), {})
                row += [f"{c.get(k, float('nan')):.{prec}f}" for k in st]
            body.append(row)
        sections.append((mode, body))

    w = [len(h) for h in header]
    for _, body in sections:
        for row in body:
            for i, s in enumerate(row):
                w[i] = max(w[i], len(s))

    def fmt(row):  # labels left-justified, numbers right-justified
        return sep.join(s.ljust(w[i]) if i < 3 else s.rjust(w[i])
                        for i, s in enumerate(row))

    def band():  # group name centred over its 5 state columns
        parts = [" " * w[i] for i in range(3)]
        idx = 3
        for grp in groups:
            span = sum(w[idx:idx + len(st)]) + len(sep) * (len(st) - 1)
            parts.append(grp.center(span))
            idx += len(st)
        return sep.join(parts)

    for mode, body in sections:
        print(f"\n# {mode}")
        print(band())
        print(fmt(header))
        for row in body:
            print(fmt(row))


if __name__ == "__main__":
    main()
