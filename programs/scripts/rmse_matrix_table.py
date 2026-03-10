#!/usr/bin/env python3
"""Build the 5-state RMSE matrix table from a run manifest.

Reads a TSV manifest (mode, assumption, pose, bias, method, csv) written by
run_rmse_matrix.sh, computes the 3D-norm RMSE per state, and prints one
Excel-pasteable block per mode with the two IMU-generation assumptions side by
side (simple = constant-global-acc, highfid = piecewise-constant IMU).

RMSE(state) = sqrt(mean_t sum_axes err^2). Units: Att [rad], Pos [m], Vel
[m/s], AccBias [m/s^2], GyrBias [rad/s].
"""
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
    rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
    data = OrderedDict()   # mode -> [(pose,bias,method)] preserving order
    cell = {}              # (mode,pose,bias,method,assumption) -> rmse dict
    for mode, asmp, pose, bias, method, csv in rows:
        data.setdefault(mode, [])
        if (pose, bias, method) not in data[mode]:
            data[mode].append((pose, bias, method))
        cell[(mode, pose, bias, method, asmp)] = rmse(csv)

    st = list(STATES)
    hdr = ("Pose\tBias\tMethod\t"
           + "\t".join(f"simple:{s}" for s in st) + "\t"
           + "\t".join(f"highfid:{s}" for s in st))
    for mode, combos in data.items():
        print(f"\n# {mode}")
        print(hdr)
        for pose, bias, method in combos:
            s = cell.get((mode, pose, bias, method, "simple"), {})
            h = cell.get((mode, pose, bias, method, "highfid"), {})
            vs = "\t".join(f"{s.get(k, float('nan')):.3f}" for k in st)
            vh = "\t".join(f"{h.get(k, float('nan')):.3f}" for k in st)
            print(f"{pose}\t{bias.upper()}\t{method}\t{vs}\t{vh}")


if __name__ == "__main__":
    main()
