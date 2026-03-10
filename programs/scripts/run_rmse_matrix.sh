#!/usr/bin/env bash
# Run the orbital-sim RMSE matrix: 3 modes x 2 IMU-gen assumptions x
# {SE3,SE23} x {CB,GM} x method, and print a 5-state (Att,Pos,Vel,AccBias,
# GyrBias) RMSE table -- one block per mode, both assumptions side by side.
#
# Modes:
#   DR    : dead reckoning, no noise, no bias        (--aiding none)
#   GNSS  : GNSS aiding, no noise, no bias
#   InRun : in-run bias stability + noise + GNSS aiding
#
# Assumptions (input dataset):
#   simple  : constant-global-acc projection  (*_aided file)
#   highfid : piecewise-constant IMU / midpoint projection (*_highfid file)
#
# Requires four datasets from run_orbital_simulation.m (aiding_Hz below):
#   clean simple : simulation_data_<run>_100Hz_aided_at_<hz>Hz_cpp.csv
#   clean highfid: simulation_data_<run>_100Hz_highfid_aided_at_<hz>Hz_cpp.csv
#   noisy+biased simple : ..._noisy_biased_aided_at_<hz>Hz_cpp.csv
#   noisy+biased highfid: ..._noisy_biased_highfid_aided_at_<hz>Hz_cpp.csv
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${BIN:-$REPO/_build/programs/SimulationFixedLagSmoother}"
TABLE_PY="$REPO/programs/scripts/rmse_matrix_table.py"
OUT="${OUT:-$REPO/programs/scripts/results}"
# Figures go to the Overleaf paper folder so a matrix run refreshes the paper.
# NOTE: the plotter only writes/overwrites its own named figures + the appendix
# .tex here -- it never deletes anything else in this directory.
FIG_DIR="${FIG_DIR:-/Users/ghms/Dropbox/Apper/Overleaf/IEEE TAES Journal/figures_debug}"
YLIM="${YLIM:-half}"       # y-limit bound: half (peak est. error after midpoint,
                           # default) | err (end est. error) |
                           # end (steady-state uncertainty) | max (peak uncert.)
TEX_RELPATH="${TEX_RELPATH:-figures_debug}"  # \includegraphics path prefix
AIDING_HZ="${AIDING_HZ:-1}"
DATA_DIR="${DATA_DIR:-/Users/ghms/ws/ntnu/parnav_ins_simulator/data}"
mkdir -p "$OUT"
source "$REPO/.venv/bin/activate" 2>/dev/null || true

# --- Check required datasets exist -----------------------------------------
missing=0
for f in \
  "simulation_data_01_100Hz_aided_at_${AIDING_HZ}Hz_cpp.csv" \
  "simulation_data_01_100Hz_highfid_aided_at_${AIDING_HZ}Hz_cpp.csv" \
  "simulation_data_01_100Hz_noisy_biased_aided_at_${AIDING_HZ}Hz_cpp.csv" \
  "simulation_data_01_100Hz_noisy_biased_highfid_aided_at_${AIDING_HZ}Hz_cpp.csv"; do
  [ -f "$DATA_DIR/$f" ] || { echo "MISSING dataset: $DATA_DIR/$f" >&2; missing=1; }
done
[ "$missing" -eq 0 ] || echo "-> generate the missing files in run_orbital_simulation.m (see header)." >&2

# mode-tag | flags
modes=(
  "DR|--aiding none --no-noise --no-bias --init-from-truth"
  "GNSS|--aiding gnss --no-noise --no-bias --init-from-truth"
  "InRun|--aiding gnss --with-noise --with-bias --init-from-truth"
)
# assumption-tag | flag
asmps=(
  "simple|--imu-gen simple"
  "highfid|--imu-gen highfid"
)
# method-label | preint | bias | extra-flags
combos=(
  "GTSAM|legacy|cb|"
  "GTSAM+bias-tmpl|legacy|gm|"
  "Brossard|se23|cb|--covmethod brossard --increment simple"
  "Our-simple|se23|cb|--covmethod ours --increment simple"
  "Our-full|se23|cb|--covmethod ours --increment full"
  "VanLoan|se23|cb|--covmethod vanloan --increment full"
  "Brossard|se23|gm|--covmethod brossard --increment simple"
  "Our-simple|se23|gm|--covmethod ours --increment simple"
  "Our-full|se23|gm|--covmethod ours --increment full"
  "VanLoan|se23|gm|--covmethod vanloan --increment full"
)

MANIFEST="$OUT/manifest.tsv"; : > "$MANIFEST"
for m in "${modes[@]}"; do
  IFS='|' read -r mtag mflags <<< "$m"
  for a in "${asmps[@]}"; do
    IFS='|' read -r atag aflag <<< "$a"
    for c in "${combos[@]}"; do
      IFS='|' read -r label preint bias extra <<< "$c"
      sfx="_${mtag}_${atag}_${label}"
      if "$BIN" --preint "$preint" --bias "$bias" $mflags $aflag $extra \
           --aiding-hz "$AIDING_HZ" --output-dir "$OUT/" \
           --output-suffix "$sfx" > "$OUT/log${sfx}_${bias}_${preint}.log" 2>&1
      then :; else echo "FAIL: $mtag/$atag/$label/$preint/$bias" >&2; fi
      s23=""; pose="SE3"; [ "$preint" = se23 ] && { s23="_se23"; pose="SE23"; }
      csv="$OUT/gtsam_fork_test_${bias}${s23}${sfx}.csv"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$mtag" "$atag" "$pose" "$bias" "$label" "$csv" >> "$MANIFEST"
    done
  done
done

python "$TABLE_PY" "$MANIFEST" | tee "$OUT/rmse_matrix.tsv"

# 3-sigma error plots (PNG+PDF+SVG, saved never displayed) go to the Overleaf
# figures folder. PLOTS=0 to skip. The plotter only writes its own named files
# and figures_appendix.tex -- it deletes nothing else in FIG_DIR.
if [ "${PLOTS:-1}" = 1 ]; then
  mkdir -p "$FIG_DIR"
  python "$REPO/programs/scripts/plot_rmse_matrix_sigma.py" \
    "$MANIFEST" "$FIG_DIR" --ylim "$YLIM" --tex-relpath "$TEX_RELPATH" >&2
fi
echo >&2 "Table + per-run CSVs/logs in $OUT ; paste $OUT/rmse_matrix.tsv into Excel."
echo >&2 "Figures (png/pdf/svg) + figures_appendix.tex in $FIG_DIR."
echo >&2 "Add \\input{$TEX_RELPATH/figures_appendix.tex} to your appendix once."
