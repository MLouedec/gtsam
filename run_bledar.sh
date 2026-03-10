#!/usr/bin/env bash
# Run run_bledar (multirotor BLE/DAR fusion) for one aiding source across the
# four (preint, bias) combinations, then plot with the shared plotter into the
# bledar_<aiding> figure/table folders.
#
#   SE3  + CB   -> bledar_cb.csv
#   SE3  + GM   -> bledar_gm.csv
#   SE23 + CB   -> bledar_cb_se23.csv
#   SE23 + GM   -> bledar_gm_se23.csv
#
# Usage:
#   ./run_bledar.sh [--aiding gnss|pars|uwb]
#                   [--method brossard|ours-simple|ours-full|vanloan]
#                   [--base-path DIR]
#                   [--output-dir DIR] [--rmse group|axis|all|none]
#                   [--robust none|gm|tukey] [--fig-dir DIR] [--latex-dir DIR]
#                   [-- <extra binary args...>]
#
# --method selects the SE_2(3) preset for the two se23 runs (the legacy runs
#   ignore it). Each preset is a (covariance method, increment model) pair:
#     brossard     = Brossard gain covariance + simple (global-acc) increment
#     ours-simple  = 4th-order series covariance    + simple increment
#     ours-full    = 4th-order series covariance    + full (body-IMU) increment
#     vanloan      = exact Van Loan covariance      + full increment
#
# Position error/RMSE is against the RTK truth (interpolated); velocity and
# attitude have no truth (those rows are zero). Figures go to
# <fig-dir>/bledar_<aiding>, tables to <latex-dir>/bledar_<aiding>.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$REPO/_build/programs/run_bledar}"
PLOTTER="$REPO/visualisations/plot_3sigma.py"

AIDING="gnss"
METHOD=""              # se23 preset: brossard | ours-simple | ours-full |
                      # vanloan. Maps to a (covmethod, increment) pair
                      # (empty -> binary defaults = brossard + simple).
                      # Applies to the two se23 runs; the legacy runs ignore it.
BASE_PATH=""
OUTPUT_DIR=""
RMSE="group"
ROBUST=""
FIG_DIR=""
LATEX_DIR=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --aiding)     AIDING="$2"; shift 2 ;;
    --method)     METHOD="$2"; shift 2 ;;
    --base-path)  BASE_PATH="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --rmse)       RMSE="$2"; shift 2 ;;
    --robust)     ROBUST="$2"; shift 2 ;;
    --fig-dir)    FIG_DIR="$2"; shift 2 ;;
    --latex-dir)  LATEX_DIR="$2"; shift 2 ;;
    --)           shift; EXTRA=("$@"); break ;;
    -h|--help)    grep '^#' "$0" | grep -v '^#!' | cut -c3-; exit 0 ;;
    *)            echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

case "$AIDING" in gnss|pars|uwb) ;; *)
  echo "Invalid --aiding: $AIDING (expected gnss|pars|uwb)" >&2; exit 1 ;;
esac

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found: $BIN" >&2
  echo "Build it first: cmake --build _build --target run_bledar" >&2
  exit 1
fi

BIN_ARGS=(--aiding "$AIDING")
[[ -n "$BASE_PATH" ]] && BIN_ARGS+=(--base-path "$BASE_PATH")
[[ -n "$OUTPUT_DIR" ]] && BIN_ARGS+=(--output-dir "$OUTPUT_DIR")
[[ -n "$ROBUST" ]] && BIN_ARGS+=(--robust "$ROBUST")

# Map the se23 preset to a (covmethod, increment) pair (the derivation fixes
# which increment goes with which covariance). se23 runs use it; legacy ignores.
case "$METHOD" in
  "")           ;;  # unset -> binary defaults (brossard + simple)
  brossard)     BIN_ARGS+=(--covmethod brossard --increment simple) ;;
  ours-simple)  BIN_ARGS+=(--covmethod ours     --increment simple) ;;
  ours-full)    BIN_ARGS+=(--covmethod ours     --increment full)   ;;
  vanloan)      BIN_ARGS+=(--covmethod vanloan  --increment full)   ;;
  *) echo "Invalid --method: $METHOD" \
          "(expected brossard|ours-simple|ours-full|vanloan)" >&2; exit 1 ;;
esac

run() {
  local preint="$1" bias="$2"
  echo
  echo "=========================================================="
  echo "aiding=$AIDING  preint=$preint  bias=$bias"
  echo "=========================================================="
  "$BIN" --preint "$preint" --bias "$bias" \
    "${BIN_ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
}

run se3  cb
run se3  gm
run se23 cb
run se23 gm

echo
echo "=== Plotting bledar_$AIDING ==="
VENV="${VENV:-$REPO/.venv}"
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi
PLOT_ARGS=(--rmse "$RMSE" --result-prefix bledar_ --scenario "bledar_$AIDING")
[[ -n "$OUTPUT_DIR" ]] && PLOT_ARGS+=(--dir "$OUTPUT_DIR")
[[ -n "$FIG_DIR" ]] && PLOT_ARGS+=(--fig-dir "$FIG_DIR")
[[ -n "$LATEX_DIR" ]] && PLOT_ARGS+=(--latex-dir "$LATEX_DIR")
python "$PLOTTER" "${PLOT_ARGS[@]}"
