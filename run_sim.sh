#!/usr/bin/env bash
# Run SimulationFixedLagSmoother for one aiding scheme across all four
# (preint, bias) variants, then plot the 3-sigma error comparison.
#
#   SE3  + CB   -> gtsam_fork_test_cb.csv
#   SE3  + GM   -> gtsam_fork_test_gm.csv
#   SE23 + CB   -> gtsam_fork_test_cb_se23.csv
#   SE23 + GM   -> gtsam_fork_test_gm_se23.csv
#
# Usage:
#   ./run_sim.sh [--aiding gnss|pars|none]
#                [--method brossard|ours-simple|ours-full|vanloan]
#                [--imu-gen simple|highfid]
#                [--output-dir DIR]
#                [--rmse group|axis|all|none] [--latex-dir DIR]
#                [--with-noise|--no-noise] [--with-bias|--no-bias]
#                [--aiding-hz N] [--duration SEC] [-- <extra sim args...>]
#
# --method selects the SE_2(3) preset for the two se23 runs (the legacy runs
#   ignore it). Each preset is a (covariance method, increment model) pair:
#     brossard     = Brossard gain covariance + simple (global-acc) increment
#     ours-simple  = 4th-order series covariance    + simple increment
#     ours-full    = 4th-order series covariance    + full (body-IMU) increment
#     vanloan      = exact Van Loan covariance      + full increment
#
# --rmse selects the RMSE table granularity in the plot output:
#   group (default) = 5 substates, axis = 15 per-axis,
#   all = each group's norm + its per-axis rows, none = skip.
# --fig-dir DIR     base dir for figures (default the Overleaf figures
#                   folder). Output nests under <dir>/sim/<scenario>, where
#                   scenario = dead_reckoning (none) | gnss | pars.
# --latex-dir DIR   base dir for LaTeX tables (default the Overleaf tables
#                   folder). Also nests under sim/<scenario>.
# --box-yscale log|linear  y-axis scale for the box plot (default log).
# --mc-runs N   run N Monte Carlo runs (input simulation_data_01..NN); the
#   3-sigma plots use run 01, the RMSE is averaged over all N (default 1).
# --no-noise / --no-bias select the noiseless / biasless input CSV
#   (both noisy+biased by default).
# --aiding-hz N selects the input CSV whose filename says _aided_at_NHz,
#   where N is the aiding rate in Hz (default 10; 1 = every second).
# --duration SEC caps the run length in seconds (0 = full data). Useful for
#   bounded dead-reckoning tests, e.g. --aiding none --duration 50.
#
# Extra args after `--` are forwarded verbatim to every sim run
# (e.g. -- --input /path/to.csv).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$REPO/_build/programs/SimulationFixedLagSmoother}"
PLOTTER="$REPO/visualisations/plot_3sigma.py"

AIDING="pars"
IMU_GEN=""             # input dataset specific-force projection: simple
                       # (default) | highfid. 'highfid' selects the *_highfid
                       # dataset (midpoint-frame specific force -> physically
                       # fair simple-vs-full comparison; truth PVA unchanged, so
                       # works with any --aiding). Applies to the input for ALL
                       # four runs; the increment MODEL is set by --method.
METHOD=""              # se23 preset: brossard | ours-simple | ours-full |
                       # vanloan. Each maps to a (covmethod, increment) pair
                       # (empty -> binary defaults = brossard + simple).
                       # Applies to the two se23 runs; the legacy runs ignore it.
OUTPUT_DIR=""          # empty -> let the binary use its default dir
RMSE="group"           # RMSE table granularity: group|axis|none
FIG_DIR=""             # empty -> plotter default (Overleaf .../figures)
LATEX_DIR=""           # empty -> plotter default (Overleaf .../tables)
BOX_YSCALE=""          # empty -> plotter default (log)
MC_RUNS=1              # number of Monte Carlo runs to smooth + aggregate
BIN_ARGS=()            # forwarded to the binary (noise/bias/aiding-hz/duration)
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --aiding)     AIDING="$2"; shift 2 ;;
    --method)     METHOD="$2"; shift 2 ;;
    --imu-gen)    IMU_GEN="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --rmse)       RMSE="$2"; shift 2 ;;
    --fig-dir)    FIG_DIR="$2"; shift 2 ;;
    --latex-dir)  LATEX_DIR="$2"; shift 2 ;;
    --box-yscale) BOX_YSCALE="$2"; shift 2 ;;
    --mc-runs)    MC_RUNS="$2"; shift 2 ;;
    --duration)   BIN_ARGS+=("$1" "$2"); shift 2 ;;
    --aiding-hz)  BIN_ARGS+=("$1" "$2"); shift 2 ;;
    --with-noise|--no-noise|--with-bias|--no-bias) BIN_ARGS+=("$1"); shift ;;
    --)           shift; EXTRA=("$@"); break ;;
    -h|--help)    grep '^#' "$0" | grep -v '^#!' | cut -c3-; exit 0 ;;
    *)            echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

case "$AIDING" in gnss|pars|none) ;; *)
  echo "Invalid --aiding: $AIDING (expected gnss|pars|none)" >&2; exit 1 ;;
esac

# Map the preset to a (covmethod, increment) pair (the derivation fixes which
# increment goes with which covariance). se23 runs use it; legacy ignores it.
case "$METHOD" in
  "")           ;;  # unset -> binary defaults (brossard + simple)
  brossard)     BIN_ARGS+=(--covmethod brossard --increment simple) ;;
  ours-simple)  BIN_ARGS+=(--covmethod ours     --increment simple) ;;
  ours-full)    BIN_ARGS+=(--covmethod ours     --increment full)   ;;
  vanloan)      BIN_ARGS+=(--covmethod vanloan  --increment full)   ;;
  *) echo "Invalid --method: $METHOD" \
          "(expected brossard|ours-simple|ours-full|vanloan)" >&2; exit 1 ;;
esac

# Select the input dataset's IMU-generation assumption (forwarded to every run;
# the binary picks the matching *_highfid file).
case "$IMU_GEN" in
  ""|simple) ;;                                    # default file
  highfid)   BIN_ARGS+=(--imu-gen highfid) ;;      # *_highfid file
  *) echo "Invalid --imu-gen: $IMU_GEN (expected simple|highfid)" >&2; exit 1 ;;
esac

# Scenario subfolder for figures/tables (sim/<scenario>).
case "$AIDING" in
  none) SCENARIO=sim/dead_reckoning ;;
  gnss) SCENARIO=sim/gnss ;;
  pars) SCENARIO=sim/pars ;;
esac

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found: $BIN" >&2
  echo "Build it first: cmake --build _build --target SimulationFixedLagSmoother" >&2
  exit 1
fi

DIR_ARGS=()
[[ -n "$OUTPUT_DIR" ]] && DIR_ARGS=(--output-dir "$OUTPUT_DIR")

run() {
  local preint="$1" bias="$2" run_idx="$3"
  local run_args=()
  # With >1 Monte Carlo run, pick the run's input CSV (--run) and tag the
  # result CSV (_runNN) so the plotter can aggregate them.
  if (( MC_RUNS > 1 )); then
    run_args=(--run "$run_idx" --output-suffix "$(printf '_run%02d' "$run_idx")")
  fi
  echo
  echo "=========================================================="
  echo "aiding=$AIDING  preint=$preint  bias=$bias  run=$run_idx/$MC_RUNS"
  echo "=========================================================="
  "$BIN" --aiding "$AIDING" --preint "$preint" --bias "$bias" \
    ${run_args[@]+"${run_args[@]}"} ${BIN_ARGS[@]+"${BIN_ARGS[@]}"} \
    ${DIR_ARGS[@]+"${DIR_ARGS[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
}

for r in $(seq 1 "$MC_RUNS"); do
  run legacy cb "$r"
  run legacy gm "$r"
  run se23   cb "$r"
  run se23   gm "$r"
done

echo
echo "=== Plotting 3-sigma error comparison ==="
VENV="${VENV:-$REPO/.venv}"
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi
PLOT_ARGS=(--rmse "$RMSE")
[[ -n "$OUTPUT_DIR" ]] && PLOT_ARGS+=(--dir "$OUTPUT_DIR")
[[ -n "$FIG_DIR" ]] && PLOT_ARGS+=(--fig-dir "$FIG_DIR")
[[ -n "$LATEX_DIR" ]] && PLOT_ARGS+=(--latex-dir "$LATEX_DIR")
[[ -n "$BOX_YSCALE" ]] && PLOT_ARGS+=(--box-yscale "$BOX_YSCALE")
PLOT_ARGS+=(--mc-runs "$MC_RUNS" --scenario "$SCENARIO")
python "$PLOTTER" "${PLOT_ARGS[@]}"
