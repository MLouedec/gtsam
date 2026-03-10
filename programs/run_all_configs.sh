#!/usr/bin/env bash
# Run the four (preintegrator, bias) configurations of
# SimulationFixedLagSmoother for a single aiding scheme and collect the
# resulting CSVs under one shared suffix.
#
#   SE3  + CB → gtsam_fork_test_cb<suffix>.csv
#   SE3  + GM → gtsam_fork_test_gm<suffix>.csv
#   SE23 + CB → gtsam_fork_test_cb_se23<suffix>.csv
#   SE23 + GM → gtsam_fork_test_gm_se23<suffix>.csv
#
# Usage:
#   run_all_configs.sh <gnss|pars|none> [--duration N] [--suffix STR]
#                      [--output-dir DIR] [--binary PATH]
#                      [-- <extra simulator args>...]
#
# When --aiding is `none`, --duration is required and the suffix defaults to
# `_none_<N>s`. For gnss/pars the suffix defaults to empty (matches the
# pre-existing filenames picked up by visualisations/plot_3sigma.py).
set -euo pipefail

usage() {
  cat >&2 <<EOF
Usage: $(basename "$0") <gnss|pars|none> [options] [-- extra-args...]
Options:
  --duration N         Run-length cap in seconds (required for 'none').
  --suffix STR         Output filename suffix (default: ''
                       for gnss/pars, '_none_<N>s' for none).
  --output-dir DIR     Passed through to the simulator.
  --binary PATH        Simulator binary
                       (default: <repo>/_build/programs/SimulationFixedLagSmoother).
  -- extra-args...     Forwarded verbatim to every simulator invocation.
EOF
  exit 1
}

[[ $# -ge 1 ]] || usage
AIDING="$1"; shift
case "$AIDING" in gnss|pars|none) ;; *) usage ;; esac

DURATION=""
SUFFIX=""
SUFFIX_SET=0
OUTPUT_DIR=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$REPO_ROOT/_build/programs/SimulationFixedLagSmoother"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration)   DURATION="$2"; shift 2 ;;
    --suffix)     SUFFIX="$2"; SUFFIX_SET=1; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --binary)     BINARY="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

if [[ "$AIDING" == "none" ]]; then
  [[ -n "$DURATION" ]] || { echo "--duration is required when aiding=none" >&2; exit 1; }
  if [[ $SUFFIX_SET -eq 0 ]]; then SUFFIX="_none_${DURATION}s"; fi
fi

[[ -x "$BINARY" ]] || { echo "Simulator binary not found/executable: $BINARY" >&2; exit 1; }

COMMON_ARGS=(--aiding "$AIDING")
[[ -n "$DURATION" ]]   && COMMON_ARGS+=(--duration "$DURATION")
[[ -n "$SUFFIX" ]]     && COMMON_ARGS+=(--output-suffix "$SUFFIX")
[[ -n "$OUTPUT_DIR" ]] && COMMON_ARGS+=(--output-dir "$OUTPUT_DIR")
COMMON_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

# (preint, bias) — 4 configs.
CONFIGS=(
  "legacy cb"
  "legacy gm"
  "se23   cb"
  "se23   gm"
)

for cfg in "${CONFIGS[@]}"; do
  read -r PREINT BIAS <<<"$cfg"
  echo "=============================================================="
  echo " Running: preint=$PREINT bias=$BIAS aiding=$AIDING${DURATION:+ duration=${DURATION}s}"
  echo "=============================================================="
  "$BINARY" --preint "$PREINT" --bias "$BIAS" "${COMMON_ARGS[@]}"
done

echo
echo "All four configurations completed."
if [[ -n "$SUFFIX" ]]; then
  echo "Plot with:"
  echo "  python3 $REPO_ROOT/visualisations/plot_3sigma.py --suffix '$SUFFIX'"
fi
