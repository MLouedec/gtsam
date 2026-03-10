#!/usr/bin/env bash
# Sweep attitude-sigma x compass-lag-ticks, run ifac_wc_2026 for each
# combination, then drive the Python analyzer. Produces a summary TSV with
# yaw RMSE per run.
#
# Usage (from the worktree root, after a clean build):
#   ./programs/scripts/sweep_yaw_zigzag.sh [BIN] [OUT_DIR]
set -euo pipefail

BIN=${1:-_build/programs/ifac_wc_2026}
BASE_OUT=${2:-$HOME/ws/ntnu/parnav/parnav-scripts/post_processing/yaw_sweep}
TRUTH=${TRUTH:-$HOME/ws/ntnu/parnav/parnav-scripts/post_processing/df_flat_truth.csv}

mkdir -p "$BASE_OUT"
SUMMARY="$BASE_OUT/summary.tsv"
echo -e "att_sigma\tlag_ticks\tyaw_rmse_deg\ttop_fft_peak_hz" > "$SUMMARY"

ATT_SIGMAS=(0.02 0.05 0.1 0.2)
LAGS=(-40 -20 -10 -5 0 5 10 20 40)

for sig in "${ATT_SIGMAS[@]}"; do
  for lag in "${LAGS[@]}"; do
    tag="sig${sig}_lag${lag}"
    run_dir="$BASE_OUT/$tag"
    mkdir -p "$run_dir"
    echo "== running $tag =="
    "$BIN" \
      --preint se3 --bias cb --handover none \
      --attitude-sigma "$sig" --compass-lag-ticks "$lag" \
      --output-dir "$run_dir/" --output-prefix "" \
      --debug-log "$run_dir/debug.csv" >"$run_dir/stdout.log" 2>&1 || {
        echo "  FAILED (see $run_dir/stdout.log)"; continue; }

    # Analyzer writes plots + prints RMSE + FFT peak. Parse its output.
    analyzer_log="$run_dir/analyzer.log"
    python3 "$(dirname "$0")/analyze_yaw_zigzag.py" \
      --debug "$run_dir/debug.csv" \
      --truth "$TRUTH" \
      --out-dir "$run_dir/plots" >"$analyzer_log" 2>&1 || {
        echo "  ANALYZER FAILED (see $analyzer_log)"; continue; }

    rmse=$(grep -E "^Yaw RMSE:" "$analyzer_log" | awk '{print $3}')
    peak=$(grep -E "^Top-5 yaw-error FFT peaks" "$analyzer_log" \
           | sed -E 's/.*\[([0-9eE+\.-]+).*/\1/')
    echo -e "$sig\t$lag\t${rmse:-NaN}\t${peak:-NaN}" >> "$SUMMARY"
    printf "  rmse=%s deg  peak=%s Hz\n" "${rmse:-?}" "${peak:-?}"
  done
done

echo
echo "== sweep complete =="
column -t -s $'\t' "$SUMMARY"
echo "Summary: $SUMMARY"
