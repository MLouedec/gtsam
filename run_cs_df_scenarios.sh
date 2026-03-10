#!/usr/bin/env bash
# Run the multirotor CS/DF fixed-lag smoother through the seven IFAC fusion
# scenarios. Outputs land in $OUTPUT_DIR (default
# /Users/ghms/ws/ntnu/parnav/ieee_fusion_cs_df_results_root_new/), filenames
# carry the same `<aiding>_<robust>_<handover>_` prefix the binary chooses.
#
# Usage: ./run_cs_df_scenarios.sh [path/to/run_multirotor_cs_df]
set -euo pipefail

BIN="${1:-/Users/ghms/ws/gtsam/_build/programs/run_multirotor_cs_df_known_baro}"
if [[ ! -x "$BIN" ]]; then
    echo "Binary not found: $BIN" >&2
    echo "Build it first: cmake --build _build --target run_multirotor_cs_df_known_baro" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-/Users/ghms/ws/ntnu/parnav/ieee_fusion_cs_df_results_root_new/}"
mkdir -p "$OUTPUT_DIR"

run() {
    local tag="$1"
    shift
    echo
    echo "=========================================================="
    echo "Scenario: $tag"
    echo "  $BIN $*"
    echo "=========================================================="
    "$BIN" --output-dir "$OUTPUT_DIR" "$@"
}

# 1. No handover — pure RTK GNSS the whole run.
run "no_handover" --handover none --robust none
#
## 2. BLE DF + CS, no robust.
run "ble_df_cs_no_robust" --handover angle-range --robust none
#
## 3. BLE DF + CS, Geman-McClure.
run "ble_df_cs_gmc" --handover angle-range --robust gm
#
# 4. UWB multilateration, no robust.
run "uwb_no_robust" --handover uwb --robust none
#
# 5. UWB multilateration, Geman-McClure.
run "uwb_gmc" --handover uwb --robust gm

BARO_ORIGIN_MSL="${BARO_ORIGIN_MSL:-98.52}"

# 6. BLE DF + CS + barometer, no robust.
run "ble_df_cs_baro_no_robust" --handover angle-baro --robust none --baro-origin-msl "$BARO_ORIGIN_MSL" --ground-temp-c -8

# 7. BLE DF + CS + barometer, Geman-McClure.
run "ble_df_cs_baro_gmc" --handover angle-baro --robust gm --baro-origin-msl "$BARO_ORIGIN_MSL" --ground-temp-c -8

echo
echo "All scenarios written to $OUTPUT_DIR"
