for off in -0.5 -0.25 0.0 0.25 0.5 0.75 1.0; do
    _build/programs/ifac_wc_2026 --preint se3 --bias cb --handover none \
        --compass-lag-ticks 241 --baseline-yaw-offset-deg $off \
        --debug-log _build/programs/dbg_off${off}.csv \
        --output-dir /tmp/ --output-prefix run_ >/dev/null 2>&1
    rmse=$(python programs/scripts/analyze_yaw_zigzag.py \
        --debug _build/programs/dbg_off${off}.csv \
        --truth ~/ws/ntnu/parnav/parnav-scripts/post_processing/df_flat_truth.csv \
        --out-dir /tmp/plots --yaw-rate-mask-deg-s 100 2>/dev/null |
        grep "between-mvr" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    echo "offset=$off rmse=$rmse"
done
