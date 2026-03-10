for r in none gm tukey; do
    _build/programs/ifac_wc_2026 \
        --preint se3 --bias cb \
        --handover angle-baro --robust $r \
        --compass-lag-ticks 0 \
        --noise-scaling 33 \
        --bias-scaling 5 \
        --baro-origin-msl 99.42 \
        --ground-temp-c 19
done
