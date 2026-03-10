#!/usr/bin/env bash
# Sweep GM bias correlation times (tau_gyro x tau_acc) for run_bledar and rank
# the runs by position consistency + RMSE (analyze_consistency.py).
#
# The stationary bias sigma is pinned to the STIM300 datasheet in run_bledar, so
# tau is the only bias knob here. Pick the min-RMSE run whose position 3-sigma
# coverage stays ~consistent.
#
# Usage:
#   sweep_bias_tau.sh [--aiding gnss|pars|uwb] [--preint se23|se3]
#                     [--out DIR] [--grid "300 600 1200 1850 3600"]
#                     [-- <extra run_bledar args...>]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${BIN:-$REPO/_build/programs/run_bledar}"
ANALYZE="$REPO/programs/scripts/analyze_consistency.py"

AIDING="gnss"
PREINT="se23"
OUT="/tmp/bledar_sweep"
GRID_STR="300 600 1200 1850 3600"   # same grid for gyro and acc (first pass)
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --aiding) AIDING="$2"; shift 2 ;;
    --preint) PREINT="$2"; shift 2 ;;
    --out)    OUT="$2"; shift 2 ;;
    --grid)   GRID_STR="$2"; shift 2 ;;
    --)       shift; EXTRA=("$@"); break ;;
    -h|--help) grep '^#' "$0" | grep -v '^#!' | cut -c3-; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done
read -r -a GRID <<< "$GRID_STR"
[[ "$PREINT" == se23 ]] && SUF="_se23" || SUF=""
mkdir -p "$OUT"
source "$REPO/.venv/bin/activate" 2>/dev/null || true

SUMMARY="$OUT/sweep_summary.txt"
: > "$SUMMARY"
n=$(( ${#GRID[@]} * ${#GRID[@]} )); i=0
for tg in "${GRID[@]}"; do
  for ta in "${GRID[@]}"; do
    i=$((i+1)); tag="tg${tg}_ta${ta}"
    echo ">>> [$i/$n] $tag"
    "$BIN" --preint "$PREINT" --bias gm --aiding "$AIDING" \
       --bias-tau-gyro "$tg" --bias-tau-acc "$ta" --output-dir "$OUT" \
       ${EXTRA[@]+"${EXTRA[@]}"} > "$OUT/run_$tag.log" 2>&1
    cp "$OUT/bledar_gm${SUF}.csv" "$OUT/bledar_$tag.csv"
    res=$(python "$ANALYZE" "$OUT/bledar_$tag.csv" 2>/dev/null | grep RESULT || true)
    echo "$tg $ta $res" | tee -a "$SUMMARY"
  done
done

echo
echo "=== ranked by 3D position RMSE (consistent runs first) ==="
python - "$SUMMARY" <<'PY'
import re, sys
rows=[]
for ln in open(sys.argv[1]):
    p=ln.split()
    if len(p)<3 or "RESULT" not in ln: continue
    d=dict(re.findall(r"(\w+)=([\w.]+)", ln))
    rows.append((float(p[0]), float(p[1]), float(d.get("rmse3d","nan")),
                 float(d.get("cov_min","nan")), float(d.get("nees_med","nan")),
                 d.get("verdict","?")))
# consistent (verdict) first, then by rmse3d
rows.sort(key=lambda r:(r[5]!="consistent", r[2]))
print(f"{'tau_gyro':>9}{'tau_acc':>9}{'rmse3d':>9}{'cov_min':>9}"
      f"{'nees_med':>9}  verdict")
for r in rows:
    print(f"{r[0]:9.0f}{r[1]:9.0f}{r[2]:9.3f}{r[3]:9.2f}{r[4]:9.3f}  {r[5]}")
PY
