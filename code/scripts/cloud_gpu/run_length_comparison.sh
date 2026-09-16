#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${1:-smoke}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

case "$MODE" in
  smoke)
    VARIABLE_CONFIG="configs/books_length_smoke_variable.json"
    FIXED_CONFIG="configs/books_length_smoke_fixed.json"
    ;;
  full)
    VARIABLE_CONFIG="configs/books_length_full_variable.json"
    FIXED_CONFIG="configs/books_length_full_fixed.json"
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

VARIABLE_TIMING="logs/length_variable_timing_${STAMP}.json"
FIXED_TIMING="logs/length_fixed_timing_${STAMP}.json"
COMPARISON="logs/length_comparison_${STAMP}.json"

echo "mode=$MODE"
echo "variable_config=$VARIABLE_CONFIG"
echo "fixed_config=$FIXED_CONFIG"
"$PYTHON" scripts/cloud_gpu/check_env.py --config "$VARIABLE_CONFIG" --require-cuda

COMMON_STAGES=(prepare encode sid generator evaluate)
COMMON_METHODS=(tiger_trie)

run_one() {
  local name="$1"
  local config="$2"
  local timing_file="$3"
  echo "=== running $name ==="
  "$PYTHON" scripts/run_pipeline.py     --config "$config"     --stages "${COMMON_STAGES[@]}"     --methods "${COMMON_METHODS[@]}"     --timing-file "$timing_file"     2>&1 | tee -a "logs/length_${name}_${STAMP}.log"
}

run_one variable "$VARIABLE_CONFIG" "$VARIABLE_TIMING"
run_one fixed "$FIXED_CONFIG" "$FIXED_TIMING"

"$PYTHON" scripts/compare_pipeline_timings.py   "$VARIABLE_TIMING"   "$FIXED_TIMING"   --output "$COMPARISON"
echo "comparison_json=$COMPARISON"
