#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${1:-smoke}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

case "$MODE" in
  smoke)
    BASE_CONFIG="configs/books_levels_base_smoke.json"
    CONFIG3="configs/books_levels3_smoke.json"
    CONFIG4="configs/books_levels4_smoke.json"
    ;;
  full)
    BASE_CONFIG="configs/books_levels_base.json"
    CONFIG3="configs/books_levels3.json"
    CONFIG4="configs/books_levels4.json"
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

COMPARISON="logs/level_3_vs_4_${MODE}_${STAMP}.json"
echo "mode=$MODE"
echo "base_config=$BASE_CONFIG"
echo "3_level_config=$CONFIG3"
echo "4_level_config=$CONFIG4"
"$PYTHON" scripts/cloud_gpu/check_env.py --config "$CONFIG3" --require-cuda

if [[ "${SKIP_BASE:-0}" != "1" ]]; then
  echo "=== preparing shared data and embeddings once ==="
  "$PYTHON" scripts/run_pipeline.py     --config "$BASE_CONFIG"     --stages prepare encode     --timing-file "logs/level_base_${MODE}_${STAMP}.json"     2>&1 | tee -a "logs/level_base_${MODE}_${STAMP}.log"
else
  echo "SKIP_BASE=1, reusing existing base artifacts"
fi

echo "=== linking shared artifacts into 3-level and 4-level output directories ==="
"$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py   --source-config "$BASE_CONFIG" --target-config "$CONFIG3"
"$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py   --source-config "$BASE_CONFIG" --target-config "$CONFIG4"

run_level() {
  local levels="$1"
  local config="$2"
  echo "=== running ${levels}-level RQ-VAE + TIGER ==="
  "$PYTHON" scripts/run_pipeline.py     --config "$config"     --stages sid generator evaluate     --methods tiger_trie     2>&1 | tee -a "logs/level_${levels}_${MODE}_${STAMP}.log"
}

run_level 3 "$CONFIG3"
run_level 4 "$CONFIG4"

"$PYTHON" scripts/compare_level_runs.py   --config3 "$CONFIG3"   --config4 "$CONFIG4"   --min-utilization 0.8   --min-recall-gain 0.002   --max-time-ratio 1.5   --output "$COMPARISON"

echo "comparison_json=$COMPARISON"
