#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE_CONFIG="configs/books_quick_base.json"
CONFIG3F="configs/books_quick3_fixed.json"
CONFIG4F="configs/books_quick4_fixed.json"
CONFIG3V="configs/books_quick3_variable.json"
CONFIG4V="configs/books_quick4_variable.json"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

echo "quick comparison stamp=$STAMP"
"$PYTHON" scripts/cloud_gpu/check_env.py --config "$CONFIG3F" --require-cuda

if [[ "${SKIP_BASE:-0}" != "1" ]]; then
  echo "=== quick shared prepare + encode ==="
  "$PYTHON" scripts/run_pipeline.py     --config "$BASE_CONFIG"     --stages prepare encode     --timing-file "logs/quick_base_${STAMP}.json"     2>&1 | tee -a "logs/quick_base_${STAMP}.log"
fi

echo "=== linking quick shared artifacts ==="
"$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py --source-config "$BASE_CONFIG" --target-config "$CONFIG3F"
"$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py --source-config "$BASE_CONFIG" --target-config "$CONFIG4F"
"$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py --source-config "$BASE_CONFIG" --target-config "$CONFIG3V"
"$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py --source-config "$BASE_CONFIG" --target-config "$CONFIG4V"

run_one() {
  local name="$1"
  local config="$2"
  echo "=== quick run: $name ==="
  "$PYTHON" scripts/run_pipeline.py     --config "$config"     --stages sid generator evaluate     --methods tiger_trie     2>&1 | tee -a "logs/quick_${name}_${STAMP}.log"
}

run_one 3_fixed "$CONFIG3F"
run_one 4_fixed "$CONFIG4F"
run_one 3_variable "$CONFIG3V"
run_one 4_variable "$CONFIG4V"

echo "=== comparing 3-level vs 4-level under fixed length ==="
"$PYTHON" scripts/compare_level_runs.py   --config3 "$CONFIG3F"   --config4 "$CONFIG4F"   --output "logs/quick_level_fixed_${STAMP}.json"

echo "=== comparing 3-level vs 4-level under variable length ==="
"$PYTHON" scripts/compare_level_runs.py \
  --config3 "$CONFIG3V" \
  --config4 "$CONFIG4V" \
  --output "logs/quick_level_variable_${STAMP}.json"

echo "=== comparing fixed vs variable at 3 levels ==="
"$PYTHON" scripts/compare_pipeline_timings.py   "artifacts/books_quick3_variable/pipeline_timing.json"   "artifacts/books_quick3_fixed/pipeline_timing.json"   --output "logs/quick_length_3_${STAMP}.json"

echo "=== comparing fixed vs variable at 4 levels ==="
"$PYTHON" scripts/compare_pipeline_timings.py   "artifacts/books_quick4_variable/pipeline_timing.json"   "artifacts/books_quick4_fixed/pipeline_timing.json"   --output "logs/quick_length_4_${STAMP}.json"

echo "quick results ready:"
echo "logs/quick_level_fixed_${STAMP}.json"
echo "logs/quick_level_variable_${STAMP}.json"
echo "logs/quick_length_3_${STAMP}.json"
echo "logs/quick_length_4_${STAMP}.json"
