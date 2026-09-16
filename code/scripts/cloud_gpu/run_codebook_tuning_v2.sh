#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE_CONFIG="configs/books_quick_base.json"
CONFIGS=(
  "configs/books_tune4_f_hard_usage.json"
  "configs/books_tune4_g_hard_reset.json"
  "configs/books_tune4_h_small_hard_reset.json"
)
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

"$PYTHON" scripts/cloud_gpu/check_env.py --config "$BASE_CONFIG" --require-cuda

for config in "${CONFIGS[@]}"; do
  echo "=== tuning v2 $config ==="
  "$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py     --source-config "$BASE_CONFIG" --target-config "$config"

  timing_file="logs/codebook_v2_$(basename "$config" .json)_${STAMP}.json"
  if ! "$PYTHON" scripts/run_pipeline.py       --config "$config"       --stages sid       --timing-file "$timing_file"       2>&1 | tee -a "logs/codebook_v2_$(basename "$config" .json)_${STAMP}.log"; then
    echo "tuning candidate failed: $config"
  fi
done

"$PYTHON" scripts/summarize_codebook_tuning.py   --configs "${CONFIGS[@]}"   --output "logs/codebook_tuning_v2_summary_${STAMP}.json"

echo "codebook tuning v2 summary=logs/codebook_tuning_v2_summary_${STAMP}.json"
