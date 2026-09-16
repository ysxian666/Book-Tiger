#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BASE_CONFIG="configs/books_quick_base.json"
CONFIGS=(
  "configs/books_tune4_i_ema_balance.json"
  "configs/books_tune4_j_ema_small.json"
  "configs/books_tune4_k_ema_compact.json"
)
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="${PYTHON:-python3}"; fi
"$PYTHON" scripts/cloud_gpu/check_env.py --config "$BASE_CONFIG" --require-cuda
for config in "${CONFIGS[@]}"; do
  echo "=== tuning v3 $config ==="
  "$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py --source-config "$BASE_CONFIG" --target-config "$config"
  if ! "$PYTHON" scripts/run_pipeline.py --config "$config" --stages sid 2>&1 | tee -a "logs/codebook_v3_$(basename "$config" .json)_${STAMP}.log"; then
    echo "tuning candidate failed: $config"
  fi
done
"$PYTHON" scripts/summarize_codebook_tuning.py --configs "${CONFIGS[@]}" --output "logs/codebook_tuning_v3_summary_${STAMP}.json"
echo "codebook tuning v3 summary=logs/codebook_tuning_v3_summary_${STAMP}.json"
