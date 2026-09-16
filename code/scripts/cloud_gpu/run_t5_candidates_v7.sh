#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CONFIGS=(
  "configs/books_tune_v7_04_k256_l4.json"
  "configs/books_tune_v7_07_k512_l8.json"
  "configs/books_tune_v7_09_l5_k256_l8.json"
  "configs/books_tune_v7_12_l5_k128_l8.json"
)
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="${PYTHON:-/root/miniconda3/bin/python}"; fi
"$PYTHON" scripts/cloud_gpu/check_env.py --config configs/books_quick_base.json --require-cuda
for config in "${CONFIGS[@]}"; do
 stem="$(basename "$config" .json)"
 log="logs/t5_v7_${stem}_${STAMP}.log"
 echo "=== T5 candidate $config ==="
 if ! "$PYTHON" scripts/train_generator.py --config "$config" 2>&1 | tee -a "$log"; then
  echo "generator failed: $config" | tee -a "$log"; continue
 fi
 for method in tiger_standard tiger_trie; do
  "$PYTHON" scripts/evaluate.py --config "$config" --method "$method" 2>&1 | tee -a "$log"
 done
done
"$PYTHON" scripts/summarize_tiger_candidates.py --configs "${CONFIGS[@]}" --output "logs/t5_v7_summary_${STAMP}.json"
echo "t5 v7 summary=logs/t5_v7_summary_${STAMP}.json"
