#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CONFIGS=(
  "configs/books_tune_v9_04_k256_l4_vocab1.json"
  "configs/books_tune_v9_09_l5_k256_l8_vocab1.json"
)
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="${PYTHON:-/root/miniconda3/bin/python}"; fi
"$PYTHON" scripts/cloud_gpu/check_env.py --config configs/books_quick_base.json --require-cuda
for config in "${CONFIGS[@]}"; do
 stem="$(basename "$config" .json)"
 log="logs/t5_v9_${stem}_${STAMP}.log"
 echo "=== T5-v9 $config ===" | tee -a "$log"
 "$PYTHON" scripts/build_semantic_ids.py --config "$config" --skip-train 2>&1 | tee -a "$log"
 "$PYTHON" scripts/train_generator.py --config "$config" 2>&1 | tee -a "$log"
 "$PYTHON" scripts/evaluate.py --config "$config" --method tiger_trie 2>&1 | tee -a "$log"
done
"$PYTHON" scripts/summarize_tiger_candidates.py --configs "${CONFIGS[@]}" --output "logs/t5_v9_summary_${STAMP}.json"
echo "t5 v9 summary=logs/t5_v9_summary_${STAMP}.json"
