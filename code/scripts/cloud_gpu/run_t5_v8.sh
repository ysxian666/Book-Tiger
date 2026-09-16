#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CONFIGS=(
  "configs/books_tune_v8_04_k256_l4_50e.json"
  "configs/books_tune_v8_09_l5_k256_l8_50e.json"
)
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="${PYTHON:-/root/miniconda3/bin/python}"; fi
"$PYTHON" scripts/cloud_gpu/check_env.py --config configs/books_quick_base.json --require-cuda
for config in "${CONFIGS[@]}"; do
 stem="$(basename "$config" .json)"
 log="logs/t5_v8_${stem}_${STAMP}.log"
 echo "=== T5-50e $config ===" | tee -a "$log"
 "$PYTHON" scripts/train_generator.py --config "$config" 2>&1 | tee -a "$log"
 "$PYTHON" scripts/evaluate.py --config "$config" --method tiger_trie --max-users 300 2>&1 | tee -a "$log"
 output_dir="$("$PYTHON" -c 'from tiger_rec.config import load_config; import sys; print(load_config(sys.argv[1]).paths.out)' "$config")"
 cp "${output_dir}/metrics/tiger_trie_test.json" "${output_dir}/metrics/tiger_trie_test_v8_50e.json"
done
"$PYTHON" scripts/summarize_tiger_candidates.py --configs "${CONFIGS[@]}" --metrics-suffix _v8_50e --output "logs/t5_v8_summary_${STAMP}.json"
echo "t5 v8 summary=logs/t5_v8_summary_${STAMP}.json"
