#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CONFIGS=(
  "configs/books_tune_v10_00_drop02.json"
  "configs/books_tune_v10_01_drop03.json"
  "configs/books_tune_v10_02_small.json"
  "configs/books_tune_v10_03_lr1e4.json"
  "configs/books_tune_v10_04_hist100.json"
)
STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="${PYTHON:-/root/miniconda3/bin/python}"; fi
mkdir -p logs
for config in "${CONFIGS[@]}"; do
 stem="$(basename "$config" .json)"
 echo "=== T5-v10 $stem ==="
 "$PYTHON" scripts/train_generator.py --config "$config"
 output_dir="$("$PYTHON" -c 'from tiger_rec.config import load_config; import sys; print(load_config(sys.argv[1]).paths.out)' "$config")"
 "$PYTHON" scripts/evaluate.py --config "$config" --method tiger_trie
 cp "${output_dir}/generator_training.json" "logs/t5_v10_${stem}_${STAMP}.training.json"
 cp "${output_dir}/metrics/tiger_trie_test.json" "logs/t5_v10_${stem}_${STAMP}.metrics.json"
 cp "${output_dir}/tiger_generator.pt" "logs/t5_v10_${stem}_${STAMP}.generator.pt"
done
"$PYTHON" scripts/summarize_t5_variants.py --pattern "logs/t5_v10_*_${STAMP}.metrics.json" --output "logs/t5_v10_summary_${STAMP}.json"
echo "t5 v10 summary=logs/t5_v10_summary_${STAMP}.json"
