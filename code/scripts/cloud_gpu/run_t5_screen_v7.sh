#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CONFIGS=(
  "configs/books_tune_v7_screen_04_k256_l4.json"
  "configs/books_tune_v7_screen_07_k512_l8.json"
  "configs/books_tune_v7_screen_09_l5_k256_l8.json"
  "configs/books_tune_v7_screen_12_l5_k128_l8.json"
)
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="${PYTHON:-/root/miniconda3/bin/python}"; fi
"$PYTHON" scripts/cloud_gpu/check_env.py --config configs/books_quick_base.json --require-cuda
for config in "${CONFIGS[@]}"; do
 stem="$(basename "$config" .json)"
 log="logs/t5_v7_screen_${stem}_${STAMP}.log"
 echo "=== screening $config ===" | tee -a "$log"
 "$PYTHON" scripts/train_generator.py --config "$config" 2>&1 | tee -a "$log"
 for method in tiger_standard tiger_trie; do
  "$PYTHON" scripts/evaluate.py --config "$config" --method "$method" --max-users 300 2>&1 | tee -a "$log"
  output_dir="$("$PYTHON" -c 'from tiger_rec.config import load_config; import sys; print(load_config(sys.argv[1]).paths.out)' "$config")"
  cp "${output_dir}/metrics/${method}_test.json" "${output_dir}/metrics/${method}_test_screen.json"
 done
done
"$PYTHON" scripts/summarize_tiger_candidates.py --configs "${CONFIGS[@]}" --metrics-suffix _screen --output "logs/t5_v7_screen_summary_${STAMP}.json"
echo "t5 v7 screen summary=logs/t5_v7_screen_summary_${STAMP}.json"
