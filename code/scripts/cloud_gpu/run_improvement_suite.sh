#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BASE_CONFIG="configs/books_quick_base.json"
CONFIGS=(
  "configs/books_improve_00_content_baseline.json"
  "configs/books_improve_01_gate.json"
  "configs/books_improve_02_frequency.json"
  "configs/books_improve_03_usage.json"
  "configs/books_improve_04_hard_reset.json"
  "configs/books_improve_05_geometry.json"
  "configs/books_improve_06_collision_all.json"
  "configs/books_improve_07_min_vocab.json"
  "configs/books_improve_a1_cross_attention.json"
  "configs/books_improve_a2_frequency_strong.json"
  "configs/books_improve_a3_geometry5.json"
  "configs/books_improve_a4_kmeans_off.json"
)
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="${PYTHON:-/root/miniconda3/bin/python}"; fi
"$PYTHON" scripts/cloud_gpu/check_env.py --config "$BASE_CONFIG" --require-cuda
for config in "${CONFIGS[@]}"; do
  stem="$(basename "$config" .json)"
  log="logs/improvement_${stem}_${STAMP}.log"
  timing="logs/improvement_${stem}_${STAMP}.timing.json"
  start="$(date +%s)"
  echo "=== improvement $stem ===" | tee -a "$log"
  "$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py --source-config "$BASE_CONFIG" --target-config "$config" 2>&1 | tee -a "$log"
  if ! "$PYTHON" scripts/build_semantic_ids.py --config "$config" --minimize-conflict-vocab 2>&1 | tee -a "$log"; then
    echo "sid failed: $stem" | tee -a "$log"
    continue
  fi
  sid_end="$(date +%s)"
  if ! "$PYTHON" scripts/train_generator.py --config "$config" 2>&1 | tee -a "$log"; then
    echo "generator failed: $stem" | tee -a "$log"
    continue
  fi
  gen_end="$(date +%s)"
  if ! "$PYTHON" scripts/evaluate.py --config "$config" --method tiger_standard 2>&1 | tee -a "$log"; then
    echo "standard evaluation failed: $stem" | tee -a "$log"
  fi
  if ! "$PYTHON" scripts/evaluate.py --config "$config" --method tiger_trie 2>&1 | tee -a "$log"; then
    echo "trie evaluation failed: $stem" | tee -a "$log"
  fi
  end="$(date +%s)"
  "$PYTHON" - "$timing" "$stem" "$sid_end" "$gen_end" "$end" "$start" <<'PY'
import json, sys
from pathlib import Path
out, name, sid_end, gen_end, end, start = sys.argv[1:]
Path(out).write_text(json.dumps({
    'name': name,
    'sid_seconds': int(sid_end) - int(start),
    'generator_seconds': int(gen_end) - int(sid_end),
    'evaluation_seconds': int(end) - int(gen_end),
    'total_seconds': int(end) - int(start),
    'finished_timestamp': int(end),
}, indent=2), encoding='utf-8')
PY
done
"$PYTHON" scripts/summarize_tiger_candidates.py --configs "${CONFIGS[@]}" --output "logs/improvement_suite_summary_${STAMP}.json"
echo "improvement suite summary=logs/improvement_suite_summary_${STAMP}.json"
