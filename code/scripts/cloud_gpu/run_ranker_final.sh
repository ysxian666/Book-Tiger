#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="${PYTHON:-/root/miniconda3/bin/python}"; fi
BEST_CFG="configs/books_improve_07_min_vocab.json"
BEST_OUT="$("$PYTHON" -c 'from tiger_rec.config import load_config; import sys; print(load_config(sys.argv[1]).paths.out)' "$BEST_CFG")"
LOG="logs/ranker_final_${STAMP}.log"
mkdir -p logs

copy_metrics() {
  local tag="$1"
  for metric in "${BEST_OUT}"/metrics/*_test.json; do
    [[ -e "$metric" ]] || continue
    cp "$metric" "logs/final_${tag}_$(basename "$metric")"
  done
}

for traditional in popularity itemcf sasrec; do
  "$PYTHON" scripts/evaluate.py --config "$BEST_CFG" --method hybrid --traditional-for-hybrid "$traditional" 2>&1 | tee -a "$LOG"
  copy_metrics "hybrid_${traditional}"
done

for traditional in popularity itemcf sasrec; do
  "$PYTHON" scripts/train_ranker.py --config "$BEST_CFG" --traditional "$traditional" --max-users 300 2>&1 | tee -a "$LOG"
  cp "${BEST_OUT}/ranker.joblib" "logs/final_ranker_${traditional}_${STAMP}.joblib"
  "$PYTHON" scripts/evaluate.py --config "$BEST_CFG" --method ranked_union --traditional-for-hybrid "$traditional" 2>&1 | tee -a "$LOG"
  copy_metrics "ranked_union_${traditional}"
done

echo "ranker final done ${STAMP}" | tee -a "$LOG"
