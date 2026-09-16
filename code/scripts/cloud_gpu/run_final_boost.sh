#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="${PYTHON:-/root/miniconda3/bin/python}"; fi
BEST_CFG="configs/books_improve_07_min_vocab.json"
BEST_OUT="$("$PYTHON" -c 'from tiger_rec.config import load_config; import sys; print(load_config(sys.argv[1]).paths.out)' "$BEST_CFG")"
LOG="logs/final_boost_${STAMP}.log"
mkdir -p logs

USER_CFG="configs/books_improve_p1_user_tokens.json"
USER_OUT="$("$PYTHON" -c 'from tiger_rec.config import load_config; import sys; print(load_config(sys.argv[1]).paths.out)' "$USER_CFG")"
"$PYTHON" scripts/evaluate.py --config "$USER_CFG" --method tiger_trie 2>&1 | tee -a "$LOG"
cp "${USER_OUT}/metrics/tiger_trie_test.json" "logs/supplement_user_trie_${STAMP}.json"

for beams in 1 3 5 10 20 50; do
  config="configs/books_improve_beam${beams}.json"
  "$PYTHON" scripts/evaluate.py --config "$config" --method tiger_trie 2>&1 | tee -a "$LOG"
  cp "${BEST_OUT}/metrics/tiger_trie_test.json" "logs/beam_sweep_${beams}_${STAMP}.json"
done

"$PYTHON" scripts/evaluate.py --config "$BEST_CFG" --method popularity 2>&1 | tee -a "$LOG"
cp "${BEST_OUT}/metrics/popularity_test.json" "logs/final_popularity_${STAMP}.json"
"$PYTHON" scripts/evaluate.py --config "$BEST_CFG" --method itemcf 2>&1 | tee -a "$LOG"
cp "${BEST_OUT}/metrics/itemcf_test.json" "logs/final_itemcf_${STAMP}.json"
"$PYTHON" scripts/train_baselines.py --config configs/books_quick_base.json --models sasrec 2>&1 | tee -a "$LOG"
cp "artifacts/books_quick_base/baselines_sasrec.pt" "${BEST_OUT}/baselines_sasrec.pt"
"$PYTHON" scripts/evaluate.py --config "$BEST_CFG" --method sasrec 2>&1 | tee -a "$LOG"
cp "${BEST_OUT}/metrics/sasrec_test.json" "logs/final_sasrec_${STAMP}.json"

for traditional in popularity itemcf sasrec; do
  "$PYTHON" scripts/evaluate.py --config "$BEST_CFG" --method hybrid --traditional-for-hybrid "$traditional" 2>&1 | tee -a "$LOG"
  cp "${BEST_OUT}/metrics/hybrid_test.json" "logs/final_hybrid_${traditional}_${STAMP}.json"
done

for traditional in popularity itemcf sasrec; do
  "$PYTHON" scripts/train_ranker.py --config "$BEST_CFG" --traditional "$traditional" --max-users 300 2>&1 | tee -a "$LOG"
  cp "${BEST_OUT}/ranker.joblib" "logs/final_ranker_${traditional}_${STAMP}.joblib"
  "$PYTHON" scripts/evaluate.py --config "$BEST_CFG" --method ranked_union --traditional-for-hybrid "$traditional" 2>&1 | tee -a "$LOG"
  cp "${BEST_OUT}/metrics/ranked_union_test.json" "logs/final_ranked_union_${traditional}_${STAMP}.json"
done

echo "final boost done stamp=${STAMP}" | tee -a "$LOG"
