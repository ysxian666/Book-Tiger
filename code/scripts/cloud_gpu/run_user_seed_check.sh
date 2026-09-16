#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="${PYTHON:-/root/miniconda3/bin/python}"; fi
mkdir -p logs
for seed in 42 43 44; do
  if [[ "$seed" == "42" ]]; then config="configs/books_improve_p1_user_tokens_beam20.json"; else config="configs/books_improve_p1_user_tokens_beam20_seed${seed}.json"; fi
  log="logs/user_seed_${seed}_${STAMP}.log"
  echo "=== user seed $seed ===" | tee -a "$log"
  if [[ "$seed" != "42" ]]; then
    "$PYTHON" scripts/train_generator.py --config "$config" 2>&1 | tee -a "$log"
  else
    echo "reuse existing seed42 training" | tee -a "$log"
  fi
  out="$("$PYTHON" -c 'from tiger_rec.config import load_config; import sys; print(load_config(sys.argv[1]).paths.out)' "$config")"
  cp "${out}/tiger_generator.pt" "logs/user_seed_${seed}_${STAMP}.pt"
  "$PYTHON" scripts/evaluate.py --config "$config" --method tiger_trie 2>&1 | tee -a "$log"
  cp "${out}/metrics/tiger_trie_test.json" "logs/user_seed_${seed}_beam20_${STAMP}.json"
done
echo "user seed check done ${STAMP}"
