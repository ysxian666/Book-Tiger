#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="${PYTHON:-/root/miniconda3/bin/python}"; fi

# X1: hashed text encoder proxy baseline, same structure as content baseline.
HASH_CFG="configs/books_improve_x_text_hash.json"
HASH_LOG="logs/supplement_hash_${STAMP}.log"
"$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py --source-config configs/books_quick_base.json --target-config "$HASH_CFG" 2>&1 | tee -a "$HASH_LOG"
HASH_OUT="$("$PYTHON" -c 'from tiger_rec.config import load_config; import sys; print(load_config(sys.argv[1]).paths.out)' "$HASH_CFG")"
rm -f "${HASH_OUT}/text_embeddings.npy" "${HASH_OUT}/text_item_ids.json" "${HASH_OUT}/text_embeddings_meta.json"
"$PYTHON" scripts/encode_items.py --config "$HASH_CFG" --skip-collab 2>&1 | tee -a "$HASH_LOG"
"$PYTHON" scripts/build_semantic_ids.py --config "$HASH_CFG" --minimize-conflict-vocab 2>&1 | tee -a "$HASH_LOG"
"$PYTHON" scripts/train_generator.py --config "$HASH_CFG" 2>&1 | tee -a "$HASH_LOG"
"$PYTHON" scripts/evaluate.py --config "$HASH_CFG" --method tiger_standard 2>&1 | tee -a "$HASH_LOG"
cp "${HASH_OUT}/metrics/tiger_standard_test.json" "logs/supplement_hash_standard_${STAMP}.json"
"$PYTHON" scripts/evaluate.py --config "$HASH_CFG" --method tiger_trie 2>&1 | tee -a "$HASH_LOG"
cp "${HASH_OUT}/metrics/tiger_trie_test.json" "logs/supplement_hash_trie_${STAMP}.json"

# P1: user-token personalization on the best structural stage.
USER_CFG="configs/books_improve_p1_user_tokens.json"
USER_LOG="logs/supplement_user_tokens_${STAMP}.log"
BEST_CFG="configs/books_improve_07_min_vocab.json"
BEST_OUT="$("$PYTHON" -c 'from tiger_rec.config import load_config; import sys; print(load_config(sys.argv[1]).paths.out)' "$BEST_CFG")"
USER_OUT="$("$PYTHON" -c 'from tiger_rec.config import load_config; import sys; print(load_config(sys.argv[1]).paths.out)' "$USER_CFG")"
TIGER_CFG="$BEST_CFG"
"$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py --source-config configs/books_quick_base.json --target-config "$USER_CFG" 2>&1 | tee -a "$USER_LOG"
cp "${BEST_OUT}/rqvae.pt" "${USER_OUT}/rqvae.pt"
"$PYTHON" scripts/build_semantic_ids.py --config "$USER_CFG" --skip-train --minimize-conflict-vocab 2>&1 | tee -a "$USER_LOG"
"$PYTHON" scripts/train_generator.py --config "$USER_CFG" 2>&1 | tee -a "$USER_LOG"
"$PYTHON" scripts/evaluate.py --config "$USER_CFG" --method tiger_standard 2>&1 | tee -a "$USER_LOG"
cp "${USER_OUT}/metrics/tiger_standard_test.json" "logs/supplement_user_standard_${STAMP}.json"
"$PYTHON" scripts/evaluate.py --config "$USER_CFG" --method tiger_trie 2>&1 | tee -a "$USER_LOG"
cp "${USER_OUT}/metrics/tiger_trie_test.json" "logs/supplement_user_trie_${STAMP}.json"

# X2: collaborative-only SID ablation.
COLLAB_CFG="configs/books_improve_a5_collab_only.json"
COLLAB_LOG="logs/supplement_collab_only_${STAMP}.log"
COLLAB_OUT="$("$PYTHON" -c 'from tiger_rec.config import load_config; import sys; print(load_config(sys.argv[1]).paths.out)' "$COLLAB_CFG")"
"$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py --source-config configs/books_quick_base.json --target-config "$COLLAB_CFG" 2>&1 | tee -a "$COLLAB_LOG"
"$PYTHON" scripts/build_semantic_ids.py --config "$COLLAB_CFG" --minimize-conflict-vocab 2>&1 | tee -a "$COLLAB_LOG"
"$PYTHON" scripts/train_generator.py --config "$COLLAB_CFG" 2>&1 | tee -a "$COLLAB_LOG"
"$PYTHON" scripts/evaluate.py --config "$COLLAB_CFG" --method tiger_trie 2>&1 | tee -a "$COLLAB_LOG"
cp "${COLLAB_OUT}/metrics/tiger_trie_test.json" "logs/supplement_collab_only_trie_${STAMP}.json"

# Standalone traditional baselines under the same split.
BASELINE_LOG="logs/supplement_baselines_${STAMP}.log"
for method in popularity itemcf; do
  "$PYTHON" scripts/evaluate.py --config "$TIGER_CFG" --method "$method" 2>&1 | tee -a "$BASELINE_LOG"
  cp "${BEST_OUT}/metrics/${method}_test.json" "logs/supplement_baseline_${method}_${STAMP}.json"
done
"$PYTHON" scripts/train_baselines.py --config configs/books_quick_base.json --models sasrec 2>&1 | tee -a "$BASELINE_LOG"
cp "artifacts/books_quick_base/baselines_sasrec.pt" "${BEST_OUT}/baselines_sasrec.pt"
"$PYTHON" scripts/evaluate.py --config "$TIGER_CFG" --method sasrec 2>&1 | tee -a "$BASELINE_LOG"
cp "${BEST_OUT}/metrics/sasrec_test.json" "logs/supplement_baseline_sasrec_${STAMP}.json"

# Hybrid recall and ranker on the selected structural candidate.
HYBRID_LOG="logs/supplement_hybrid_ranker_${STAMP}.log"
for traditional in popularity itemcf; do
  "$PYTHON" scripts/evaluate.py --config "$TIGER_CFG" --method hybrid --traditional-for-hybrid "$traditional" 2>&1 | tee -a "$HYBRID_LOG"
  cp "${BEST_OUT}/metrics/hybrid_test.json" "logs/supplement_hybrid_${traditional}_${STAMP}.json"
done
"$PYTHON" scripts/evaluate.py --config "$TIGER_CFG" --method hybrid --traditional-for-hybrid sasrec 2>&1 | tee -a "$HYBRID_LOG"
cp "${BEST_OUT}/metrics/hybrid_test.json" "logs/supplement_hybrid_sasrec_${STAMP}.json"
for traditional in popularity itemcf sasrec; do
  "$PYTHON" scripts/train_ranker.py --config "$TIGER_CFG" --traditional "$traditional" --max-users 300 2>&1 | tee -a "$HYBRID_LOG"
  cp "${BEST_OUT}/ranker.joblib" "logs/supplement_ranker_${traditional}_${STAMP}.joblib"
  "$PYTHON" scripts/evaluate.py --config "$TIGER_CFG" --method ranked_union --traditional-for-hybrid "$traditional" 2>&1 | tee -a "$HYBRID_LOG"
  cp "${BEST_OUT}/metrics/ranked_union_test.json" "logs/supplement_ranked_union_${traditional}_${STAMP}.json"
done

"$PYTHON" scripts/summarize_tiger_candidates.py --configs "$HASH_CFG" "$USER_CFG" --output "logs/supplementary_summary_${STAMP}.json"
echo "supplementary summary=logs/supplementary_summary_${STAMP}.json"
