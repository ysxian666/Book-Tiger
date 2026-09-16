#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
MAX_USERS="${MAX_USERS:-0}"
CONFIG="configs/optimized_seed43.json"
OUT="runs/optimized_seed43"
"$PYTHON" code/scripts/cloud_gpu/link_shared_artifacts.py --source-config configs/quick_source.json --target-config "$CONFIG" --copy
if [[ "${1:-}" == "--train-from-scratch" ]]; then
  "$PYTHON" code/scripts/build_semantic_ids.py --config "$CONFIG" --minimize-conflict-vocab
  "$PYTHON" code/scripts/train_generator.py --config "$CONFIG"
else
  mkdir -p "$OUT"
  cp checkpoints/optimized_seed43/rqvae.pt "$OUT/rqvae.pt"
  cp checkpoints/optimized_seed43/semantic_ids.json "$OUT/semantic_ids.json"
  cp checkpoints/optimized_seed43/sid_token_space.json "$OUT/sid_token_space.json"
  cp checkpoints/optimized_seed43/tiger_generator.pt "$OUT/tiger_generator.pt"
  cp checkpoints/optimized_seed43/semantic_id_report.json "$OUT/semantic_id_report.json"
fi
"$PYTHON" code/scripts/evaluate.py --config "$CONFIG" --method tiger_trie --max-users "$MAX_USERS"
"$PYTHON" code/scripts/evaluate.py --config "$CONFIG" --method hybrid --traditional-for-hybrid itemcf --hybrid-tiger-weight 0.1 --max-users "$MAX_USERS"
