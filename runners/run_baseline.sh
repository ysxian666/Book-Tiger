#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
"$PYTHON" code/scripts/cloud_gpu/link_shared_artifacts.py --source-config configs/quick_source.json --target-config configs/baseline.json --copy
"$PYTHON" code/scripts/build_semantic_ids.py --config configs/baseline.json --minimize-conflict-vocab
"$PYTHON" code/scripts/train_generator.py --config configs/baseline.json
"$PYTHON" code/scripts/evaluate.py --config configs/baseline.json --method tiger_standard
"$PYTHON" code/scripts/evaluate.py --config configs/baseline.json --method tiger_trie
