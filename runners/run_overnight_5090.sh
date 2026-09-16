#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.cache/pip}"
export TMPDIR="${TMPDIR:-$ROOT/.cache/tmp}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" scripts/link_or_download_data.py --root "$ROOT" --strict-size
"$PYTHON_BIN" scripts/check_env.py --config configs/overnight_5090.json
"$PYTHON_BIN" scripts/run_overnight.py --config configs/overnight_5090.json --stage all
