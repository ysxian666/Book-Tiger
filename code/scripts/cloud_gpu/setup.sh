#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$ROOT"

if [[ "${SKIP_VENV:-0}" != "1" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

python -m pip install --upgrade pip setuptools wheel
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  python -m pip install torch --index-url "$TORCH_INDEX_URL"
fi
python -m pip install -r requirements.txt

python scripts/cloud_gpu/check_env.py --config configs/books_cloud_smoke.json
