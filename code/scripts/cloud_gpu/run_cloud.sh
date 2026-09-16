#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${1:-smoke}"
if [[ -n "${CONFIG:-}" ]]; then
  CFG="$CONFIG"
else
  case "$MODE" in
    smoke)
      CFG="configs/books_cloud_smoke.json"
      ;;
    full|tiger|prepare|encode|sid|generator|baselines|ranker|evaluate)
      CFG="configs/books_cloud_gpu.json"
      ;;
    *)
      echo "Usage: $0 [smoke|prepare|encode|sid|generator|baselines|ranker|evaluate|tiger|full]" >&2
      exit 2
      ;;
  esac
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

case "$MODE" in
  smoke)
    STAGES=(prepare encode sid generator baselines ranker evaluate)
    METHODS=(popularity itemcf twotower sasrec tiger_standard tiger_trie hybrid ranked_union)
    ;;
  prepare)
    STAGES=(prepare)
    METHODS=()
    ;;
  encode)
    STAGES=(encode)
    METHODS=()
    ;;
  sid)
    STAGES=(sid)
    METHODS=()
    ;;
  generator)
    STAGES=(generator)
    METHODS=()
    ;;
  baselines)
    STAGES=(baselines)
    METHODS=()
    ;;
  ranker)
    STAGES=(ranker)
    METHODS=()
    ;;
  evaluate)
    STAGES=(evaluate)
    METHODS=(popularity itemcf twotower gru4rec sasrec bert4rec tiger_standard tiger_trie hybrid ranked_union)
    ;;
  tiger)
    STAGES=(prepare encode sid generator evaluate)
    METHODS=(tiger_standard tiger_trie)
    ;;
  full)
    STAGES=(prepare encode sid generator baselines ranker evaluate)
    METHODS=(popularity itemcf twotower gru4rec sasrec bert4rec tiger_standard tiger_trie hybrid ranked_union)
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

mkdir -p logs
LOG="logs/${MODE}_$(date +%Y%m%d_%H%M%S).log"
echo "mode=$MODE config=$CFG log=$LOG python=$PYTHON"
"$PYTHON" scripts/cloud_gpu/check_env.py --config "$CFG" --require-cuda

CMD=("$PYTHON" scripts/run_pipeline.py --config "$CFG" --stages "${STAGES[@]}")
if (( ${#METHODS[@]} > 0 )); then
  CMD+=(--methods "${METHODS[@]}")
fi
printf '+'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee -a "$LOG"
