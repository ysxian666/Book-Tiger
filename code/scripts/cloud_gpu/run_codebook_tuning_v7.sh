#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BASE_CONFIG="configs/books_quick_base.json"
CONFIGS=(
  "configs/books_tune_v7_00_k256_l8_seed42.json"
  "configs/books_tune_v7_01_k256_l8_seed43.json"
  "configs/books_tune_v7_02_k256_l8_seed44.json"
  "configs/books_tune_v7_03_k256_l8_seed45.json"
  "configs/books_tune_v7_04_k256_l4.json"
  "configs/books_tune_v7_05_k256_l6.json"
  "configs/books_tune_v7_06_k256_l12.json"
  "configs/books_tune_v7_07_k512_l8.json"
  "configs/books_tune_v7_08_k512_l16.json"
  "configs/books_tune_v7_09_l5_k256_l8.json"
  "configs/books_tune_v7_10_l5_k256_l4.json"
  "configs/books_tune_v7_11_l5_k256_l6.json"
  "configs/books_tune_v7_12_l5_k128_l8.json"
  "configs/books_tune_v7_13_l6_k128_l8.json"
  "configs/books_tune_v7_14_k128_l8_seed43.json"
  "configs/books_tune_v7_15_k128_l8_hard02.json"
  "configs/books_tune_v7_16_k256_l8_hard02.json"
  "configs/books_tune_v7_17_l5_k256_l8_hard02.json"
  "configs/books_tune_v7_18_l5_k256_l8_reset20.json"
  "configs/books_tune_v7_19_k256_l8_reset20.json"
)
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="${PYTHON:-/root/miniconda3/bin/python}"; fi
"$PYTHON" scripts/cloud_gpu/check_env.py --config "$BASE_CONFIG" --require-cuda
MANIFEST="logs/codebook_v7_manifest_${STAMP}.json"
"$PYTHON" - "$MANIFEST" "${CONFIGS[@]}" <<'PY'
import hashlib,json,sys
from pathlib import Path
rows=[]
for name in sys.argv[2:]:
 p=Path(name); rows.append({'config':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size})
Path(sys.argv[1]).write_text(json.dumps({'configs':rows},indent=2),encoding='utf-8')
PY
for config in "${CONFIGS[@]}"; do
 stem="$(basename "$config" .json)"
 echo "=== tuning v7 $config ==="
 "$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py --source-config "$BASE_CONFIG" --target-config "$config"
 timing_file="logs/codebook_v7_${stem}_${STAMP}.json"
 log_file="logs/codebook_v7_${stem}_${STAMP}.log"
 if ! "$PYTHON" scripts/run_pipeline.py --config "$config" --stages sid --timing-file "$timing_file" 2>&1 | tee -a "$log_file"; then
  echo "tuning candidate failed: $config" | tee -a "$log_file"
 fi
done
"$PYTHON" scripts/summarize_codebook_tuning.py --configs "${CONFIGS[@]}" --output "logs/codebook_v7_summary_${STAMP}.json"
echo "codebook v7 manifest=logs/codebook_v7_manifest_${STAMP}.json"
echo "codebook v7 summary=logs/codebook_v7_summary_${STAMP}.json"
