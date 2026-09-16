#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BASE_CONFIG="configs/books_quick_base.json"
CONFIGS=(
  "configs/books_tune_v5_00_h_4_k64_l32_errreset.json"
  "configs/books_tune_v5_01_g_4_k128_l16_errreset.json"
  "configs/books_tune_v5_02_h_6_k32_l16_errreset.json"
  "configs/books_tune_v5_03_h_5_k32_l16_errreset.json"
  "configs/books_tune_v5_04_h_6_k64_l32_errreset.json"
  "configs/books_tune_v5_05_h_6_k32_l16_jc00001.json"
  "configs/books_tune_v5_06_h_6_k32_l16_jc0001.json"
  "configs/books_tune_v5_07_h_6_k32_l16_jc0005.json"
  "configs/books_tune_v5_08_h_6_k32_l16_jc001.json"
  "configs/books_tune_v5_09_h_6_k32_l16_jc005.json"
  "configs/books_tune_v5_10_h_6_k32_l16_reset5.json"
  "configs/books_tune_v5_11_h_6_k32_l16_reset20.json"
  "configs/books_tune_v5_12_h_6_k32_l16_thr1.json"
  "configs/books_tune_v5_13_h_6_k64_l32_jc0001.json"
  "configs/books_tune_v5_14_h_5_k32_l16_jc0001.json"
  "configs/books_tune_v5_15_h_5_k64_l32_jc0001.json"
)
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
fi

"$PYTHON" scripts/cloud_gpu/check_env.py --config "$BASE_CONFIG" --require-cuda

MANIFEST="logs/codebook_v5_manifest_${STAMP}.json"
"$PYTHON" - "$MANIFEST" "${CONFIGS[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path
rows = []
for config_name in sys.argv[2:]:
    path = Path(config_name)
    rows.append({
        'config': config_name,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'size_bytes': path.stat().st_size,
    })
Path(sys.argv[1]).write_text(json.dumps({'configs': rows}, indent=2), encoding='utf-8')
PY

for config in "${CONFIGS[@]}"; do
  stem="$(basename "$config" .json)"
  echo "=== tuning v5 $config ==="
  "$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py \
    --source-config "$BASE_CONFIG" --target-config "$config"
  timing_file="logs/codebook_v5_${stem}_${STAMP}.json"
  log_file="logs/codebook_v5_${stem}_${STAMP}.log"
  if ! "$PYTHON" scripts/run_pipeline.py \
      --config "$config" --stages sid --timing-file "$timing_file" \
      2>&1 | tee -a "$log_file"; then
    echo "tuning candidate failed: $config" | tee -a "$log_file"
  fi
done

"$PYTHON" scripts/summarize_codebook_tuning.py \
  --configs "${CONFIGS[@]}" \
  --output "logs/codebook_v5_summary_${STAMP}.json"

echo "codebook v5 manifest=logs/codebook_v5_manifest_${STAMP}.json"
echo "codebook v5 summary=logs/codebook_v5_summary_${STAMP}.json"
