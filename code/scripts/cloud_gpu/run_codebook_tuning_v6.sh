#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BASE_CONFIG="configs/books_quick_base.json"
CONFIGS=(
  "configs/books_tune_v6_00_g_4_k128_l16_seed42.json"
  "configs/books_tune_v6_01_g_4_k128_l16_seed43.json"
  "configs/books_tune_v6_02_g_4_k128_l16_seed44.json"
  "configs/books_tune_v6_03_g_4_k128_l16_seed45.json"
  "configs/books_tune_v6_04_g_4_k128_l16_seed46.json"
  "configs/books_tune_v6_05_g_5_k128_l16.json"
  "configs/books_tune_v6_06_g_6_k128_l16.json"
  "configs/books_tune_v6_07_g_4_k256_l16.json"
  "configs/books_tune_v6_08_g_5_k256_l16.json"
  "configs/books_tune_v6_09_g_4_k128_l8.json"
  "configs/books_tune_v6_10_g_4_k128_l24.json"
  "configs/books_tune_v6_11_g_4_k128_l32.json"
  "configs/books_tune_v6_12_g_4_k128_l16_hard05.json"
  "configs/books_tune_v6_13_g_4_k128_l16_hard20.json"
  "configs/books_tune_v6_14_g_4_k128_l16_reset5.json"
  "configs/books_tune_v6_15_g_4_k128_l16_reset20.json"
  "configs/books_tune_v6_16_g_5_k64_l32.json"
  "configs/books_tune_v6_17_g_6_k64_l32.json"
  "configs/books_tune_v6_18_g_4_k128_l16_jc0001.json"
  "configs/books_tune_v6_19_g_4_k128_l16_jc001.json"
  "configs/books_tune_v6_20_g_4_k128_l16_jc01.json"
)
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
fi

"$PYTHON" scripts/cloud_gpu/check_env.py --config "$BASE_CONFIG" --require-cuda

MANIFEST="logs/codebook_v6_manifest_${STAMP}.json"
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
  echo "=== tuning v6 $config ==="
  "$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py \
    --source-config "$BASE_CONFIG" --target-config "$config"
  timing_file="logs/codebook_v6_${stem}_${STAMP}.json"
  log_file="logs/codebook_v6_${stem}_${STAMP}.log"
  if ! "$PYTHON" scripts/run_pipeline.py \
      --config "$config" --stages sid --timing-file "$timing_file" \
      2>&1 | tee -a "$log_file"; then
    echo "tuning candidate failed: $config" | tee -a "$log_file"
  fi
done

"$PYTHON" scripts/summarize_codebook_tuning.py \
  --configs "${CONFIGS[@]}" \
  --output "logs/codebook_v6_summary_${STAMP}.json"

echo "codebook v6 manifest=logs/codebook_v6_manifest_${STAMP}.json"
echo "codebook v6 summary=logs/codebook_v6_summary_${STAMP}.json"
