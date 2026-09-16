#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
BASE_CONFIG="configs/books_quick_base.json"
CONFIGS=(
  "configs/books_tune_v4_00_h_control.json"
  "configs/books_tune_v4_01_h_latent16.json"
  "configs/books_tune_v4_02_g_latent16.json"
  "configs/books_tune_v4_03_h_cb32_lat32.json"
  "configs/books_tune_v4_04_h_cb16_lat16.json"
  "configs/books_tune_v4_05_h_reset20.json"
  "configs/books_tune_v4_06_g_reset20.json"
  "configs/books_tune_v4_07_h_thr001.json"
  "configs/books_tune_v4_08_g_thr001.json"
  "configs/books_tune_v4_09_h_hard01.json"
  "configs/books_tune_v4_10_g_hard02.json"
  "configs/books_tune_v4_11_levels5_h_cb64_lat32.json"
  "configs/books_tune_v4_12_levels6_h_cb32_lat16.json"
  "configs/books_tune_v4_13_levels5_h_cb32_lat16.json"
)
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
fi

"$PYTHON" scripts/cloud_gpu/check_env.py --config "$BASE_CONFIG" --require-cuda

MANIFEST="logs/codebook_v4_manifest_${STAMP}.json"
"$PYTHON" - "$MANIFEST" "${CONFIGS[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest_path = Path(sys.argv[1])
rows = []
for config_name in sys.argv[2:]:
    path = Path(config_name)
    rows.append({
        'config': config_name,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'size_bytes': path.stat().st_size,
    })
manifest_path.write_text(json.dumps({'configs': rows}, indent=2), encoding='utf-8')
PY

for config in "${CONFIGS[@]}"; do
  stem="$(basename "$config" .json)"
  echo "=== tuning v4 $config ==="
  "$PYTHON" scripts/cloud_gpu/link_shared_artifacts.py \
    --source-config "$BASE_CONFIG" --target-config "$config"

  timing_file="logs/codebook_v4_${stem}_${STAMP}.json"
  log_file="logs/codebook_v4_${stem}_${STAMP}.log"
  if ! "$PYTHON" scripts/run_pipeline.py \
      --config "$config" --stages sid --timing-file "$timing_file" \
      2>&1 | tee -a "$log_file"; then
    echo "tuning candidate failed: $config" | tee -a "$log_file"
  fi
done

"$PYTHON" scripts/summarize_codebook_tuning.py \
  --configs "${CONFIGS[@]}" \
  --output "logs/codebook_v4_summary_${STAMP}.json"

echo "codebook v4 manifest=logs/codebook_v4_manifest_${STAMP}.json"
echo "codebook v4 summary=logs/codebook_v4_summary_${STAMP}.json"
