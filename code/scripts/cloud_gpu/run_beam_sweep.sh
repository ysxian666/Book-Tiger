#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="${PYTHON:-/root/miniconda3/bin/python}"; fi
mkdir -p logs
for beams in 1 3 5 10 20 50; do
  config="configs/books_improve_beam${beams}.json"
  log="logs/beam_sweep_${beams}_${STAMP}.log"
  echo "=== beam $beams ===" | tee -a "$log"
  "$PYTHON" scripts/evaluate.py --config "$config" --method tiger_trie 2>&1 | tee -a "$log"
  output_dir="$("$PYTHON" -c 'from tiger_rec.config import load_config; import sys; print(load_config(sys.argv[1]).paths.out)' "$config")"
  cp "${output_dir}/metrics/tiger_trie_test.json" "logs/beam_sweep_${beams}_${STAMP}.json"
done
"$PYTHON" - "$STAMP" <<'PY'
import glob, json, sys
from pathlib import Path
stamp=sys.argv[1]
rows=[]
for path in sorted(glob.glob(f'logs/beam_sweep_*_{stamp}.json')):
 d=json.loads(Path(path).read_text(encoding='utf-8'))
 name=Path(path).name
 beams=int(name.split('_')[2])
 rows.append({'beams':beams,'recall@20':d.get('recall@20'),'ndcg@10':d.get('ndcg@10'),'ndcg@20':d.get('ndcg@20'),'coverage':d.get('coverage'),'invalid_sid_rate':d.get('invalid_sid_rate'),'p99_ms':d.get('p99_ms'),'mean_ms':d.get('mean_ms'),'wall_time_sec':d.get('wall_time_sec')})
rows.sort(key=lambda r:r['beams'])
Path(f'logs/beam_sweep_summary_{stamp}.json').write_text(json.dumps({'rows':rows},indent=2),encoding='utf-8')
print(json.dumps({'rows':rows},indent=2))
PY
echo "beam summary=logs/beam_sweep_summary_${STAMP}.json"
