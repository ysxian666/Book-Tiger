#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash runners/run_baseline.sh
bash runners/run_optimization1.sh
bash runners/run_optimized.sh
