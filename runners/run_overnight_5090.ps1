$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = "$Root\code;$env:PYTHONPATH"
$env:TOKENIZERS_PARALLELISM = "false"
$env:OMP_NUM_THREADS = "8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
if (-not $env:HF_ENDPOINT) { $env:HF_ENDPOINT = "https://hf-mirror.com" }
if (-not $env:HF_HOME) { $env:HF_HOME = "$Root\.cache\huggingface" }
if (-not $env:PIP_CACHE_DIR) { $env:PIP_CACHE_DIR = "$Root\.cache\pip" }
if (-not $env:TMPDIR) { $env:TMPDIR = "$Root\.cache\tmp" }
New-Item -ItemType Directory -Force -Path $env:HF_HOME,$env:PIP_CACHE_DIR,$env:TMPDIR | Out-Null
python scripts\link_or_download_data.py --root $Root --strict-size
python scripts\check_env.py --config configs\overnight_5090.json
python scripts\run_overnight.py --config configs\overnight_5090.json --stage all
