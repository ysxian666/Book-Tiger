# Reproduction Guide

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For cloud CUDA reproducibility:

```bash
pip install -r requirements-lock.txt
```

## Unit Tests

```bash
python -m pytest -q
```

Expected result: 19 tests passed.

## Baseline

```bash
bash runners/run_baseline.sh
```

This copies shared quick artifacts into `runs/baseline`, trains the baseline RQ-VAE and T5 generator, then evaluates standard and trie decoding.

## Optimization Point 1

```bash
bash runners/run_optimization1.sh
```

This enables hard usage and full-data error-aware dead-code reset.

## Optimization Point 2

With the included selected checkpoint:

```bash
bash runners/run_optimized.sh
```

From scratch:

```bash
bash runners/run_optimized.sh --train-from-scratch
```

The selected config uses:

```text
4 levels
K = 256
latent_dim = 4
SID length = 5
max_conflict_tokens = 1
all-level semantic reallocation
512 hashed user buckets
beam = 20
```

## Outputs

The runners write to:

```text
runs/baseline/
runs/optimization1/
runs/optimized_seed43/
```

Reference results are already committed under `results/`.

## Raw Books Data

Raw data is not committed. To regenerate the derived quick data from scratch, place:

```text
data/raw/review_Books.jsonl
data/raw/meta_Books.jsonl
```

and run:

```bash
python code/scripts/prepare_data.py --config configs/quick_source.json
python code/scripts/encode_items.py --config configs/quick_source.json
```

## Device Selection

The code chooses CUDA automatically when available. Set `device` fields in the configs to `cpu` only for smoke testing; CPU training is substantially slower.
