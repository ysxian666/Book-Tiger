# Optimization Changes

## Optimization 1: Error-Aware Dead-Code Revival

Baseline behavior:

- codebook samples are not reset;
- only soft usage regularization is available;
- later residual levels reuse no error-aware migration.

Optimization 1 adds:

- hard usage loss;
- full-catalog encoding at each reset interval;
- per-level residual traversal;
- dead-code replacement by high-reconstruction-error residuals;
- optional noise for replacement stability;
- epoch-level utilization and dead-code diagnostics.

Key files:

- `code/tiger_rec/semantic_id/rqvae.py`
- `code/tiger_rec/semantic_id/build.py`
- `code/tiger_rec/config.py`

## Optimization 2: Compact SID and User-Token Generation

Optimization 2 changes:

- RQ-VAE geometry from 3×256/latent16 to 4×256/latent4;
- conflict resolution from last-level-only to all-level reallocation;
- token vocabulary is rebuilt to the minimum conflict-token requirement;
- optional hashed user token is prepended to generator input sequences;
- T5 retrieval uses beam=20 for the selected final model;
- evaluation supports a low TIGER weight in ItemCF/TIGER RRF.

Key files:

- `code/tiger_rec/semantic_id/collisions.py`
- `code/tiger_rec/semantic_id/token_space.py`
- `code/tiger_rec/data/dataset.py`
- `code/tiger_rec/retrieval/tiger.py`
- `code/tiger_rec/retrieval/hybrid.py`
- `code/tiger_rec/evaluation/protocol.py`
- `code/scripts/build_semantic_ids.py`
- `code/scripts/evaluate.py`

## Configuration Interfaces

```json
{
  "hard_usage_weight": 0.2,
  "dead_code_reset": true,
  "dead_code_reset_full_data": true,
  "dead_code_reset_error_aware": true,
  "dead_code_reset_usage": "epoch",
  "conflict_reallocation_mode": "all",
  "max_conflict_tokens": 1,
  "use_user_tokens": true,
  "num_user_buckets": 512,
  "num_beams": 20
}
```

## Baseline Compatibility

The baseline config disables all optimization-only switches:

```json
{
  "fusion_type": "content_only",
  "inverse_frequency_power": 0.0,
  "usage_weight": 0.0,
  "entropy_weight": 0.0,
  "hard_usage_weight": 0.0,
  "dead_code_reset": false,
  "conflict_reallocation_mode": "last",
  "use_user_tokens": false
}
```
