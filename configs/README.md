# Configs

Release-runnable configs:

- `quick_source.json`: shared quick-data artifact paths;
- `baseline.json`: content-only standard RQ-VAE baseline;
- `optimization1.json`: error-aware dead-code revival;
- `optimized_seed43.json`: selected compact-SID + user-token + beam20 candidate.

`ablations/` preserves the raw experiment configs used during tuning. Those files are archival and may reference the original experiment directory structure; use the four release configs above for reproduction.
