# TIGER RTX 5090 Overnight Screen

This repository implements the "RTX 5090 一晚折中执行版" from `创新方向总结.md` on Amazon Reviews 2023 Books.

## Fixed contract

- 3 semantic levels, codebook size 256, latent dimension 32.
- SID length 4 (three RQ-VAE codes + suffix token).
- 2,000 hashed user buckets, beam 20, seed 42.
- Approximately 1M final interactions, top 50K catalog items.
- O1 L2-lite, O2 BC-GSID, O3 UCG, O4 BC-GSID+UCG, O5 Joint.
- Standard beam, trie beam, 2K-3K test users, 500-user exhaustive audit.
- Valid/test interactions never enter item popularity, collaborative SVD, behavior clustering, RQ-VAE fitting, or SID assignment.

## Data

The official Books review and metadata files are expected at:

```text
data/raw/Books.jsonl.gz
data/raw/meta_Books.jsonl.gz
```

`scripts/link_or_download_data.py --strict-size` first reuses an existing local hardlink or copy and otherwise downloads from the McAuley Lab public Amazon 2023 URLs. The local files were verified against the official HTTP Content-Length values: Books is 6,216,820,644 bytes and metadata is 4,942,125,770 bytes. The preprocessing scan is full-stream; it does not use a first-N review cutoff. Repeated `user>=5`, `item>=5` filtering first produces the full 776,419-user 5-core pool required by the protocol. The configured user cap is then increased to 240,000 because the top-50K and second k-core pass reduce the retained interaction count; this is the documented "increase sampling if below 1M" fallback.

## Cloud setup

Use a CUDA PyTorch wheel compatible with the RTX 5090 (`sm_120`). Install the remaining packages:

```bash
pip install -r requirements-overnight.txt
python scripts/link_or_download_data.py --root .
python scripts/check_env.py --config configs/overnight_5090.json
```

The environment check requires CUDA, BF16, both data files, and at least 150GB free disk space. The raw files themselves do not need to be copied when the cloud machine can mount or download them.

## Run

```bash
bash runners/run_overnight_5090.sh
```

For an SSH session, launch it under `tmux` or persist stdout:

```bash
nohup bash runners/run_overnight_5090.sh > overnight.log 2>&1 &
tail -f overnight.log
```

Equivalent staged commands:

```bash
python scripts/run_overnight.py --stage preprocess
python scripts/run_overnight.py --stage features
python scripts/run_overnight.py --stage rqvae
python scripts/run_overnight.py --stage sid
python scripts/run_overnight.py --stage generators
python scripts/run_overnight.py --stage evaluate
```

`--max-steps` exists only for a deliberately tiny diagnostic run. Never use it for the reported overnight screen.

## Main artifacts

```text
results/overnight/
├── sample_config.json
├── final_data_stats.json
├── rqvae/{rqvae.pt,training.json,utilization.json}
├── sid/baseline/{semantic_ids.json,sid_report.json}
├── sid/bcgsid/{semantic_ids.json,assignment_report.json,solver_report.json}
├── generators/{l2_lite.pt,bcgsid.pt,ucg.pt,bcgsid_ucg.pt,joint.pt}
├── eval/{o1_l2_lite.json,o2_bcgsid.json,o3_ucg.json,o4_bcgsid_ucg.json,o5_joint.json,comparison.csv}
└── overnight_report.md
```

## Method implementation

- **BC-GSID** uses train-only user behavior clusters, item cluster distributions, and top code alternatives. Candidate SIDs are scored by reconstruction/semantic distance, behavior L1, cluster KL, codebook usage, and suffix cost. The exact min-cost-flow path is intentionally replaced by the documented overnight fallback `greedy_behavior_aware`; `solver_report.json` records this.
- **UCG** projects long and short user histories into four soft prefix vectors through a two-layer Transformer. The hashed user token is retained.
- **Joint** continues from O4 and adds RQ-VAE reconstruction, next-SID, ranking, context alignment, and prefix consistency losses while freezing codebooks by default.

## Interpretation rule

`overnight_report.md` labels results as an overnight screen. A positive directional signal is not a paper-level result until random/shuffle controls, multiple seeds, a larger test split, and a larger item cap are run. The default five-model matrix does not train extra shuffle-control generators, so `effective_signal` in the report intentionally remains conservative.

## Cleanup

After the GPU run, retain checkpoints and metrics but remove transient caches:

```bash
python scripts/cleanup_temp.py --root .
```

This removes `__pycache__`, `.pytest_cache`, `cache/overnight`, temporary DuckDB data, and the local debug directory. It does not remove `data/raw`, `data/overnight`, or `results/overnight`.

## Completed overnight result

The completed AutoDL RTX 5090 run used the official full review stream, performed global 5-core filtering, then sampled 240,000 users. Final data contained 1,134,600 interactions, 102,167 users, 49,765 catalog items, and metadata coverage 1.0. RQ-VAE stopped at epoch 2000 under the 40-minute fallback and reached mean codebook utilization 0.995.

| Metric | O1 L2-lite | O2 BC-GSID | O3 UCG | O4 BC-GSID+UCG | O5 Joint |
|---|---:|---:|---:|---:|---:|
| Recall@10 | 0.004412 | 0.007220 | 0.006819 | 0.010028 | 0.009226 |
| Recall@20 | 0.009627 | 0.011633 | 0.011231 | 0.015243 | 0.012836 |
| NDCG@10 | 0.001946 | 0.004309 | 0.003712 | 0.005315 | 0.004375 |
| Invalid SID rate | 0.002587 | 0.014160 | 0.000542 | 0.011412 | 0.081127 |
| Codebook utilization | 0.994792 | 0.981771 | 0.994792 | 0.981771 | 0.981771 |

O4 BC-GSID+UCG produced the strongest directional Recall@20 signal: 38 hits among 2,493 users, +58.3% relative to O1, and +28.6% on short-history users. However, BC-GSID increased semantic drift by 6.23x relative to O1, failing the strict 5% semantic-drift gate; UCG alone reduced short-history Recall@20 by 7.1%; Joint reduced Recall@20 by 15.8% relative to O4. The Wilson 95% intervals overlap, shuffle controls were not run, and this remains a single-seed screening result rather than a paper-level claim.

## Control experiments for causal attribution

Four controls were trained with the same 50K-step budget as O1-O4:

| Variant | SID | User context | Recall@10 | Recall@20 | NDCG@10 | Hits |
|---|---|---|---:|---:|---:|---:|
| C1 BC-GSID shuffled behavior | shuffled item profiles | hashed user token | 0.003209 | 0.006819 | 0.001840 | 17 |
| C2 BC-GSID hard semantic | train behavior, 1.05x semantic budget | hashed user token | 0.006017 | **0.016847** | 0.002887 | 42 |
| C3 UCG shuffled context | baseline suffix | shuffled long/short context | 0.003209 | 0.010830 | 0.001949 | 27 |
| C4 BC-hard + UCG | hard semantic | true soft-prefix context | 0.004813 | 0.008424 | 0.002253 | 21 |

The shuffled-behavior control drops Recall@20 from 1.1633% for true BC-GSID to 0.6819%, supporting the existence of behavior-specific signal. The hard-semantic control still performs 10,690 code reallocations while keeping mean semantic drift at 4.0216e-5, below both the baseline 4.0842e-5 and the 1.05x budget of 4.2884e-5; it reaches Recall@20 1.6847% and 42 hits. This indicates that BC-GSID's Recall gain can be reproduced with semantically valid reallocations. In contrast, shuffled UCG context reaches 1.0830%, only 3.6% below true UCG, and hard-semantic plus UCG drops to 0.8424%, 44.7% below the original O4. The UCG-specific and combined semantic-constrained gains are therefore not yet robust; the strongest defensible signal is hard-semantic BC-GSID alone.
