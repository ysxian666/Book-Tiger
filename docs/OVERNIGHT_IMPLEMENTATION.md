# Overnight implementation map

| Requirement | Implementation | Evidence output |
|---|---|---|
| Full-stream Books scan, no first-N truncation | `code/tiger_overnight/data.py` | `user_sampling_stats.json` |
| Deterministic 120K-150K user sample | `BooksOvernightPreprocessor._select_user_sample` | `sample_config.json` |
| Train-only top-50K item cap | `_apply_item_cap` | `item_cap_stats.json` |
| Train/valid/test leakage prevention | `_split_tables`, train-only feature builders | `split_stats.json`, feature reports |
| Metadata coverage >= 0.9 audit | `_create_catalog` | `metadata_audit.json` |
| 3x256/latent32 RQ-VAE | `rqvae_train.py`, `configs/overnight_5090.json` | `rqvae/training.json`, `utilization.json` |
| Baseline suffix SID | `semantic_ids.build_baseline_sids` | `sid/baseline/*` |
| BC-GSID global assignment | `semantic_ids.build_bcgsid_sids` | `sid/bcgsid/*` |
| Long/short UCG soft prefix | `generator_model.UserContextEncoder` | `generators/ucg.pt`, `bcgsid_ucg.pt` |
| Ranking-aware Joint | `generator_train.train_generator` (`o5_joint`) | `generators/joint.pt`, training JSON |
| Standard and trie beam | `decoding.beam_search` | `eval/o*.json` |
| 500-user exhaustive ranking | `decoding.exhaustive_rank` | `eval/o*.json` |
| Fixed comparison table and decision thresholds | `report.write_comparison` | `comparison.csv`, `overnight_report.md` |
| RTX 5090 preflight | `scripts/check_env.py` | console JSON |
| Cache cleanup | `scripts/cleanup_temp.py` | console list |

## Explicit overnight fallback

The exact hierarchical min-cost-flow solver is not run for 50K items. The pipeline uses the document-approved fallback `greedy_behavior_aware`, and records the reason in `solver_report.json`. This is intentional and not presented as the exact solver.

## Claims boundary

The default run is single-seed and does not train shuffle-control generators. It can only produce directional screening evidence. It cannot support a final paper claim of improvement over standard TIGER.