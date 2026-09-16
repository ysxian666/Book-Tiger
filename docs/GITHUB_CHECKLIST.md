# GitHub Publishing Checklist

- [x] Raw 10.6 GB Books files excluded.
- [x] 4.4 GB preprocess DuckDB excluded.
- [x] Derived quick data included.
- [x] Baseline and optimized code share one source tree.
- [x] Release-runnable baseline and optimized configs included.
- [x] Selected seed43 RQ-VAE and T5 checkpoints included.
- [x] No experiment checkpoint other than the selected bundle is included.
- [x] Results, seed variance, system metrics, logs, and plots included.
- [x] SHA-256 manifest generated.
- [x] No absolute remote paths in release configs.
- [x] Git repository initialized locally.
- [ ] Add LICENSE before public release.
- [ ] Create the GitHub remote and push the initial commit.
- [ ] Optionally enable GitHub Releases or Git LFS if future checkpoint files exceed 100 MB.

Recommended repository description:

```text
Quick-data TIGER reproduction and two-stage codebook/SID optimization on Amazon Reviews 2023 Books.
```
