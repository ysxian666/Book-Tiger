"""Train multi-view RQ-VAE and build collision-free SIDs and a prefix trie."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiger_rec.config import load_config
from tiger_rec.semantic_id.build import build_semantic_id_artifacts, train_rqvae
from tiger_rec.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--minimize-conflict-vocab", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.sid.seed)
    if not args.skip_train:
        train_rqvae(cfg)
    report = build_semantic_id_artifacts(cfg)
    if args.minimize_conflict_vocab:
        required = max(1, int(report.get("max_suffix_used", 0)) + 1)
        if required < int(cfg.sid.max_conflict_tokens):
            cfg.sid.max_conflict_tokens = required
            print(f"[SID] rebuilt with minimal conflict vocab={required}")
            report = build_semantic_id_artifacts(cfg)
    primary = "Recall@20 scheduled after generator evaluation"
    print({"semantic_id_report": report, "metric_placeholder": primary})


if __name__ == "__main__":
    main()
