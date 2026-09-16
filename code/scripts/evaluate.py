"""Evaluate one retrieval system on the leave-one-out test split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiger_rec.config import load_config
from tiger_rec.evaluation.protocol import evaluate_and_save
from tiger_rec.evaluation.protocol import build_group_definitions
from tiger_rec.models.ranker import LogisticRanker
from tiger_rec.retrieval.hybrid import HybridRetriever, RankedUnionRetriever
from tiger_rec.retrieval.tiger import TigerRetriever
from tiger_rec.retrieval.traditional import build_traditional_retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "popularity", "itemcf", "twotower", "gru4rec", "sasrec", "bert4rec",
            "tiger_standard", "tiger_trie", "hybrid", "ranked_union",
        ],
    )
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--traditional-for-hybrid", default="sasrec")
    parser.add_argument("--hybrid-tiger-weight", type=float, default=1.0)
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.method in {"popularity", "itemcf", "twotower", "gru4rec", "sasrec", "bert4rec"}:
        retriever = build_traditional_retriever(cfg, args.method)
    elif args.method == "tiger_standard":
        retriever = TigerRetriever(cfg, constrained=False)
    elif args.method == "tiger_trie":
        retriever = TigerRetriever(cfg, constrained=True)
    elif args.method == "hybrid":
        retriever = HybridRetriever(
            [
                build_traditional_retriever(cfg, args.traditional_for_hybrid),
                TigerRetriever(cfg, constrained=True),
            ],
            weights=[1.0, float(args.hybrid_tiger_weight)],
        )
    elif args.method == "ranked_union":
        long_tail, _, popularity = build_group_definitions(cfg)
        ranker = LogisticRanker.load(str(cfg.paths.artifact(cfg.ranker.checkpoint_name)))
        retriever = RankedUnionRetriever(
            [
                build_traditional_retriever(cfg, args.traditional_for_hybrid),
                TigerRetriever(cfg, constrained=True),
            ],
            ranker=ranker,
            popularity=popularity,
            long_tail=long_tail,
        )
    else:
        raise ValueError(args.method)
    metrics = evaluate_and_save(cfg, retriever, split=args.split, max_users=args.max_users)
    print(metrics)


if __name__ == "__main__":
    main()
