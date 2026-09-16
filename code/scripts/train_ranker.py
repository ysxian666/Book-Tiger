"""Train the shared logistic ranker on validation candidates."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiger_rec.config import load_config
from tiger_rec.evaluation.protocol import build_group_definitions
from tiger_rec.models.ranker import LogisticRanker
from tiger_rec.retrieval.base import batches, load_eval_examples
from tiger_rec.retrieval.tiger import TigerRetriever
from tiger_rec.retrieval.traditional import build_traditional_retriever
from tiger_rec.utils import set_seed, write_json


def candidate_features(histories, targets, batches, popularity, long_tail):
    rows, labels = [], []
    for user_index, history in enumerate(histories):
        seen = set(history)
        rrf: dict[str, float] = {}
        source: dict[str, float] = {}
        for output in batches:
            for rank, item_id in enumerate(output.items[user_index], start=1):
                if item_id in seen:
                    continue
                rrf[item_id] = rrf.get(item_id, 0.0) + 1.0 / (60.0 + rank)
                source[item_id] = max(source.get(item_id, -1e9), output.scores[user_index][rank - 1])
        if targets[user_index] not in rrf:
            continue
        for item_id in rrf:
            rows.append([
                rrf[item_id],
                source[item_id],
                float(np.log1p(popularity.get(item_id, 0))),
                1.0 if item_id in long_tail else 0.0,
            ])
            labels.append(1 if item_id == targets[user_index] else 0)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--traditional", default="sasrec")
    parser.add_argument("--max-users", type=int, default=30000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.ranker.random_state)
    examples = load_eval_examples(cfg, split="valid")
    if args.max_users > 0:
        random.Random(cfg.ranker.random_state).shuffle(examples)
        examples = examples[:args.max_users]
    _, _, popularity = build_group_definitions(cfg)
    long_tail, _, _ = build_group_definitions(cfg)
    traditional = build_traditional_retriever(cfg, args.traditional)
    tiger = TigerRetriever(cfg, constrained=True)
    all_features, all_labels = [], []
    for batch in batches(examples, cfg.evaluation.batch_size):
        targets = [example.target_item for example in batch]
        histories = [example.history_items for example in batch]
        outputs = [traditional.retrieve(histories, cfg.ranker.max_candidates), tiger.retrieve(histories, cfg.ranker.max_candidates)]
        features, labels = candidate_features(histories, targets, outputs, popularity, long_tail)
        if features.size == 0:
            features = np.zeros((0, 4), dtype=np.float64)
        if labels.size == 0:
            labels = np.zeros((0,), dtype=np.int64)
        all_features.append(features)
        all_labels.append(labels)
        print(f"[ranker] collected users={len(examples)}")
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if len(np.unique(labels)) < 2:
        raise RuntimeError("ranker training needs retrieved positive and negative candidates")
    ranker = LogisticRanker(cfg.ranker.c_value, cfg.ranker.random_state).fit(features, labels)
    path = cfg.paths.artifact(cfg.ranker.checkpoint_name)
    ranker.save(str(path))
    report = {"rows": int(features.shape[0]), "positives": int(labels.sum()), "users": len(examples), "path": str(path)}
    write_json(cfg.paths.artifact("ranker_training.json"), report)
    print(report)
