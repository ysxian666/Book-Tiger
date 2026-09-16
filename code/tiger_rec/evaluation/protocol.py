"""End-to-end offline evaluation protocol."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from tiger_rec.config import Config
from tiger_rec.evaluation.metrics import aggregate_metrics
from tiger_rec.retrieval.base import batches, load_eval_examples
from tiger_rec.utils import write_json


def build_group_definitions(cfg: Config) -> tuple[set[str], set[str], dict[str, int]]:
    table = pq.read_table(cfg.paths.artifact("item_stats.parquet"), columns=["item_id", "train_count"])
    item_ids = [str(value) for value in table.column("item_id").to_pylist()]
    counts = table.column("train_count").to_numpy(zero_copy_only=False).astype(np.int64)
    threshold = float(np.quantile(counts, float(cfg.evaluation.long_tail_quantile)))
    long_tail = {item_id for item_id, count in zip(item_ids, counts) if int(count) <= threshold}
    cold_start = {
        item_id
        for item_id, count in zip(item_ids, counts)
        if int(count) <= int(cfg.evaluation.cold_start_max_train_count)
    }
    return long_tail, cold_start, {item_id: int(count) for item_id, count in zip(item_ids, counts)}


def evaluate_retriever(
    cfg: Config,
    retriever,
    split: str = "test",
    max_users: int = 0,
) -> dict[str, Any]:
    examples = load_eval_examples(cfg, split=split)
    if max_users > 0:
        examples = examples[: int(max_users)]
    long_tail, cold_start, _ = build_group_definitions(cfg)
    primary_k = int(cfg.evaluation.primary_k)
    all_recommendations: list[list[str]] = []
    targets: list[str] = []
    valid_flags: list[bool] = []
    latencies: list[float] = []
    started = time.perf_counter()
    for batch in batches(examples, cfg.evaluation.batch_size):
        histories = [example.history_items for example in batch]
        top_k = max(cfg.evaluation.top_k)
        if getattr(retriever, "supports_user_ids", False):
            output = retriever.retrieve(
                histories,
                top_k,
                user_ids=[example.user_id for example in batch],
            )
        else:
            output = retriever.retrieve(histories, top_k)
        all_recommendations.extend(output.items)
        targets.extend(example.target_item for example in batch)
        latencies.append(float(output.latency_ms))
        if output.total_beams:
            valid_flags.extend([False] * int(output.invalid_beams))
            valid_flags.extend([True] * int(output.total_beams - output.invalid_beams))
    catalog_size = int(pq.read_table(cfg.paths.artifact("catalog.parquet"), columns=["item_id"]).num_rows)
    metrics = aggregate_metrics(
        recommendations=all_recommendations,
        targets=targets,
        catalog_size=catalog_size,
        long_tail_items=long_tail,
        cold_start_items=cold_start,
        ks=cfg.evaluation.top_k,
        valid_flags=valid_flags if valid_flags else None,
        latencies_ms=latencies,
    )
    metrics["users"] = float(len(examples))
    metrics["wall_time_sec"] = float(time.perf_counter() - started)
    metrics["split"] = split
    metrics["method"] = getattr(retriever, "name", type(retriever).__name__)
    return metrics


def evaluate_and_save(cfg: Config, retriever, split: str = "test", max_users: int = 0) -> dict[str, Any]:
    metrics = evaluate_retriever(cfg, retriever, split=split, max_users=max_users)
    safe_method = str(metrics['method']).replace(':', '_').replace('+', '_')
    output_path = cfg.paths.artifact("metrics", f"{safe_method}_{split}.json")
    write_json(output_path, metrics)
    print(f"[eval] {metrics['method']} recall@{cfg.evaluation.primary_k}={metrics.get(f'recall@{cfg.evaluation.primary_k}', 0.0):.4f}")
    return metrics
