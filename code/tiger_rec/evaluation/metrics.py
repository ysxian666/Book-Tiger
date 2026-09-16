"""Ranking and system metrics."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def recall_at_k(recommendations: Sequence[Sequence[str]], targets: Sequence[str], k: int) -> float:
    hits = 0
    for ranked, target in zip(recommendations, targets):
        hits += int(target in list(ranked)[:k])
    return hits / max(len(targets), 1)


def ndcg_at_k(recommendations: Sequence[Sequence[str]], targets: Sequence[str], k: int) -> float:
    value = 0.0
    for ranked, target in zip(recommendations, targets):
        for rank, item in enumerate(list(ranked)[:k], start=1):
            if item == target:
                value += 1.0 / math.log2(rank + 1)
                break
    return value / max(len(targets), 1)


def mrr(recommendations: Sequence[Sequence[str]], targets: Sequence[str], k: int | None = None) -> float:
    total = 0.0
    for ranked, target in zip(recommendations, targets):
        values = list(ranked) if k is None else list(ranked)[:k]
        for rank, item in enumerate(values, start=1):
            if item == target:
                total += 1.0 / rank
                break
    return total / max(len(targets), 1)


def catalog_coverage(recommendations: Sequence[Sequence[str]], catalog_size: int) -> float:
    seen: set[str] = set()
    for ranked in recommendations:
        seen.update(ranked)
    return len(seen) / max(catalog_size, 1)


def group_recall_at_k(recommendations: Sequence[Sequence[str]], targets: Sequence[str], group_items: set[str], k: int) -> float:
    selected = [(ranked, target) for ranked, target in zip(recommendations, targets) if target in group_items]
    if not selected:
        return 0.0
    return recall_at_k([row[0] for row in selected], [row[1] for row in selected], k)


def invalid_rate(valid_flags: Sequence[bool]) -> float:
    if not valid_flags:
        return 0.0
    return sum(not value for value in valid_flags) / len(valid_flags)


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    values = np.asarray(latencies_ms, dtype=np.float64)
    return {"mean_ms": float(values.mean()), "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99))}


def aggregate_metrics(recommendations: Sequence[Sequence[str]], targets: Sequence[str], catalog_size: int, long_tail_items: set[str], cold_start_items: set[str], ks: Iterable[int] = (10, 20, 50), valid_flags: Sequence[bool] | None = None, latencies_ms: Sequence[float] | None = None) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(recommendations, targets, int(k))
        metrics[f"ndcg@{k}"] = ndcg_at_k(recommendations, targets, int(k))
        metrics[f"mrr@{k}"] = mrr(recommendations, targets, int(k))
        metrics[f"long_tail_recall@{k}"] = group_recall_at_k(recommendations, targets, long_tail_items, int(k))
        metrics[f"cold_start_recall@{k}"] = group_recall_at_k(recommendations, targets, cold_start_items, int(k))
    metrics["coverage"] = catalog_coverage(recommendations, catalog_size)
    if valid_flags is not None:
        metrics["invalid_sid_rate"] = invalid_rate(valid_flags)
    if latencies_ms is not None:
        metrics.update(latency_summary(latencies_ms))
    return metrics
