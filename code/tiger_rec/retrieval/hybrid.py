"""Hybrid retrieval."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Sequence

from tiger_rec.retrieval.base import RetrievalBatch


def reciprocal_rank_fusion(batches: Sequence[RetrievalBatch], weights: Sequence[float] | None = None, k: int = 20, rrf_constant: float = 60.0) -> RetrievalBatch:
    if not batches:
        return RetrievalBatch(items=[], scores=[], latency_ms=0.0)
    weights = list(weights) if weights is not None else [1.0] * len(batches)
    items, scores = [], []
    for user_index in range(len(batches[0].items)):
        fused: dict[str, float] = defaultdict(float)
        for batch, weight in zip(batches, weights):
            for rank, item_id in enumerate(batch.items[user_index], start=1):
                fused[item_id] += float(weight) / (rrf_constant + rank)
        ranked = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)[:k]
        items.append([item_id for item_id, _ in ranked])
        scores.append([score for _, score in ranked])
    return RetrievalBatch(items=items, scores=scores, latency_ms=sum(batch.latency_ms for batch in batches), invalid_beams=sum(batch.invalid_beams for batch in batches), total_beams=sum(batch.total_beams for batch in batches))


class HybridRetriever:
    def __init__(self, retrievers, weights: Sequence[float] | None = None, rrf_constant: float = 60.0):
        self.retrievers = list(retrievers)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.retrievers)
        self.rrf_constant = float(rrf_constant)
        self.name = "hybrid:" + "+".join(getattr(retriever, "name", "retriever") for retriever in self.retrievers)
        self.supports_user_ids = any(getattr(retriever, "supports_user_ids", False) for retriever in self.retrievers)

    def retrieve(self, histories: list[list[str]], k: int, user_ids: list[str] | None = None) -> RetrievalBatch:
        start = time.perf_counter()
        candidate_batches = []
        for retriever in self.retrievers:
            if getattr(retriever, "supports_user_ids", False):
                candidate_batches.append(retriever.retrieve(histories, max(k * 5, k), user_ids=user_ids))
            else:
                candidate_batches.append(retriever.retrieve(histories, max(k * 5, k)))
        fused = reciprocal_rank_fusion(candidate_batches, self.weights, k=k, rrf_constant=self.rrf_constant)
        fused.latency_ms = (time.perf_counter() - start) * 1000.0
        return fused


class RankedUnionRetriever:
    """Union candidates from several retrievers and apply one shared ranker."""

    def __init__(self, retrievers, ranker, popularity: dict[str, int], long_tail: set[str], candidate_multiplier: int = 10):
        self.retrievers = list(retrievers)
        self.ranker = ranker
        self.popularity = popularity
        self.long_tail = long_tail
        self.candidate_multiplier = int(candidate_multiplier)
        self.name = "ranked_union:" + "+".join(getattr(retriever, "name", "retriever") for retriever in self.retrievers)
        self.supports_user_ids = any(getattr(retriever, "supports_user_ids", False) for retriever in self.retrievers)

    def retrieve(self, histories: list[list[str]], k: int, user_ids: list[str] | None = None) -> RetrievalBatch:
        import numpy as np

        start = time.perf_counter()
        candidate_k = max(k * self.candidate_multiplier, 100)
        batches = []
        for retriever in self.retrievers:
            if getattr(retriever, "supports_user_ids", False):
                batches.append(retriever.retrieve(histories, candidate_k, user_ids=user_ids))
            else:
                batches.append(retriever.retrieve(histories, candidate_k))
        items: list[list[str]] = []
        scores: list[list[float]] = []
        for user_index, history in enumerate(histories):
            rrf: dict[str, float] = defaultdict(float)
            source: dict[str, float] = defaultdict(float)
            for batch in batches:
                for rank, item_id in enumerate(batch.items[user_index], start=1):
                    rrf[item_id] += 1.0 / (60.0 + rank)
                    source[item_id] = max(source[item_id], batch.scores[user_index][rank - 1])
            candidate_ids = [item_id for item_id in rrf if item_id not in set(history)]
            if not candidate_ids:
                items.append([])
                scores.append([])
                continue
            features = np.asarray([
                [
                    rrf[item_id],
                    source[item_id],
                    np.log1p(self.popularity.get(item_id, 0)),
                    1.0 if item_id in self.long_tail else 0.0,
                ]
                for item_id in candidate_ids
            ], dtype=np.float64)
            predicted = self.ranker.predict(features)
            order = np.argsort(-predicted)[:k]
            items.append([candidate_ids[int(index)] for index in order])
            scores.append([float(predicted[int(index)]) for index in order])
        return RetrievalBatch(
            items=items,
            scores=scores,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            invalid_beams=sum(batch.invalid_beams for batch in batches),
            total_beams=sum(batch.total_beams for batch in batches),
        )
