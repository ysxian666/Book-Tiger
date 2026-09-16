"""Popularity, ItemCF and neural traditional retrieval baselines."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.sparse import csr_matrix

from tiger_rec.retrieval.base import RetrievalBatch


def load_item_ids(cfg) -> list[str]:
    table = pq.read_table(cfg.paths.artifact("catalog.parquet"), columns=["item_id"])
    return [str(value) for value in table.column("item_id").to_pylist()]


def _top_items(scores: np.ndarray, item_ids: list[str], seen: set[str], k: int, exclude_seen: bool = True) -> tuple[list[str], list[float]]:
    scores = np.asarray(scores, dtype=np.float64).copy()
    if exclude_seen and seen:
        item_index = {item_id: index for index, item_id in enumerate(item_ids)}
        for item in seen:
            index = item_index.get(item)
            if index is not None:
                scores[index] = -np.inf
    k = min(max(1, int(k)), len(item_ids))
    indices = np.argpartition(-scores, k - 1)[:k]
    indices = indices[np.argsort(-scores[indices])]
    return [item_ids[int(index)] for index in indices], [float(scores[int(index)]) for index in indices]


class PopularityRetriever:
    name = "popularity"

    def __init__(self, cfg):
        self.item_ids = load_item_ids(cfg)
        table = pq.read_table(cfg.paths.artifact("catalog.parquet"), columns=["item_id", "train_count"])
        counts = dict(zip((str(v) for v in table.column("item_id").to_pylist()), table.column("train_count").to_pylist()))
        self.scores = np.asarray([float(counts.get(item_id, 0)) for item_id in self.item_ids], dtype=np.float64)

    def retrieve(self, histories: list[list[str]], k: int) -> RetrievalBatch:
        start = time.perf_counter()
        items, scores = [], []
        for history in histories:
            ranked, values = _top_items(self.scores, self.item_ids, set(history), k)
            items.append(ranked)
            scores.append(values)
        return RetrievalBatch(items=items, scores=scores, latency_ms=(time.perf_counter() - start) * 1000.0)


class ItemCFRetriever:
    name = "itemcf"

    def __init__(self, cfg):
        self.item_ids = load_item_ids(cfg)
        self.index = {item_id: index for index, item_id in enumerate(self.item_ids)}
        rows = pq.read_table(cfg.paths.artifact("user_sequences.parquet"), columns=["item_seq"]).column("item_seq").to_pylist()
        user_indices: list[int] = []
        item_indices: list[int] = []
        for user_index, sequence in enumerate(rows):
            for item_id in sequence:
                index = self.index.get(str(item_id))
                if index is not None:
                    user_indices.append(user_index)
                    item_indices.append(index)
        matrix = csr_matrix((np.ones(len(item_indices), dtype=np.float32), (user_indices, item_indices)), shape=(len(rows), len(self.item_ids)))
        matrix.data[:] = 1.0
        similarity = (matrix.transpose() @ matrix).astype(np.float32)
        similarity.setdiag(0.0)
        similarity.eliminate_zeros()
        degrees = np.asarray(similarity.sum(axis=1)).ravel().clip(min=1e-12)
        normalization = 1.0 / np.sqrt(degrees)
        similarity = similarity.multiply(normalization[:, None]).multiply(normalization[None, :]).tocsr()
        self.similarity = similarity

    def retrieve(self, histories: list[list[str]], k: int) -> RetrievalBatch:
        start = time.perf_counter()
        all_items, all_scores = [], []
        for history in histories:
            indices = [self.index[item] for item in history if item in self.index]
            if not indices:
                scores = np.zeros(len(self.item_ids), dtype=np.float64)
            else:
                scores = np.asarray(self.similarity[indices].sum(axis=0)).ravel().astype(np.float64)
            ranked, values = _top_items(scores, self.item_ids, set(history), k)
            all_items.append(ranked)
            all_scores.append(values)
        return RetrievalBatch(items=all_items, scores=all_scores, latency_ms=(time.perf_counter() - start) * 1000.0)

class NeuralRetriever:
    """Full-catalog scorer for TwoTower / GRU4Rec / SASRec / BERT4Rec."""

    def __init__(self, cfg, model_name: str, checkpoint_path: str | None = None):
        from tiger_rec.models.baselines import BERT4Rec, GRU4Rec, SASRec, TwoTower

        self.name = model_name
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        default_checkpoint = cfg.paths.artifact(f"baselines_{model_name}.pt")
        legacy_checkpoint = cfg.paths.artifact(cfg.baselines.checkpoint_name)
        if checkpoint_path is None and not default_checkpoint.exists() and legacy_checkpoint.exists():
            default_checkpoint = legacy_checkpoint
        self.checkpoint_path = checkpoint_path or str(default_checkpoint)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.item_ids = [str(value) for value in checkpoint["item_ids"]]
        self.item_to_model = {str(key): int(value) for key, value in checkpoint["item_to_model"].items()}
        self.dim = int(checkpoint["dim"])
        self.num_items = int(checkpoint["num_items"])
        cls = {"twotower": TwoTower, "gru4rec": GRU4Rec, "sasrec": SASRec, "bert4rec": BERT4Rec}[model_name]
        self.model = cls(self.num_items, dim=self.dim, num_layers=int(cfg.baselines.num_layers), num_heads=int(cfg.baselines.num_heads), dropout=float(cfg.baselines.dropout), max_len=int(cfg.baselines.max_history_items) + 2).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model_indices = torch.tensor([self.item_to_model[item_id] for item_id in self.item_ids], device=self.device, dtype=torch.long)

    def _sequence_tensor(self, histories: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(1, min(int(self.cfg.baselines.max_history_items), max(len(row) for row in histories)))
        sequence = torch.zeros((len(histories), max_len), dtype=torch.long, device=self.device)
        lengths = torch.zeros(len(histories), dtype=torch.long, device=self.device)
        for row_index, history in enumerate(histories):
            values = [self.item_to_model[item] for item in history if item in self.item_to_model][-max_len:]
            lengths[row_index] = len(values)
            if values:
                sequence[row_index, :len(values)] = torch.tensor(values, dtype=torch.long, device=self.device)
        return sequence, lengths.clamp_min(1)

    @torch.inference_mode()
    def retrieve(self, histories: list[list[str]], k: int) -> RetrievalBatch:
        start = time.perf_counter()
        sequence, lengths = self._sequence_tensor(histories)
        user_vectors = self.model.encode(sequence, lengths)
        item_embeddings = self.model.item_embedding(self.model_indices)
        score_matrix = (user_vectors @ item_embeddings.t()).float().cpu().numpy()
        items, scores = [], []
        for row_index, history in enumerate(histories):
            ranked, values = _top_items(score_matrix[row_index], self.item_ids, set(history), k)
            items.append(ranked)
            scores.append(values)
        return RetrievalBatch(items=items, scores=scores, latency_ms=(time.perf_counter() - start) * 1000.0)


def build_traditional_retriever(cfg, name: str):
    if name == "popularity":
        return PopularityRetriever(cfg)
    if name == "itemcf":
        return ItemCFRetriever(cfg)
    if name in {"twotower", "gru4rec", "sasrec", "bert4rec"}:
        return NeuralRetriever(cfg, name)
    raise ValueError(f"unsupported traditional retriever: {name}")
