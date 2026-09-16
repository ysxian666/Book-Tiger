"""Common retrieval interfaces and evaluation example loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import pyarrow.parquet as pq

from tiger_rec.config import Config


@dataclass
class EvalExample:
    user_id: str
    history_items: list[str]
    history_ratings: list[float]
    history_timestamps: list[int]
    target_item: str


@dataclass
class RetrievalBatch:
    items: list[list[str]]
    scores: list[list[float]]
    latency_ms: float
    invalid_beams: int = 0
    total_beams: int = 0


class Retriever(Protocol):
    name: str

    def retrieve(self, histories: list[list[str]], k: int) -> RetrievalBatch:
        ...


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def load_eval_examples(cfg: Config, split: str = "test") -> list[EvalExample]:
    if split not in {"valid", "test"}:
        raise ValueError("split must be valid or test")
    train_rows = _read_rows(cfg.paths.artifact("user_sequences.parquet"))
    target_rows = _read_rows(cfg.paths.artifact(f"{split}_interactions.parquet"))
    target_by_user = {str(row["user_id"]): row for row in target_rows}
    valid_by_user: dict[str, dict[str, Any]] = {}
    if split == "test":
        valid_by_user = {str(row["user_id"]): row for row in _read_rows(cfg.paths.artifact("valid_interactions.parquet"))}
    examples: list[EvalExample] = []
    for row in train_rows:
        user_id = str(row["user_id"])
        target = target_by_user.get(user_id)
        if target is None:
            continue
        items = [str(value) for value in row["item_seq"]]
        ratings = [float(value) for value in row["rating_seq"]]
        timestamps = [int(value) for value in row["timestamp_seq"]]
        if split == "test":
            valid = valid_by_user.get(user_id)
            if valid is None:
                continue
            items = items + [str(valid["item_id"])]
            ratings = ratings + [float(valid["rating"])]
            timestamps = timestamps + [int(valid["timestamp"])]
        examples.append(EvalExample(user_id, items, ratings, timestamps, str(target["item_id"])))
    return examples


def batches(rows: list[Any], batch_size: int) -> Iterable[list[Any]]:
    size = max(1, int(batch_size))
    for start in range(0, len(rows), size):
        yield rows[start:start + size]
