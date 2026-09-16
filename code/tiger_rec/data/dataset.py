"""PyTorch datasets that convert user histories into Semantic ID token sequences."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from tiger_rec.config import Config
from tiger_rec.semantic_id.token_space import EOS, PAD, TokenSpace


@dataclass
class SequenceExample:
    user_id: str
    history_items: list[str]
    history_ratings: list[float]
    history_timestamps: list[int]
    target_item: str
    target_rating: float
    target_timestamp: int


class TimeBucketizer:
    def __init__(self, num_buckets: int = 16):
        self.num_buckets = int(num_buckets)

    def bucket(self, timestamp: int, previous_timestamp: int | None) -> int:
        if previous_timestamp is None:
            return 0
        seconds = max(0.0, (int(timestamp) - int(previous_timestamp)) / 1000.0)
        upper = math.log1p(365.0 * 24.0 * 3600.0)
        bucket = int(math.log1p(seconds) / max(upper, 1e-9) * self.num_buckets)
        return min(max(bucket, 0), self.num_buckets - 1)


def rating_to_behavior(rating: float) -> int:
    if rating <= 2.0:
        return 0
    if rating < 4.0:
        return 1
    return 2


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()

class AmazonSequenceDataset(Dataset):
    """Leave-one-out next-item examples: train prefix -> next item SID."""

    def __init__(self, cfg: Config, split: str = "train"):
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"unknown split: {split}")
        self.cfg = cfg
        self.split = split
        self.token_space = self._load_token_space()
        self.semantic_ids = self._load_semantic_ids()
        self._index: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
        self._build_index()
        self.time_bucketizer = TimeBucketizer(cfg.generator.num_time_buckets)

    def _load_token_space(self) -> TokenSpace:
        with self.cfg.paths.artifact("sid_token_space.json").open("r", encoding="utf-8") as f:
            return TokenSpace.from_dict(json.load(f))

    def _load_semantic_ids(self) -> dict[str, dict[str, Any]]:
        with self.cfg.paths.artifact("semantic_ids.json").open("r", encoding="utf-8") as f:
            return json.load(f)

    def _build_index(self) -> None:
        train_rows = _read_rows(self.cfg.paths.artifact("user_sequences.parquet"))
        split_rows = _read_rows(self.cfg.paths.artifact(f"{self.split}_interactions.parquet"))
        split_by_user = {str(row["user_id"]): row for row in split_rows}
        valid_by_user: dict[str, dict[str, Any]] = {}
        if self.split == "test":
            valid_rows = _read_rows(self.cfg.paths.artifact("valid_interactions.parquet"))
            valid_by_user = {str(row["user_id"]): row for row in valid_rows}
        for row in train_rows:
            user_id = str(row["user_id"])
            history = [str(value) for value in row["item_seq"]]
            ratings = [float(value) for value in row["rating_seq"]]
            timestamps = [int(value) for value in row["timestamp_seq"]]
            target_row = split_by_user.get(user_id)
            base_row = {
                "user_id": user_id,
                "item_seq": history,
                "rating_seq": ratings,
                "timestamp_seq": timestamps,
            }
            if self.split == "train":
                for target_pos in range(1, len(history)):
                    target_item = str(history[target_pos])
                    if target_item not in self.semantic_ids:
                        continue
                    self._index.append((
                        base_row,
                        target_pos,
                        {"item_id": target_item, "rating": ratings[target_pos], "timestamp": timestamps[target_pos]},
                    ))
                continue
            if target_row is None:
                continue
            if str(target_row["item_id"]) not in self.semantic_ids:
                continue
            if self.split == "valid":
                hist_items, hist_ratings, hist_times = history, ratings, timestamps
            else:
                valid_row = valid_by_user.get(user_id)
                if valid_row is None:
                    continue
                hist_items = history + [str(valid_row["item_id"])]
                hist_ratings = ratings + [float(valid_row["rating"])]
                hist_times = timestamps + [int(valid_row["timestamp"])]
            self._index.append((
                {"user_id": user_id, "item_seq": hist_items, "rating_seq": hist_ratings, "timestamp_seq": hist_times},
                len(hist_items),
                {"item_id": str(target_row["item_id"]), "rating": float(target_row["rating"]), "timestamp": int(target_row["timestamp"])},
            ))

    def __len__(self) -> int:
        return len(self._index)

    def _encode_history(self, row: dict[str, Any]) -> list[int]:
        chunks: list[list[int]] = []
        previous_timestamp: int | None = None
        items = row["item_seq"][-self.cfg.generator.max_history_items:]
        ratings = row["rating_seq"][-self.cfg.generator.max_history_items:]
        timestamps = row["timestamp_seq"][-self.cfg.generator.max_history_items:]
        for item_id, rating, timestamp in zip(items, ratings, timestamps):
            sid = self.semantic_ids.get(str(item_id))
            if sid is None:
                continue
            chunk = [int(token) for token in sid["token_ids"]]
            if self.cfg.generator.use_behavior_tokens:
                chunk.append(self.token_space.behavior_token(rating_to_behavior(float(rating))))
            if self.cfg.generator.use_time_buckets:
                chunk.append(self.token_space.time_token(self.time_bucketizer.bucket(int(timestamp), previous_timestamp)))
            chunks.append(chunk)
            previous_timestamp = int(timestamp)
        while chunks and sum(len(chunk) for chunk in chunks) > self.cfg.generator.max_input_tokens:
            chunks.pop(0)
        tokens = [token for chunk in chunks for token in chunk]
        if self.cfg.generator.use_user_tokens:
            digest = hashlib.blake2b(str(row["user_id"]).encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "little") % int(self.cfg.generator.num_user_buckets)
            tokens = [self.token_space.user_token(bucket)] + tokens
        return tokens

    def _encode_target(self, target: dict[str, Any]) -> list[int]:
        sid = self.semantic_ids[str(target["item_id"])]
        tokens = [int(token) for token in sid["token_ids"]] + [EOS]
        return tokens[:self.cfg.generator.max_target_tokens]

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_row, target_pos, target = self._index[index]
        history = {
            "user_id": base_row["user_id"],
            "item_seq": base_row["item_seq"][:target_pos],
            "rating_seq": base_row["rating_seq"][:target_pos],
            "timestamp_seq": base_row["timestamp_seq"][:target_pos],
        }
        return {
            "input_ids": torch.tensor(self._encode_history(history), dtype=torch.long),
            "labels": torch.tensor(self._encode_target(target), dtype=torch.long),
            "user_id": str(history["user_id"]),
            "target_item_id": str(target["item_id"]),
            "target_item": str(target["item_id"]),
        }


def collate_sequence(batch: list[dict[str, Any]], pad_token_id: int = PAD) -> dict[str, Any]:
    max_input = max((row["input_ids"].numel() for row in batch), default=0)
    max_label = max((row["labels"].numel() for row in batch), default=0)
    input_ids = torch.full((len(batch), max_input), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_input), dtype=torch.long)
    labels = torch.full((len(batch), max_label), -100, dtype=torch.long)
    for index, row in enumerate(batch):
        input_ids[index, :row["input_ids"].numel()] = row["input_ids"]
        attention_mask[index, :row["input_ids"].numel()] = 1
        labels[index, :row["labels"].numel()] = row["labels"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "user_ids": [row["user_id"] for row in batch],
        "target_item_ids": [row["target_item_id"] for row in batch],
    }
