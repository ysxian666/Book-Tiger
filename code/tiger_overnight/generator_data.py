"""Datasets and batching for TIGER generator training and evaluation."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tiger_rec.semantic_id.token_space import BOS, EOS, PAD, TokenSpace

from .config import artifact_path, data_path


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def user_bucket(user_id: str, num_buckets: int) -> int:
    digest = hashlib.blake2b(str(user_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % int(num_buckets)


def load_token_space(config: dict[str, Any]) -> TokenSpace:
    return TokenSpace(
        num_levels=int(config["rqvae"]["num_levels"]),
        codebook_size=int(config["rqvae"]["codebook_size"]),
        max_conflict_tokens=int(config["data"]["max_suffix_tokens"]),
        use_behavior_tokens=False,
        use_time_buckets=bool(config["generator"]["use_time_buckets"]),
        use_user_tokens=bool(config["generator"]["use_hashed_user_token"]),
        num_user_buckets=int(config["generator"]["num_user_buckets"]),
        num_time_buckets=int(config["generator"]["num_time_buckets"]),
        behavior_count=0,
    )


SID_DIRS = {
    "baseline": "baseline",
    "bcgsid": "bcgsid",
    "bcgsid_shuffled": "bcgsid_shuffled",
    "bcgsid_hard": "bcgsid_hard",
}


def sid_dir(config: dict[str, Any], mode: str) -> Path:
    if mode not in SID_DIRS:
        raise ValueError(f"unknown SID mode: {mode}")
    return artifact_path(config, "sid", SID_DIRS[mode])


def load_sid_map_and_order(config: dict[str, Any], mode: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    directory = sid_dir(config, mode)
    mapping = _load_json(directory / "semantic_ids.json")
    item_ids = [str(value) for value in _load_json(data_path(config, "text_item_ids.json"))]
    if any(item_id not in mapping for item_id in item_ids):
        raise RuntimeError(f"incomplete {mode} SID mapping for catalog")
    return mapping, item_ids


class TigerSequenceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        config: dict[str, Any],
        sid_mode: str,
        split: str,
        use_soft_prefix: bool,
        sample_limit: int = 0,
        user_filter: set[str] | None = None,
        shuffle_user_context: bool = False,
    ) -> None:
        self.config = config
        self.sid_mode = sid_mode
        self.use_soft_prefix = bool(use_soft_prefix)
        self.shuffle_user_context = bool(shuffle_user_context)
        self.max_history = int(config["generator"]["max_history_items"])
        self.token_space = load_token_space(config)
        self.sid_map, self.item_ids = load_sid_map_and_order(config, sid_mode)
        self.item_index = {item_id: index for index, item_id in enumerate(self.item_ids)}
        self.item_latents = np.load(sid_dir(config, sid_mode) / "item_latents.npy").astype(np.float32)
        self.target_text = np.load(data_path(config, "text_embeddings.npy")).astype(np.float32)
        self.target_collab = np.load(data_path(config, "collab_embeddings.npy")).astype(np.float32)
        self.use_time = bool(config["generator"]["use_time_buckets"])
        self.num_time_buckets = int(config["generator"]["num_time_buckets"])
        self.split = split
        self.train_items: dict[str, list[str]] = {}
        self.eval_history: dict[str, list[str]] = {}
        self.context_histories: dict[str, list[str]] = {}
        self.context_permutation: dict[str, str] = {}
        if self.shuffle_user_context:
            context_rows = pd.read_parquet(
                data_path(config, "user_sequences.parquet"),
                columns=["user_id", "history_item_ids"],
            )
            for row in context_rows.itertuples(index=False):
                self.context_histories[str(row.user_id)] = [
                    str(value) for value in list(row.history_item_ids)
                ][-self.max_history:]
            context_users = sorted(self.context_histories)
            rng = np.random.default_rng(int(config["seed"]) + 202)
            permutation = rng.permutation(len(context_users))
            self.context_permutation = {
                context_users[index]: context_users[int(permutation[index])]
                for index in range(len(context_users))
            }
        self.examples: list[tuple[str, int, str]] = []
        if split == "train":
            frame = pd.read_parquet(data_path(config, "train_interactions.parquet"), columns=["user_id", "item_id", "timestamp"])
            frame = frame.sort_values(["user_id", "timestamp", "item_id"])
            for user_id, group in frame.groupby("user_id", sort=False):
                user_id = str(user_id)
                items = group["item_id"].astype(str).tolist()
                self.train_items[user_id] = items
                for target_index in range(1, len(items)):
                    self.examples.append((user_id, target_index, ""))
        elif split in ("valid", "test"):
            target_column = "valid_item_id" if split == "valid" else "test_item_id"
            sequences = pd.read_parquet(
                data_path(config, "user_sequences.parquet"),
                columns=["user_id", "history_item_ids", target_column],
            )
            for row in sequences.itertuples(index=False):
                user_id = str(row.user_id)
                target_item = str(getattr(row, target_column))
                if target_item not in self.sid_map:
                    continue
                self.eval_history[user_id] = [str(value) for value in list(row.history_item_ids)][-self.max_history:]
                self.examples.append((user_id, -1, target_item))
        else:
            raise ValueError(f"unsupported split: {split}")
        if user_filter is not None:
            self.examples = [row for row in self.examples if row[0] in user_filter]
        if sample_limit > 0:
            self.examples = self.examples[: int(sample_limit)]
        if not self.examples:
            raise RuntimeError(f"no examples available for split={split}, sid={sid_mode}")

    def __len__(self) -> int:
        return len(self.examples)

    def _time_tokens(self, length: int) -> list[int]:
        if not self.use_time:
            return []
        tokens = []
        for position in range(length):
            # The newest history item receives the largest bucket.
            distance = length - position - 1
            bucket = self.num_time_buckets - 1 - min(distance, self.num_time_buckets - 1)
            tokens.append(self.token_space.time_token(bucket))
        return tokens

    def _encode_input(self, user_id: str, history: list[str]) -> tuple[list[int], list[int]]:
        input_ids = [BOS]
        attention = [1]
        if self.token_space.use_user_tokens:
            input_ids.append(self.token_space.user_token(user_bucket(user_id, self.token_space.num_user_buckets)))
            attention.append(1)
        history = [item_id for item_id in history if item_id in self.sid_map][-self.max_history:]
        time_tokens = self._time_tokens(len(history))
        for index, item_id in enumerate(history):
            input_ids.extend(int(value) for value in self.sid_map[item_id]["token_ids"])
            attention.extend([1] * len(self.sid_map[item_id]["token_ids"]))
            if self.use_time:
                input_ids.append(time_tokens[index])
                attention.append(1)
        if self.use_soft_prefix:
            soft_count = int(self.config["user_context"]["num_soft_tokens"])
            input_ids = [PAD] * soft_count + input_ids
            attention = [1] * soft_count + attention
        max_tokens = int(self.config["generator"]["max_input_tokens"])
        if len(input_ids) > max_tokens:
            if self.use_soft_prefix:
                soft_count = int(self.config["user_context"]["num_soft_tokens"])
                prefix_ids = input_ids[:soft_count]
                prefix_mask = attention[:soft_count]
                input_ids = prefix_ids + input_ids[-(max_tokens - soft_count):]
                attention = prefix_mask + attention[-(max_tokens - soft_count):]
            else:
                input_ids = input_ids[:max_tokens]
                attention = attention[:max_tokens]
        return input_ids, attention

    def _contexts(self, user_id: str, history: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if self.shuffle_user_context:
            source_user = self.context_permutation.get(user_id, user_id)
            history = self.context_histories.get(source_user, history)
        indices = [self.item_index[item_id] for item_id in history if item_id in self.item_index]
        if not indices:
            dim = self.item_latents.shape[1]
            zero = np.zeros(dim, dtype=np.float32)
            return zero, zero
        long_window = int(self.config["user_context"]["long_window"])
        short_window = int(self.config["user_context"]["short_window"])
        long_values = self.item_latents[indices[-long_window:]].mean(axis=0)
        short_values = self.item_latents[indices[-short_window:]].mean(axis=0)
        return long_values.astype(np.float32), short_values.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        user_id, target_position, target_item = self.examples[index]
        if self.split == "train":
            items = self.train_items[user_id]
            history = items[max(0, target_position - self.max_history):target_position]
            target_item = items[target_position]
        else:
            history = self.eval_history[user_id]
        input_ids, attention = self._encode_input(user_id, history)
        target = self.sid_map[target_item]
        labels = [int(value) for value in target["token_ids"]] + [EOS]
        long_context, short_context = self._contexts(user_id, history)
        target_index = self.item_index[target_item]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "long_context": torch.from_numpy(long_context),
            "short_context": torch.from_numpy(short_context),
            "history_indices": torch.tensor([self.item_index[item_id] for item_id in history if item_id in self.item_index], dtype=torch.long),
            "target_index": torch.tensor(target_index, dtype=torch.long),
            "target_text": torch.from_numpy(self.target_text[target_index].copy()),
            "target_collab": torch.from_numpy(self.target_collab[target_index].copy()),
            "user_id": user_id,
            "history_length": len(history),
            "target_item_id": target_item,
        }


@dataclass
class SequenceBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    long_context: torch.Tensor
    short_context: torch.Tensor
    target_index: torch.Tensor
    target_text: torch.Tensor
    target_collab: torch.Tensor
    history_indices: list[torch.Tensor]
    user_ids: list[str]
    target_item_ids: list[str]


def collate_sequences(rows: list[dict[str, Any]]) -> SequenceBatch:
    max_input = max(int(row["input_ids"].numel()) for row in rows)
    max_label = max(int(row["labels"].numel()) for row in rows)
    input_ids = torch.full((len(rows), max_input), PAD, dtype=torch.long)
    attention = torch.zeros((len(rows), max_input), dtype=torch.long)
    labels = torch.full((len(rows), max_label), PAD, dtype=torch.long)
    for index, row in enumerate(rows):
        length = int(row["input_ids"].numel())
        label_length = int(row["labels"].numel())
        input_ids[index, :length] = row["input_ids"]
        attention[index, :length] = row["attention_mask"]
        labels[index, :label_length] = row["labels"]
    return SequenceBatch(
        input_ids=input_ids,
        attention_mask=attention,
        labels=labels,
        long_context=torch.stack([row["long_context"] for row in rows]),
        short_context=torch.stack([row["short_context"] for row in rows]),
        target_index=torch.stack([row["target_index"] for row in rows]),
        target_text=torch.stack([row["target_text"] for row in rows]),
        target_collab=torch.stack([row["target_collab"] for row in rows]),
        history_indices=[row["history_indices"] for row in rows],
        user_ids=[str(row["user_id"]) for row in rows],
        target_item_ids=[str(row["target_item_id"]) for row in rows],
    )


def select_eval_users(config: dict[str, Any], split: str, sample_size: int, seed: int | None = None) -> list[str]:
    target = "valid_item_id" if split == "valid" else "test_item_id"
    sequences = pd.read_parquet(
        data_path(config, "user_sequences.parquet"),
        columns=["user_id", "history_length", "history_item_ids", target],
    )
    sequences = sequences.dropna(subset=[target]).copy()
    sequences["history_bucket"] = pd.cut(
        sequences["history_length"],
        bins=[-1, 5, 20, np.inf],
        labels=["short", "medium", "long"],
    )
    rng = np.random.default_rng(int(seed if seed is not None else config["seed"]))
    per_bucket = max(1, int(math.ceil(int(sample_size) / 3)))
    selected: list[str] = []
    for bucket in ("short", "medium", "long"):
        group = sequences[sequences["history_bucket"] == bucket]
        if group.empty:
            continue
        take = min(len(group), per_bucket if bucket != "long" else max(1, int(sample_size) - len(selected)))
        chosen = rng.choice(group["user_id"].astype(str).to_numpy(), size=take, replace=False)
        selected.extend(str(value) for value in chosen)
        if len(selected) >= int(sample_size):
            break
    if len(selected) < int(sample_size):
        remaining = sequences[~sequences["user_id"].astype(str).isin(selected)]
        if not remaining.empty:
            take = min(len(remaining), int(sample_size) - len(selected))
            chosen = rng.choice(remaining["user_id"].astype(str).to_numpy(), size=take, replace=False)
            selected.extend(str(value) for value in chosen)
    return selected[: int(sample_size)]
