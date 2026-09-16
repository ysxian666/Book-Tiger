"""TIGER generative retriever with standard or trie-constrained decoding."""
from __future__ import annotations

import hashlib
import json
import time

import torch

from tiger_rec.config import Config
from tiger_rec.decoding.constrained_beam import beam_search
from tiger_rec.models.generator import load_generator
from tiger_rec.retrieval.base import RetrievalBatch
from tiger_rec.semantic_id.build import build_trie_from_semantic_ids
from tiger_rec.semantic_id.token_space import PAD, TokenSpace


class TigerRetriever:
    def __init__(self, cfg: Config, constrained: bool = True):
        self.cfg = cfg
        self.name = "tiger_trie" if constrained else "tiger_standard"
        self.supports_user_ids = True
        self.constrained = constrained
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with cfg.paths.artifact("sid_token_space.json").open("r", encoding="utf-8") as f:
            self.token_space = TokenSpace.from_dict(json.load(f))
        with cfg.paths.artifact("semantic_ids.json").open("r", encoding="utf-8") as f:
            self.semantic_ids = json.load(f)
        self.trie = build_trie_from_semantic_ids(self.semantic_ids)
        self.model, self.metadata = load_generator(str(cfg.paths.artifact(cfg.generator.checkpoint_name)), self.token_space, cfg.generator, map_location=self.device)
        self.model.to(self.device).eval()

    def encode_histories(
        self, histories: list[list[str]], user_ids: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences: list[list[int]] = []
        for history_index, history in enumerate(histories):
            chunks: list[list[int]] = []
            for item_id in history[-self.cfg.generator.max_history_items:]:
                row = self.semantic_ids.get(str(item_id))
                if row is None:
                    continue
                chunk = [int(token) for token in row["token_ids"]]
                if self.cfg.generator.use_behavior_tokens:
                    chunk.append(self.token_space.behavior_token(2))
                if self.cfg.generator.use_time_buckets:
                    chunk.append(self.token_space.time_token(0))
                chunks.append(chunk)
            while chunks and sum(len(chunk) for chunk in chunks) > self.cfg.generator.max_input_tokens:
                chunks.pop(0)
            tokens = [token for chunk in chunks for token in chunk]
            if self.cfg.generator.use_user_tokens:
                if user_ids is None:
                    raise ValueError("user_ids are required when generator.use_user_tokens is enabled")
                digest = hashlib.blake2b(str(user_ids[history_index]).encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest, "little") % int(self.cfg.generator.num_user_buckets)
                tokens = [self.token_space.user_token(bucket)] + tokens
            sequences.append(tokens)
        max_len = max(1, max(len(row) for row in sequences))
        input_ids = torch.full((len(sequences), max_len), PAD, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=self.device)
        for index, row in enumerate(sequences):
            if row:
                input_ids[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=self.device)
                attention_mask[index, :len(row)] = 1
        return input_ids, attention_mask

    @torch.inference_mode()
    def retrieve(
        self, histories: list[list[str]], k: int, user_ids: list[str] | None = None,
    ) -> RetrievalBatch:
        start = time.perf_counter()
        input_ids, attention_mask = self.encode_histories(histories, user_ids=user_ids)
        decoded = beam_search(
            self.model, input_ids, attention_mask, self.trie, self.token_space,
            num_beams=int(self.cfg.generator.num_beams),
            max_steps=int(self.cfg.generator.max_decode_steps),
            length_penalty=float(self.cfg.generator.length_penalty),
            constrained=bool(self.constrained),
        )
        items: list[list[str]] = []
        scores: list[list[float]] = []
        invalid_beams = 0
        total_beams = 0
        for user_index, beams in enumerate(decoded):
            ranked_items: list[str] = []
            ranked_scores: list[float] = []
            seen = set(histories[user_index])
            for beam in beams:
                total_beams += 1
                if not beam.valid:
                    invalid_beams += 1
                    continue
                for item_id in beam.item_ids:
                    if item_id in seen or item_id in ranked_items:
                        continue
                    ranked_items.append(item_id)
                    ranked_scores.append(beam.score)
                    if len(ranked_items) >= k:
                        break
                if len(ranked_items) >= k:
                    break
            items.append(ranked_items)
            scores.append(ranked_scores)
        return RetrievalBatch(
            items=items,
            scores=scores,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            invalid_beams=invalid_beams,
            total_beams=total_beams,
        )
