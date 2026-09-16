"""Text encoders for item content.

The default full configuration uses the public BLaIR RoBERTa checkpoint. A
deterministic hashing encoder is included so smoke tests can run offline.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch

from tiger_rec.config import Config, TextConfig
from tiger_rec.utils import ensure_parent, resolve_device, sha256_file, write_json


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ; ".join(f"{k}: {_stringify(v)}" for k, v in value.items() if v is not None)
    if isinstance(value, (list, tuple, set)):
        return " ; ".join(_stringify(v) for v in value if v is not None)
    return str(value)


def build_item_text(row: dict[str, Any]) -> str:
    fields = [row.get("title"), row.get("subtitle"), row.get("author"), row.get("store"), row.get("main_category"), row.get("categories"), row.get("features"), row.get("description"), row.get("details")]
    text = " | ".join(part for part in (_stringify(value) for value in fields) if part)
    if not text:
        text = _stringify(row.get("item_id"))
    return " ".join(text.split())


class HashTextEncoder:
    def __init__(self, dim: int = 384):
        self.dim = int(dim)
        self.token_pattern = re.compile(r"[A-Za-z0-9_]+")

    def encode(self, texts: list[str]) -> np.ndarray:
        output = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = self.token_pattern.findall(text.lower())
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                number = int.from_bytes(digest, "little")
                index = number % self.dim
                sign = 1.0 if (number >> 63) == 0 else -1.0
                output[row, index] += sign
            norm = np.linalg.norm(output[row])
            if norm > 0:
                output[row] /= norm
        return output


class HuggingFaceTextEncoder:
    def __init__(self, config: TextConfig):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for HuggingFace text encoding") from exc
        self.config = config
        self.device = resolve_device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=config.trust_remote_code)
        self.model = AutoModel.from_pretrained(config.model_name, trust_remote_code=config.trust_remote_code).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        batch = self.tokenizer([self.config.passage_prefix + text for text in texts], padding=True, truncation=True, max_length=int(self.config.max_length), return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        output = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(output.dtype)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        embeddings = pooled.float().cpu().numpy()
        if self.config.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
        return embeddings.astype(np.float32)


def create_text_encoder(config: TextConfig):
    if config.model_name == "hash":
        return HashTextEncoder(dim=int(config.output_dim))
    try:
        return HuggingFaceTextEncoder(config)
    except Exception:
        if not config.allow_hash_fallback:
            raise
        print("[text] HuggingFace model unavailable; falling back to HashTextEncoder")
        return HashTextEncoder(dim=int(config.output_dim))


def iter_catalog_batches(catalog_path: str | Path, batch_size: int = 2048) -> Iterable[tuple[list[str], list[str]]]:
    parquet = pq.ParquetFile(catalog_path)
    columns = ["item_id", "title", "subtitle", "author", "store", "main_category", "categories", "features", "description", "details"]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        rows = batch.to_pylist()
        yield [str(row["item_id"]) for row in rows], [build_item_text(row) for row in rows]


def encode_catalog(cfg: Config) -> dict[str, Any]:
    catalog_path = cfg.paths.artifact("catalog.parquet")
    if not catalog_path.exists():
        raise FileNotFoundError(f"catalog not found: {catalog_path}")
    encoder = create_text_encoder(cfg.text)
    item_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    for batch_index, (batch_ids, batch_texts) in enumerate(iter_catalog_batches(catalog_path, batch_size=cfg.text.batch_size)):
        item_ids.extend(batch_ids)
        embeddings.append(encoder.encode(batch_texts).astype(np.float32))
        if batch_index % 20 == 0:
            print(f"[text] encoded {len(item_ids):,} items")
    matrix = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, cfg.text.output_dim), dtype=np.float32)
    output_path = cfg.paths.artifact("text_embeddings.npy")
    ensure_parent(output_path)
    np.save(output_path, matrix)
    write_json(cfg.paths.artifact("text_item_ids.json"), item_ids)
    metadata = {
        "model_name": cfg.text.model_name,
        "output_dim": int(matrix.shape[1]),
        "items": int(matrix.shape[0]),
        "normalize": bool(cfg.text.normalize),
        "max_length": int(cfg.text.max_length),
        "catalog_sha256": sha256_file(catalog_path),
    }
    write_json(cfg.paths.artifact("text_embeddings_meta.json"), metadata)
    print(f"[text] saved {matrix.shape} to {output_path}")
    return metadata
