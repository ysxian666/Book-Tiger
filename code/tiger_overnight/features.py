"""Train-only content, collaborative, and behavior-cluster features."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

from .config import data_path, ensure_run_dirs


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ; ".join(f"{k}: {_as_text(v)}" for k, v in value.items() if v is not None)
    if isinstance(value, (list, tuple, set)):
        return " ; ".join(_as_text(v) for v in value if v is not None)
    return str(value)


def build_item_text(row: dict[str, Any]) -> str:
    keys = ("title", "subtitle", "author", "store", "main_category", "categories", "features", "description", "details")
    text = " | ".join(part for part in (_as_text(row.get(key)) for key in keys) if part)
    return " ".join(text.split()) or str(row.get("item_id", ""))


class BlairTextEncoder:
    def __init__(self, config: dict[str, Any]) -> None:
        from transformers import AutoModel, AutoTokenizer

        feature_cfg = config["features"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = str(feature_cfg["text_model"])
        self.batch_size = int(feature_cfg["text_batch_size"])
        self.max_length = int(feature_cfg["text_max_length"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        output = pooled.float().cpu().numpy()
        norm = np.linalg.norm(output, axis=1, keepdims=True).clip(min=1e-12)
        return (output / norm).astype(np.float32)

def encode_catalog_text(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    ensure_run_dirs(config)
    output = data_path(config, "text_embeddings.npy")
    ids_path = data_path(config, "text_item_ids.json")
    report_path = data_path(config, "text_feature_report.json")
    if output.exists() and ids_path.exists() and report_path.exists() and not force:
        with report_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    encoder = BlairTextEncoder(config)
    catalog_path = data_path(config, "catalog.parquet")
    item_ids: list[str] = []
    batches: list[np.ndarray] = []
    parquet = pq.ParquetFile(catalog_path)
    columns = ["item_id", "title", "subtitle", "author", "store", "main_category", "categories", "features", "description", "details"]
    for batch in parquet.iter_batches(batch_size=encoder.batch_size, columns=columns):
        rows = batch.to_pylist()
        item_ids.extend(str(row["item_id"]) for row in rows)
        batches.append(encoder.encode([build_item_text(row) for row in rows]))
    embeddings = np.concatenate(batches, axis=0) if batches else np.zeros((0, 768), dtype=np.float32)
    np.save(output, embeddings)
    _write_json(ids_path, item_ids)
    report = {
        "model": encoder.model_name,
        "items": len(item_ids),
        "dimension": int(embeddings.shape[1]),
        "normalized": True,
        "max_length": encoder.max_length,
        "source_split": "catalog content only",
    }
    _write_json(report_path, report)
    return report


def _catalog_item_ids(config: dict[str, Any]) -> list[str]:
    table = pq.read_table(data_path(config, "catalog.parquet"), columns=["item_id"])
    return [str(value) for value in table.column("item_id").to_pylist()]


def _train_matrix(config: dict[str, Any], item_ids: list[str]) -> tuple[sparse.csr_matrix, pd.Index]:
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    train = pd.read_parquet(data_path(config, "train_interactions.parquet"), columns=["user_id", "item_id"])
    users = pd.Index(train["user_id"].astype(str).unique())
    user_index = {user_id: index for index, user_id in enumerate(users)}
    rows = train["user_id"].astype(str).map(user_index).to_numpy(dtype=np.int64)
    cols = train["item_id"].astype(str).map(item_index).to_numpy(dtype=np.int64)
    valid = cols >= 0
    matrix = sparse.csr_matrix(
        (np.ones(int(valid.sum()), dtype=np.float32), (rows[valid], cols[valid])),
        shape=(len(users), len(item_ids)),
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    return matrix, users

def build_train_only_collaborative(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    ensure_run_dirs(config)
    emb_path = data_path(config, "collab_embeddings.npy")
    ids_path = data_path(config, "collab_item_ids.json")
    report_path = data_path(config, "collab_feature_report.json")
    if emb_path.exists() and ids_path.exists() and report_path.exists() and not force:
        with report_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    item_ids = _catalog_item_ids(config)
    matrix, users = _train_matrix(config, item_ids)
    components = min(int(config["features"]["collaborative_svd_components"]), matrix.shape[0] - 1, matrix.shape[1] - 1)
    if components < 2:
        raise RuntimeError("not enough train interactions for collaborative SVD")
    svd = TruncatedSVD(n_components=components, random_state=int(config["seed"]), n_iter=7)
    item_embeddings = svd.fit_transform(matrix.T).astype(np.float32)
    target_dim = int(config["features"]["collaborative_dim"])
    if components < target_dim:
        item_embeddings = np.pad(item_embeddings, ((0, 0), (0, target_dim - components)))
    item_embeddings = item_embeddings[:, :target_dim]
    item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True).clip(min=1e-12)
    np.save(emb_path, item_embeddings)
    _write_json(ids_path, item_ids)
    report = {
        "method": "truncated_svd_on_train_only_user_item_binary_matrix",
        "items": len(item_ids),
        "train_users": int(len(users)),
        "train_interactions": int(matrix.nnz),
        "components": int(components),
        "dimension": int(item_embeddings.shape[1]),
        "valid_or_test_used": False,
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
    }
    _write_json(report_path, report)
    return report


def build_behavior_clusters(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    ensure_run_dirs(config)
    labels_path = data_path(config, "user_clusters.parquet")
    profiles_path = data_path(config, "item_behavior_profiles.npy")
    report_path = data_path(config, "behavior_cluster_report.json")
    if labels_path.exists() and profiles_path.exists() and report_path.exists() and not force:
        with report_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    item_ids = _catalog_item_ids(config)
    matrix, users = _train_matrix(config, item_ids)
    n_components = min(64, matrix.shape[0] - 1, matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=int(config["seed"]), n_iter=5)
    user_features = svd.fit_transform(matrix).astype(np.float32)
    user_features /= np.linalg.norm(user_features, axis=1, keepdims=True).clip(min=1e-12)
    n_clusters = min(int(config["sid_allocation"]["num_user_clusters"]), len(users))
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=int(config["seed"]),
        batch_size=min(8192, max(256, len(users))),
        n_init=10,
        max_iter=200,
    )
    labels = kmeans.fit_predict(user_features).astype(np.int32)
    pd.DataFrame({"user_id": users.astype(str), "cluster": labels}).to_parquet(labels_path, index=False)
    user_rows, item_cols = matrix.nonzero()
    interaction_matrix = sparse.csr_matrix(
        (np.ones(len(user_rows), dtype=np.float32), (item_cols, labels[user_rows].astype(np.int64))),
        shape=(len(item_ids), n_clusters),
    )
    profiles = interaction_matrix.toarray().astype(np.float32)
    profiles = profiles / np.maximum(profiles.sum(axis=1, keepdims=True), 1.0)
    np.save(profiles_path, profiles)
    report = {
        "source": "train_interactions_only",
        "users": int(len(users)),
        "items": int(len(item_ids)),
        "num_clusters": int(n_clusters),
        "user_svd_components": int(n_components),
        "item_profile": "empirical cluster distribution from train interactions",
        "valid_or_test_used": False,
        "cluster_sizes": np.bincount(labels, minlength=n_clusters).astype(int).tolist(),
    }
    _write_json(report_path, report)
    return report

def build_features(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    return {
        "text": encode_catalog_text(config, force=force),
        "collaborative": build_train_only_collaborative(config, force=force),
        "behavior": build_behavior_clusters(config, force=force),
    }
