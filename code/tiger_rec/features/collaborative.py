"""Collaborative item embeddings learned from the user-item interaction graph."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from tiger_rec.config import Config
from tiger_rec.utils import ensure_parent, sha256_file, write_json


def _load_catalog_ids(path: str | Path) -> list[str]:
    table = pq.read_table(path, columns=["item_id"])
    return [str(value) for value in table.column("item_id").to_pylist()]


def build_user_item_csr(
    interactions_path: str | Path,
    item_to_index: dict[str, int],
    min_user_degree: int = 2,
) -> tuple[csr_matrix, dict[str, int]]:
    user_to_index: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    parquet = pq.ParquetFile(interactions_path)
    for batch in parquet.iter_batches(batch_size=65536, columns=["user_id", "item_id"]):
        users = batch.column("user_id").to_pylist()
        items = batch.column("item_id").to_pylist()
        for user, item in zip(users, items):
            user = str(user)
            item = str(item)
            if item not in item_to_index:
                continue
            user_index = user_to_index.setdefault(user, len(user_to_index))
            rows.append(user_index)
            cols.append(item_to_index[item])
    matrix = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(len(user_to_index), len(item_to_index)),
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    if min_user_degree > 1:
        degrees = np.asarray(matrix.sum(axis=1)).ravel()
        keep_users = np.flatnonzero(degrees >= min_user_degree)
        matrix = matrix[keep_users]
    remapped_users = {f"u{index}": index for index in range(matrix.shape[0])}
    return matrix, remapped_users


def build_collaborative_embeddings(cfg: Config) -> dict[str, Any]:
    catalog_path = cfg.paths.artifact("catalog.parquet")
    interactions_path = cfg.paths.artifact("interactions.parquet")
    for path in [catalog_path, interactions_path]:
        if not path.exists():
            raise FileNotFoundError(f"required preprocessing artifact not found: {path}")

    item_ids = _load_catalog_ids(catalog_path)
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    matrix, _ = build_user_item_csr(
        interactions_path,
        item_to_index=item_to_index,
        min_user_degree=int(cfg.collaborative.min_user_degree),
    )
    max_components = max(2, min(matrix.shape[0], matrix.shape[1]) - 1)
    n_components = min(int(cfg.collaborative.svd_components), max_components)
    svd = TruncatedSVD(
        n_components=n_components,
        algorithm="randomized",
        random_state=int(cfg.collaborative.random_state),
    )
    item_embeddings = svd.fit_transform(matrix.transpose()).astype(np.float32)
    if cfg.collaborative.normalize:
        item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True).clip(min=1e-12)

    output_path = cfg.paths.artifact("collab_embeddings.npy")
    ensure_parent(output_path)
    np.save(output_path, item_embeddings)
    write_json(cfg.paths.artifact("collab_item_ids.json"), item_ids)
    metadata = {
        "items": int(item_embeddings.shape[0]),
        "dim": int(item_embeddings.shape[1]),
        "requested_dim": int(cfg.collaborative.dim),
        "user_item_rows": int(matrix.shape[0]),
        "interactions_sha256": sha256_file(interactions_path),
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
    }
    write_json(cfg.paths.artifact("collab_embeddings_meta.json"), metadata)
    print(f"[collab] saved {item_embeddings.shape} to {output_path}")
    return metadata
