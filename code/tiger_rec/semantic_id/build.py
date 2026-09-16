"""Train the multi-view RQ-VAE and materialize collision-free Semantic IDs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, TensorDataset

from tiger_rec.config import Config
from tiger_rec.semantic_id.collisions import resolve_semantic_id_collisions
from tiger_rec.semantic_id.rqvae import MultiViewRQVAE, codebook_metrics, inverse_frequency_weights, kmeans_initialize_codebooks, load_rqvae, reset_dead_codebooks, reset_dead_codebooks_error_aware, save_rqvae
from tiger_rec.semantic_id.token_space import TokenSpace
from tiger_rec.semantic_id.trie import SIDTrie
from tiger_rec.utils import resolve_device, set_seed, write_json


def _catalog_items(path: str | Path) -> tuple[list[str], np.ndarray]:
    table = pq.read_table(path, columns=["item_id", "train_count"])
    item_ids = [str(value) for value in table.column("item_id").to_pylist()]
    counts = np.asarray(table.column("train_count").to_numpy(zero_copy_only=False), dtype=np.int64)
    return item_ids, counts


def load_aligned_embeddings(cfg: Config) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    item_ids, counts = _catalog_items(cfg.paths.artifact("catalog.parquet"))
    with cfg.paths.artifact("text_item_ids.json").open("r", encoding="utf-8") as f:
        text_ids = [str(value) for value in json.load(f)]
    with cfg.paths.artifact("collab_item_ids.json").open("r", encoding="utf-8") as f:
        collab_ids = [str(value) for value in json.load(f)]
    text_matrix = np.load(cfg.paths.artifact("text_embeddings.npy"))
    collab_matrix = np.load(cfg.paths.artifact("collab_embeddings.npy"))
    if len(item_ids) != len(text_ids) or item_ids != text_ids:
        raise ValueError("catalog and text embedding item order differ; rerun encode_items")
    if len(collab_ids) != len(item_ids):
        raise ValueError("collaborative embedding catalog size differs from catalog")
    collab_index = {item_id: index for index, item_id in enumerate(collab_ids)}
    aligned_collab = np.stack([collab_matrix[collab_index[item_id]] for item_id in item_ids], axis=0)
    return item_ids, text_matrix.astype(np.float32), aligned_collab.astype(np.float32), counts

def train_rqvae(cfg: Config) -> dict[str, Any]:
    set_seed(cfg.sid.seed)
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    item_ids, text, collab, counts = load_aligned_embeddings(cfg)
    weights = inverse_frequency_weights(counts, cfg.sid.inverse_frequency_power, cfg.sid.inverse_frequency_clip)
    model = MultiViewRQVAE(
        text_dim=text.shape[1], collab_dim=collab.shape[1], hidden_dim=cfg.sid.hidden_dim,
        latent_dim=cfg.sid.latent_dim, num_levels=cfg.sid.num_levels,
        codebook_size=cfg.sid.codebook_size, fusion_type=cfg.sid.fusion_type,
        gate_hidden_dim=cfg.sid.gate_hidden_dim, dropout=cfg.sid.dropout,
        temperature=cfg.sid.temperature,
        usage_ema_decay=cfg.sid.usage_ema_decay,
    ).to(device)
    text_tensor = torch.from_numpy(text)
    collab_tensor = torch.from_numpy(collab)
    weight_tensor = torch.from_numpy(weights)
    loader = DataLoader(TensorDataset(text_tensor, collab_tensor, weight_tensor), batch_size=int(cfg.sid.batch_size), shuffle=True)

    if cfg.sid.kmeans_init and len(item_ids) > 1:
        sample_size = min(len(item_ids), 100000)
        sample_index = np.linspace(0, len(item_ids) - 1, sample_size, dtype=np.int64)
        kmeans_initialize_codebooks(
            model, text_tensor[sample_index].to(device),
            collab_tensor[sample_index].to(device) if cfg.sid.fusion_type != "content_only" else None,
            max_iter=cfg.sid.kmeans_max_iter, seed=cfg.sid.seed,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.sid.learning_rate, weight_decay=cfg.sid.weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.sid.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "reconstruction": 0.0, "commitment": 0.0, "codebook": 0.0, "usage": 0.0, "hard_usage": 0.0, "joint_collision": 0.0}
        seen = 0
        code_usage = [torch.zeros(int(cfg.sid.codebook_size), dtype=torch.long, device=device) for _ in range(int(cfg.sid.num_levels))]
        for batch_text, batch_collab, batch_weight in loader:
            batch_text = batch_text.to(device)
            batch_collab = batch_collab.to(device) if cfg.sid.fusion_type != "content_only" else None
            batch_weight = batch_weight.to(device)
            output = model(batch_text, batch_collab, sample_weights=batch_weight, topk=1)
            quantized = output["quantized"]
            reconstruction = output["reconstruction_loss"]
            assert not isinstance(quantized, torch.Tensor)
            assert isinstance(reconstruction, torch.Tensor)
            loss = (reconstruction + cfg.sid.commitment_weight * quantized.commitment_loss
                    + cfg.sid.codebook_weight * quantized.codebook_loss
                    + cfg.sid.usage_weight * quantized.usage_loss
                    + cfg.sid.hard_usage_weight * quantized.hard_usage_loss
                    + cfg.sid.joint_collision_weight * quantized.joint_collision_loss
                    - cfg.sid.entropy_weight * quantized.entropy)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_size = batch_text.shape[0]
            totals["loss"] += float(loss.detach().cpu()) * batch_size
            totals["reconstruction"] += float(reconstruction.detach().cpu()) * batch_size
            totals["commitment"] += float(quantized.commitment_loss.detach().cpu()) * batch_size
            totals["codebook"] += float(quantized.codebook_loss.detach().cpu()) * batch_size
            totals["usage"] += float(quantized.usage_loss.detach().cpu()) * batch_size
            totals["hard_usage"] += float(quantized.hard_usage_loss.detach().cpu()) * batch_size
            totals["joint_collision"] += float(quantized.joint_collision_loss.detach().cpu()) * batch_size
            for level in range(int(cfg.sid.num_levels)):
                code_usage[level] += torch.bincount(quantized.codes[:, level], minlength=int(cfg.sid.codebook_size))
            seen += batch_size
        row = {key: value / max(seen, 1) for key, value in totals.items()}
        row["epoch"] = float(epoch)
        for level, usage in enumerate(code_usage):
            row[f"epoch_utilization_level_{level}"] = float((usage > 0).float().mean().detach().cpu())
            row[f"epoch_dead_codes_level_{level}"] = float((usage == 0).sum().detach().cpu())
        if cfg.sid.dead_code_reset and epoch % max(1, int(cfg.sid.dead_code_reset_interval)) == 0:
            model.eval()
            with torch.no_grad():
                if cfg.sid.dead_code_reset_full_data:
                    reset_text = text_tensor.to(device)
                    reset_collab = (
                        collab_tensor.to(device)
                        if cfg.sid.fusion_type != "content_only" else None
                    )
                    reset_latent = model.encoder(model.fusion(reset_text, reset_collab))
                else:
                    reset_latent = model.encoder(model.fusion(
                        batch_text.to(device),
                        batch_collab.to(device) if batch_collab is not None else None,
                    ))
                if cfg.sid.dead_code_reset_error_aware:
                    reset_counts = reset_dead_codebooks_error_aware(
                        model,
                        reset_latent,
                        threshold=float(cfg.sid.dead_code_threshold),
                        noise=float(cfg.sid.dead_code_noise),
                    )
                    row["dead_codes_reset"] = float(reset_counts["total"])
                    for level in range(int(cfg.sid.num_levels)):
                        row[f"dead_codes_reset_level_{level}"] = float(
                            reset_counts.get(f"level_{level}", 0)
                        )
                else:
                    reset_usage_ema = (
                        model.quantizer.usage_ema
                        if str(cfg.sid.dead_code_reset_usage).lower() == "ema" else None
                    )
                    row["dead_codes_reset"] = float(reset_dead_codebooks(
                        model,
                        reset_latent,
                        usage_counts=code_usage,
                        threshold=float(cfg.sid.dead_code_threshold),
                        noise=float(cfg.sid.dead_code_noise),
                        usage_ema=reset_usage_ema,
                    ))
            model.train()
        history.append(row)
        print(f"[RQ-VAE] epoch={epoch:03d} " + " ".join(f"{k}={v:.6f}" for k, v in row.items() if k != "epoch"))
    metadata = {
        "text_dim": int(text.shape[1]), "collab_dim": int(collab.shape[1]),
        "hidden_dim": int(cfg.sid.hidden_dim), "latent_dim": int(cfg.sid.latent_dim),
        "num_levels": int(cfg.sid.num_levels), "codebook_size": int(cfg.sid.codebook_size),
        "fusion_type": cfg.sid.fusion_type, "gate_hidden_dim": int(cfg.sid.gate_hidden_dim),
        "dropout": float(cfg.sid.dropout), "temperature": float(cfg.sid.temperature),
        "usage_ema_decay": float(cfg.sid.usage_ema_decay),
        "dead_code_threshold": float(cfg.sid.dead_code_threshold),
        "joint_collision_weight": float(cfg.sid.joint_collision_weight),
        "dead_code_reset_full_data": bool(cfg.sid.dead_code_reset_full_data),
        "dead_code_reset_error_aware": bool(cfg.sid.dead_code_reset_error_aware),
        "dead_code_reset_usage": str(cfg.sid.dead_code_reset_usage),
        "items": len(item_ids), "history": history,
    }
    save_rqvae(model, str(cfg.paths.artifact("rqvae.pt")), metadata)
    write_json(cfg.paths.artifact("rqvae_training.json"), metadata)
    return metadata

@torch.no_grad()
def encode_all(cfg: Config) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
    item_ids, text, collab, counts = load_aligned_embeddings(cfg)
    model, metadata = load_rqvae(str(cfg.paths.artifact("rqvae.pt")), map_location=device)
    model = model.to(device).eval()
    codes: list[np.ndarray] = []
    topk_codes: list[np.ndarray] = []
    for start in range(0, len(item_ids), int(cfg.sid.batch_size)):
        end = min(start + int(cfg.sid.batch_size), len(item_ids))
        batch_text = torch.from_numpy(text[start:end]).to(device)
        batch_collab = torch.from_numpy(collab[start:end]).to(device)
        if metadata["fusion_type"] == "content_only":
            batch_collab = None
        output = model.encode_codes(batch_text, batch_collab, topk=cfg.sid.topk_for_conflict)
        codes.append(output.codes.cpu().numpy())
        topk_codes.append(output.topk_codes.cpu().numpy())
    return item_ids, np.concatenate(codes, axis=0), np.concatenate(topk_codes, axis=0), counts


def build_semantic_id_artifacts(cfg: Config) -> dict[str, Any]:
    item_ids, base_codes, topk_codes, counts = encode_all(cfg)
    token_space = TokenSpace(
        num_levels=int(cfg.sid.num_levels),
        codebook_size=int(cfg.sid.codebook_size),
        max_conflict_tokens=int(cfg.sid.max_conflict_tokens),
        use_behavior_tokens=bool(cfg.generator.use_behavior_tokens),
        use_time_buckets=bool(cfg.generator.use_time_buckets),
        use_user_tokens=bool(cfg.generator.use_user_tokens),
        num_user_buckets=int(cfg.generator.num_user_buckets),
        num_time_buckets=int(cfg.generator.num_time_buckets),
    )
    if int(cfg.generator.max_target_tokens) <= token_space.max_sid_length:
        raise ValueError(
            "generator.max_target_tokens must be at least max_sid_length + 1 "
            "(the extra token is EOS); "
            f"got {cfg.generator.max_target_tokens} and max_sid_length={token_space.max_sid_length}"
        )
    if int(cfg.generator.max_decode_steps) < token_space.max_sid_length:
        raise ValueError(
            "generator.max_decode_steps must be at least max_sid_length; "
            f"got {cfg.generator.max_decode_steps} and max_sid_length={token_space.max_sid_length}"
        )
    np.save(cfg.paths.artifact("semantic_base_codes.npy"), base_codes)
    np.save(cfg.paths.artifact("semantic_topk_codes.npy"), topk_codes.astype(np.int32, copy=False))
    write_json(cfg.paths.artifact("semantic_item_ids.json"), item_ids)

    assignments, collision_report = resolve_semantic_id_collisions(
        item_ids=item_ids,
        base_codes=base_codes,
        token_space=token_space,
        train_counts=counts,
        topk_codes=topk_codes,
        max_suffix_tokens=cfg.sid.max_conflict_tokens,
        length_mode=cfg.sid.length_mode,
        reallocation_mode=cfg.sid.conflict_reallocation_mode,
    )
    sid_map = {item_id: assignment.to_dict() for item_id, assignment in assignments.items()}
    write_json(cfg.paths.artifact("semantic_ids.json"), sid_map)
    write_json(cfg.paths.artifact("sid_token_space.json"), token_space.to_dict())
    token_rows = [assignments[item_id].token_ids for item_id in item_ids]
    max_token_len = max(len(row) for row in token_rows)
    padded_tokens = np.full((len(token_rows), max_token_len), -1, dtype=np.int64)
    for row_index, row in enumerate(token_rows):
        padded_tokens[row_index, :len(row)] = row
    np.save(cfg.paths.artifact("semantic_token_ids.npy"), padded_tokens)
    metrics = codebook_metrics(base_codes, cfg.sid.codebook_size)
    report = {
        **collision_report.__dict__,
        **metrics,
        "token_vocab_size": token_space.vocab_size,
        "token_space": token_space.to_dict(),
    }
    write_json(cfg.paths.artifact("semantic_id_report.json"), report)
    print(f"[SID] mode={collision_report.length_mode} items={len(item_ids):,} "
          f"collision_rate={collision_report.base_collision_rate:.4f} "
          f"reallocated={collision_report.semantic_reallocated:,} suffix={collision_report.suffix_resolved:,} "
          f"mean_length={collision_report.mean_sid_length:.3f} elapsed={collision_report.resolution_seconds:.2f}s")
    return report


def load_semantic_id_map(cfg: Config) -> dict[str, dict[str, Any]]:
    with cfg.paths.artifact("semantic_ids.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_token_space(cfg: Config) -> TokenSpace:
    with cfg.paths.artifact("sid_token_space.json").open("r", encoding="utf-8") as f:
        return TokenSpace.from_dict(json.load(f))


def build_trie_from_semantic_ids(semantic_ids: dict[str, dict[str, Any]]) -> SIDTrie:
    trie = SIDTrie()
    for item_id, row in semantic_ids.items():
        trie.insert(row["token_ids"], item_id)
    return trie


def load_trie(cfg: Config) -> SIDTrie:
    return build_trie_from_semantic_ids(load_semantic_id_map(cfg))
