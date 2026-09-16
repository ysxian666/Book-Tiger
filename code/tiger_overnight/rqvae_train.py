"""RQ-VAE training and checkpointing for the overnight protocol."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tiger_rec.semantic_id.rqvae import (
    MultiViewRQVAE,
    inverse_frequency_weights,
    kmeans_initialize_codebooks,
    load_rqvae,
    reset_dead_codebooks_error_aware,
    save_rqvae,
)

from .config import artifact_path, data_path, ensure_run_dirs


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_aligned_item_features(config: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    catalog = pq.read_table(data_path(config, "catalog.parquet"), columns=["item_id", "train_count"])
    item_ids = [str(value) for value in catalog.column("item_id").to_pylist()]
    counts = np.asarray(catalog.column("train_count").to_numpy(zero_copy_only=False), dtype=np.int64)
    text_ids = [str(value) for value in _load_json(data_path(config, "text_item_ids.json"))]
    collab_ids = [str(value) for value in _load_json(data_path(config, "collab_item_ids.json"))]
    text = np.load(data_path(config, "text_embeddings.npy")).astype(np.float32)
    collab = np.load(data_path(config, "collab_embeddings.npy")).astype(np.float32)
    if text_ids != item_ids or collab_ids != item_ids:
        raise RuntimeError("feature item order does not match catalog; rebuild features")
    return item_ids, text, collab, counts


def _latent(model: MultiViewRQVAE, text: torch.Tensor, collab: torch.Tensor) -> torch.Tensor:
    return model.encoder(model.fusion(text, collab))


def _usage_metrics(code_usage: list[torch.Tensor], codebook_size: int) -> dict[str, float]:
    utilization: dict[str, float] = {}
    perplexity: dict[str, float] = {}
    for level, counts in enumerate(code_usage):
        values = counts.detach().float().cpu()
        total = values.sum().clamp_min(1.0)
        probability = values / total
        utilization[f"level_{level}"] = float((values > 0).float().mean())
        perplexity[f"level_{level}"] = float(torch.exp(-(probability * (probability + 1e-12).log()).sum()))
    return {
        "mean_utilization": float(np.mean(list(utilization.values()))),
        "mean_perplexity": float(np.mean(list(perplexity.values()))),
        **{f"utilization_{key}": value for key, value in utilization.items()},
        **{f"perplexity_{key}": value for key, value in perplexity.items()},
    }


def train_rqvae(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    ensure_run_dirs(config)
    checkpoint_path = artifact_path(config, "rqvae", "rqvae.pt")
    training_path = artifact_path(config, "rqvae", "training.json")
    utilization_path = artifact_path(config, "rqvae", "utilization.json")
    if checkpoint_path.exists() and training_path.exists() and utilization_path.exists() and not force:
        return _load_json(training_path)

    item_ids, text_np, collab_np, counts = load_aligned_item_features(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sid_cfg = config["rqvae"]
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    model = MultiViewRQVAE(
        text_dim=int(text_np.shape[1]),
        collab_dim=int(collab_np.shape[1]),
        hidden_dim=int(sid_cfg["hidden_dim"]),
        latent_dim=int(sid_cfg["latent_dim"]),
        num_levels=int(sid_cfg["num_levels"]),
        codebook_size=int(sid_cfg["codebook_size"]),
        fusion_type="gate",
        gate_hidden_dim=int(sid_cfg["hidden_dim"]),
        dropout=float(sid_cfg["dropout"]),
        temperature=float(sid_cfg["temperature"]),
        usage_ema_decay=0.9,
    ).to(device)
    text = torch.from_numpy(text_np)
    collab = torch.from_numpy(collab_np)
    sample_weights = torch.from_numpy(
        inverse_frequency_weights(counts, float(sid_cfg["inverse_frequency_power"]), float(sid_cfg["inverse_frequency_clip"]))
    )
    if bool(sid_cfg["kmeans_init"]):
        sample_count = min(len(item_ids), 50000)
        sample_index = np.linspace(0, len(item_ids) - 1, sample_count, dtype=np.int64)
        kmeans_initialize_codebooks(
            model,
            text[sample_index].to(device),
            collab[sample_index].to(device),
            max_iter=int(sid_cfg["kmeans_max_iter"]),
            seed=int(config["seed"]),
        )
    loader = DataLoader(
        TensorDataset(text, collab, sample_weights),
        batch_size=int(sid_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(sid_cfg["learning_rate"]), weight_decay=float(sid_cfg["weight_decay"]))
    amp_enabled = device.type == "cuda" and bool(config["generator"].get("bf16", True))
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    requested_epochs = int(sid_cfg["epochs"])
    effective_epochs = requested_epochs
    for epoch in range(1, effective_epochs + 1):
        model.train()
        totals = {"loss": 0.0, "reconstruction": 0.0, "commitment": 0.0, "codebook": 0.0, "usage": 0.0, "hard_usage": 0.0, "entropy": 0.0, "diversity": 0.0}
        seen = 0
        code_usage = [torch.zeros(int(sid_cfg["codebook_size"]), dtype=torch.long, device=device) for _ in range(int(sid_cfg["num_levels"]))]
        for batch_text, batch_collab, batch_weight in loader:
            batch_text = batch_text.to(device, non_blocking=True)
            batch_collab = batch_collab.to(device, non_blocking=True)
            batch_weight = batch_weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                output = model(batch_text, batch_collab, sample_weights=batch_weight, topk=1)
                quantized = output["quantized"]
                assert not isinstance(quantized, torch.Tensor)
                fused_std = output["fused"].float().std(dim=0, unbiased=False).mean()
                latent_std = output["latent"].float().std(dim=0, unbiased=False).mean()
                diversity_loss = (
                    torch.relu(float(sid_cfg["fused_std_floor"]) - fused_std)
                    + torch.relu(float(sid_cfg["latent_std_floor"]) - latent_std)
                )
                loss = (
                    output["reconstruction_loss"]
                    + float(sid_cfg["commitment_weight"]) * quantized.commitment_loss
                    + float(sid_cfg["codebook_weight"]) * quantized.codebook_loss
                    + float(sid_cfg["usage_weight"]) * quantized.usage_loss
                    + float(sid_cfg["hard_usage_weight"]) * quantized.hard_usage_loss
                    - float(sid_cfg["entropy_weight"]) * quantized.entropy
                    + float(sid_cfg["diversity_weight"]) * diversity_loss
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for level in range(int(quantized.codes.shape[1])):
                code = quantized.codes[:, level]
                code_usage[level].scatter_add_(0, code.detach(), torch.ones_like(code, dtype=torch.long))
            batch_size = int(batch_text.shape[0])
            seen += batch_size
            values = {
                "loss": float(loss.detach().cpu()),
                "reconstruction": float(output["reconstruction_loss"].detach().cpu()),
                "commitment": float(quantized.commitment_loss.detach().cpu()),
                "codebook": float(quantized.codebook_loss.detach().cpu()),
                "usage": float(quantized.usage_loss.detach().cpu()),
                "hard_usage": float(quantized.hard_usage_loss.detach().cpu()),
                "entropy": float(quantized.entropy.detach().cpu()),
                "diversity": float(diversity_loss.detach().cpu()),
            }
            for key, value in values.items():
                totals[key] += value * batch_size
        metrics = _usage_metrics(code_usage, int(sid_cfg["codebook_size"]))
        reset_counts = {"total": 0}
        if bool(sid_cfg["dead_code_reset"]) and epoch % int(sid_cfg["dead_code_reset_interval"]) == 0:
            with torch.no_grad():
                latent = _latent(model, text.to(device), collab.to(device))
                reset_counts = reset_dead_codebooks_error_aware(
                    model,
                    latent,
                    threshold=float(sid_cfg["dead_code_threshold"]),
                    noise=float(sid_cfg["dead_code_noise"]),
                )
        record = {
            "epoch": epoch,
            **{key: value / max(seen, 1) for key, value in totals.items()},
            **metrics,
            "dead_codes_reset": int(reset_counts.get("total", 0)),
            "elapsed_sec": float(time.perf_counter() - started),
        }
        for level, counts in enumerate(code_usage):
            record[f"epoch_utilization_level_{level}"] = float((counts > 0).float().mean().detach().cpu())
            record[f"epoch_dead_codes_level_{level}"] = int((counts == 0).sum().detach().cpu())
        history.append(record)
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"[RQ-VAE] epoch={epoch}/{requested_epochs} loss={record['loss']:.5f} "
                f"recon={record['reconstruction']:.5f} util={record['mean_utilization']:.3f} reset={record['dead_codes_reset']}"
            )
        if epoch == 2000 and requested_epochs > 2000:
            elapsed_minutes = (time.perf_counter() - started) / 60.0
            projected_minutes = elapsed_minutes * requested_epochs / epoch
            if projected_minutes > float(sid_cfg["max_train_minutes"]):
                effective_epochs = 2000
                print(
                    f"[RQ-VAE] projected {projected_minutes:.1f} min exceeds "
                    f"{sid_cfg['max_train_minutes']} min; stopping at 2000 epochs"
                )
                break

    final_metrics = _usage_metrics(code_usage, int(sid_cfg["codebook_size"]))
    metadata = {
        **sid_cfg,
        "text_dim": int(text_np.shape[1]),
        "collab_dim": int(collab_np.shape[1]),
        "fusion_type": "gate",
        "gate_hidden_dim": int(sid_cfg["hidden_dim"]),
        "items": len(item_ids),
        "requested_epochs": requested_epochs,
        "completed_epochs": len(history),
    }
    save_rqvae(model, str(checkpoint_path), metadata)
    _write_json(training_path, {
        "config": metadata,
        "history": history,
        "final": final_metrics,
        "elapsed_sec": float(time.perf_counter() - started),
    })
    _write_json(utilization_path, {
        "final": final_metrics,
        "per_epoch": [
            {
                "epoch": row["epoch"],
                "mean_utilization": row["mean_utilization"],
                "mean_perplexity": row["mean_perplexity"],
            }
            for row in history
        ],
    })
    return _load_json(training_path)


def encode_items_with_rqvae(config: dict[str, Any], topk: int | None = None) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return item ids, hard codes, top-k codes, latent vectors, and fused vectors."""
    item_ids, text_np, collab_np, counts = load_aligned_item_features(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_rqvae(str(artifact_path(config, "rqvae", "rqvae.pt")), map_location=device)
    model = model.to(device).eval()
    batch_size = int(config["rqvae"]["batch_size"])
    k = int(topk or config["sid_allocation"]["top_codes_per_level"])
    codes: list[np.ndarray] = []
    topk_codes: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    fused_vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(item_ids), batch_size):
            end = min(start + batch_size, len(item_ids))
            batch_text = torch.from_numpy(text_np[start:end]).to(device)
            batch_collab = torch.from_numpy(collab_np[start:end]).to(device)
            fused = model.fusion(batch_text, batch_collab)
            latent = model.encoder(fused)
            output = model.quantizer(latent, topk=k)
            codes.append(output.codes.cpu().numpy())
            topk_codes.append(output.topk_codes.cpu().numpy())
            latents.append(latent.float().cpu().numpy())
            fused_vectors.append(fused.float().cpu().numpy())
    return (
        item_ids,
        np.concatenate(codes, axis=0),
        np.concatenate(topk_codes, axis=0),
        np.concatenate(latents, axis=0),
        np.concatenate(fused_vectors, axis=0),
    )
