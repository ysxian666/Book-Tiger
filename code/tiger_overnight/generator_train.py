"""Training and checkpoint selection for overnight generator variants."""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tiger_rec.semantic_id.rqvae import MultiViewRQVAE, load_rqvae

from .config import artifact_path, ensure_run_dirs
from .decoding import beam_search
from .generator_data import SequenceBatch, TigerSequenceDataset, collate_sequences, load_token_space
from .generator_model import TigerGenerator


VARIANT_SPECS = {
    "o1_l2_lite": {"sid_mode": "baseline", "soft_prefix": False, "artifact_name": "l2_lite"},
    "o2_bcgsid": {"sid_mode": "bcgsid", "soft_prefix": False, "artifact_name": "bcgsid"},
    "o3_ucg": {"sid_mode": "baseline", "soft_prefix": True, "artifact_name": "ucg"},
    "o4_bcgsid_ucg": {"sid_mode": "bcgsid", "soft_prefix": True, "artifact_name": "bcgsid_ucg"},
    "o5_joint": {"sid_mode": "bcgsid", "soft_prefix": True, "artifact_name": "joint", "initial_checkpoint": "bcgsid_ucg"},
    "c1_bcgsid_shuffled": {"sid_mode": "bcgsid_shuffled", "soft_prefix": False, "artifact_name": "bcgsid_shuffled"},
    "c2_bcgsid_hard": {"sid_mode": "bcgsid_hard", "soft_prefix": False, "artifact_name": "bcgsid_hard"},
    "c3_ucg_shuffled": {"sid_mode": "baseline", "soft_prefix": True, "artifact_name": "ucg_shuffled", "shuffle_user_context": True},
    "c4_bcgsid_hard_ucg": {"sid_mode": "bcgsid_hard", "soft_prefix": True, "artifact_name": "bcgsid_hard_ucg"},
}


CONTROL_ORDER = ["c1_bcgsid_shuffled", "c2_bcgsid_hard", "c3_ucg_shuffled", "c4_bcgsid_hard_ucg"]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _to_device(batch: SequenceBatch, device: torch.device) -> SequenceBatch:
    batch.input_ids = batch.input_ids.to(device, non_blocking=True)
    batch.attention_mask = batch.attention_mask.to(device, non_blocking=True)
    batch.labels = batch.labels.to(device, non_blocking=True)
    batch.long_context = batch.long_context.to(device, non_blocking=True)
    batch.short_context = batch.short_context.to(device, non_blocking=True)
    batch.target_index = batch.target_index.to(device, non_blocking=True)
    batch.target_text = batch.target_text.to(device, non_blocking=True)
    batch.target_collab = batch.target_collab.to(device, non_blocking=True)
    return batch


def _model_inputs(batch: SequenceBatch) -> dict[str, torch.Tensor]:
    return {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "labels": batch.labels,
        "long_context": batch.long_context,
        "short_context": batch.short_context,
    }


def _valid_loss(model: TigerGenerator, loader: DataLoader, device: torch.device, max_batches: int, amp: bool) -> float:
    model.eval()
    total = 0.0
    seen = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= int(max_batches):
                break
            batch = _to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                output = model(**_model_inputs(batch))
            size = int(batch.labels.shape[0])
            total += float(output["loss"].detach().cpu()) * size
            seen += size
    model.train()
    return total / max(seen, 1)


@torch.inference_mode()
def _quick_valid_recall(
    model: TigerGenerator,
    dataset: TigerSequenceDataset,
    token_space,
    device: torch.device,
    user_count: int,
    beam: int,
    amp: bool,
) -> float:
    if user_count <= 0:
        return 0.0
    limit = min(int(user_count), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, limit, dtype=np.int64)
    rows = [dataset[int(index)] for index in indices]
    batch = collate_sequences(rows)
    batch = _to_device(batch, device)
    token_to_item = {tuple(int(value) for value in row["token_ids"]): item_id for item_id, row in dataset.sid_map.items()}
    history_ids = [
        {dataset.item_ids[int(index)] for index in row["history_indices"].tolist()}
        for row in rows
    ]
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        decoded = beam_search(
            model,
            batch.input_ids,
            batch.attention_mask,
            batch.long_context,
            batch.short_context,
            token_space,
            token_to_item,
            history_ids,
            beam_size=int(beam),
            constrained_trie=None,
            max_steps=int(token_space.max_sid_length),
        )
    hits = 0
    for index, output in enumerate(decoded):
        if rows[index]["target_item_id"] in output.item_ids[:20]:
            hits += 1
    model.train()
    return hits / max(len(decoded), 1)


def _sequence_log_scores(
    model: TigerGenerator,
    encoder_hidden: torch.Tensor,
    encoder_mask: torch.Tensor,
    sequences: torch.Tensor,
) -> torch.Tensor:
    decoder_input = torch.cat([
        torch.full((sequences.shape[0], 1), model.t5.config.decoder_start_token_id, dtype=torch.long, device=sequences.device),
        sequences[:, :-1],
    ], dim=1)
    logits = model.decode_logits(encoder_hidden, encoder_mask, decoder_input).float()
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(2, sequences.unsqueeze(-1)).squeeze(-1).sum(dim=1)


def _ranking_loss(model: TigerGenerator, batch: SequenceBatch, max_items: int = 32) -> torch.Tensor:
    count = min(int(batch.labels.shape[0]), int(max_items))
    if count < 2:
        return batch.labels.new_zeros((), dtype=torch.float32)
    input_ids = batch.input_ids[:count]
    attention = batch.attention_mask[:count]
    long_context = batch.long_context[:count]
    short_context = batch.short_context[:count]
    targets = batch.labels[:count]
    encoder_hidden, encoder_mask = model.encode(input_ids, attention, long_context, short_context)
    positive = _sequence_log_scores(model, encoder_hidden, encoder_mask, targets)
    negative_indices = torch.roll(torch.arange(count, device=targets.device), shifts=1)
    negatives = targets[negative_indices]
    negative_flat = negatives.unsqueeze(0).expand(count, -1, -1).reshape(count * count, -1)
    hidden_flat = encoder_hidden.repeat_interleave(count, dim=0)
    mask_flat = encoder_mask.repeat_interleave(count, dim=0)
    negative_scores = _sequence_log_scores(model, hidden_flat, mask_flat, negative_flat).reshape(count, count)
    logits = negative_scores
    labels = torch.arange(count, device=targets.device)
    logits[torch.arange(count, device=targets.device), labels] = positive
    return F.cross_entropy(logits, labels)


def train_generator(config: dict[str, Any], variant: str, force: bool = False, max_steps_override: int | None = None) -> dict[str, Any]:
    if variant not in VARIANT_SPECS:
        raise KeyError(f"unknown variant: {variant}")
    ensure_run_dirs(config)
    spec = VARIANT_SPECS[variant]
    artifact_name = str(spec["artifact_name"])
    output_path = artifact_path(config, "generators", f"{artifact_name}.pt")
    metrics_path = artifact_path(config, "generators", f"{artifact_name}_training.json")
    if output_path.exists() and metrics_path.exists() and not force:
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["generator"]["bf16"]) and device.type == "cuda"
    token_space = load_token_space(config)
    shuffle_user_context = bool(spec.get("shuffle_user_context", False))
    train_dataset = TigerSequenceDataset(
        config,
        sid_mode=spec["sid_mode"],
        split="train",
        use_soft_prefix=bool(spec["soft_prefix"]),
        shuffle_user_context=shuffle_user_context,
    )
    valid_dataset = TigerSequenceDataset(
        config,
        sid_mode=spec["sid_mode"],
        split="valid",
        use_soft_prefix=bool(spec["soft_prefix"]),
        sample_limit=4096,
        shuffle_user_context=shuffle_user_context,
    )
    context_dim = int(train_dataset.item_latents.shape[1])
    model = TigerGenerator(
        token_space=token_space,
        generator_config=config["generator"],
        context_dim=context_dim,
        user_context_config=config["user_context"] if spec["soft_prefix"] else None,
    ).to(device)
    initial_checkpoint = spec.get("initial_checkpoint")
    if initial_checkpoint:
        checkpoint_path = artifact_path(config, "generators", f"{initial_checkpoint}.pt")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing initial checkpoint for {variant}: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

    num_workers = int(config["generator"]["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["generator"]["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_sequences,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config["generator"]["batch_size"]),
        shuffle=False,
        num_workers=min(2, num_workers),
        collate_fn=collate_sequences,
        pin_memory=False,
    )
    is_joint = variant == "o5_joint"
    rqvae: MultiViewRQVAE | None = None
    optimizer_params: list[dict[str, Any]] = [{
        "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
        "lr": float(config["joint_training"]["generator_lr"] if is_joint else config["generator"]["learning_rate"]),
    }]
    if is_joint:
        rqvae, _metadata = load_rqvae(str(artifact_path(config, "rqvae", "rqvae.pt")), map_location=device)
        rqvae = rqvae.to(device)
        if bool(config["joint_training"]["freeze_codebooks"]):
            for codebook in rqvae.quantizer.codebooks:
                codebook.requires_grad = False
        optimizer_params.append({
            "params": [parameter for parameter in rqvae.parameters() if parameter.requires_grad],
            "lr": float(config["joint_training"]["rqvae_lr"]),
        })
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=float(config["generator"]["weight_decay"]))
    total_steps = int(max_steps_override or (config["joint_training"]["joint_steps"] if is_joint else config["generator"]["train_steps"]))
    warmup = int(config["generator"]["warmup_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / max(warmup, 1)),
    )
    model.train()
    if rqvae is not None:
        rqvae.train()
    iterator = itertools.cycle(train_loader)
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_valid_loss = float("inf")
    best_step = 0
    started = time.perf_counter()
    eval_every = int(config["generator"]["eval_every_steps"])
    checkpoint_every = int(config["generator"]["checkpoint_interval_steps"])
    latest_path = artifact_path(config, "generators", f"{artifact_name}_latest.pt")

    for step in range(1, total_steps + 1):
        if step > total_steps:
            break
        if variant == "o1_l2_lite" and total_steps > 20000 and step >= 20000 and (time.perf_counter() - started) / 60.0 > 70.0:
            total_steps = 20000
            print(f"[{variant}] 70-minute generator budget exceeded; stopping at 20000 uniform steps")
            break
        batch = _to_device(next(iterator), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            output = model(**_model_inputs(batch))
            next_sid_loss = output["loss"]
            prefix_loss = F.cross_entropy(
                output["logits"][:, :3, :].float().reshape(-1, output["logits"].shape[-1]),
                batch.labels[:, :3].reshape(-1),
                ignore_index=0,
            )
            ranking_loss = _ranking_loss(model, batch) if is_joint else batch.labels.new_zeros((), dtype=torch.float32)
            alignment_loss = (
                model.context_alignment_loss(batch.long_context, batch.short_context, batch.long_context)
                if spec["soft_prefix"]
                else batch.labels.new_zeros((), dtype=torch.float32)
            )
            reconstruction_loss = batch.labels.new_zeros((), dtype=torch.float32)
            if is_joint:
                assert rqvae is not None
                rqvae_output = rqvae(batch.target_text, batch.target_collab, sample_weights=None, topk=1)
                reconstruction_loss = rqvae_output["reconstruction_loss"]
                target_latent = rqvae_output["latent"].detach()
                alignment_loss = model.context_alignment_loss(batch.long_context, batch.short_context, target_latent)
            joint_cfg = config["joint_training"]
            loss = (
                float(joint_cfg["lambda_next_sid"] if is_joint else 1.0) * next_sid_loss
                + float(joint_cfg["lambda_ranking"] if is_joint else 0.0) * ranking_loss
                + float(joint_cfg["lambda_alignment"] if is_joint else 0.0) * alignment_loss
                + float(joint_cfg["lambda_prefix"] if is_joint else 0.0) * prefix_loss
                + float(joint_cfg["lambda_recon"] if is_joint else 0.0) * reconstruction_loss
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + ([] if rqvae is None else list(rqvae.parameters())),
            float(config["generator"]["gradient_clip"]),
        )
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0:
            history.append({
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "next_sid_loss": float(next_sid_loss.detach().cpu()),
                "ranking_loss": float(ranking_loss.detach().cpu()),
                "alignment_loss": float(alignment_loss.detach().cpu()),
                "user_context_loss": float(alignment_loss.detach().cpu()),
                "prefix_loss": float(prefix_loss.detach().cpu()),
                "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "elapsed_sec": float(time.perf_counter() - started),
            })
        if step % checkpoint_every == 0 or step == total_steps:
            model.save(str(latest_path), {
                "variant": variant,
                "sid_mode": spec["sid_mode"],
                "soft_prefix": bool(spec["soft_prefix"]),
                "shuffle_user_context": shuffle_user_context,
                "token_space": token_space.to_dict(),
                "item_ids": train_dataset.item_ids,
                "step": step,
            })
        if step % eval_every == 0 or step == total_steps:
            valid_loss = _valid_loss(model, valid_loader, device, max_batches=32, amp=amp)
            valid_recall20 = _quick_valid_recall(
                model,
                valid_dataset,
                token_space,
                device,
                user_count=min(128, len(valid_dataset)),
                beam=min(5, int(config["generator"]["num_beams"])),
                amp=amp,
            )
            score = valid_loss - 0.05 * valid_recall20
            record = {
                "step": step,
                "valid_loss": valid_loss,
                "valid_recall@20_quick": valid_recall20,
                "selection_score": score,
                "elapsed_sec": float(time.perf_counter() - started),
            }
            history.append(record)
            print(
                f"[{variant}] step={step}/{total_steps} train={float(loss.detach().cpu()):.4f} "
                f"valid={valid_loss:.4f} quickR20={valid_recall20:.4f}"
            )
            if score < best_score:
                best_score = score
                best_valid_loss = valid_loss
                best_step = step
                model.save(str(output_path), {
                    "variant": variant,
                    "sid_mode": spec["sid_mode"],
                    "soft_prefix": bool(spec["soft_prefix"]),
                    "shuffle_user_context": shuffle_user_context,
                    "token_space": token_space.to_dict(),
                    "item_ids": train_dataset.item_ids,
                    "best_step": step,
                    "valid_loss": valid_loss,
                    "valid_recall@20_quick": valid_recall20,
                })
    if not output_path.exists():
        model.save(str(output_path), {
            "variant": variant,
            "sid_mode": spec["sid_mode"],
            "soft_prefix": bool(spec["soft_prefix"]),
            "shuffle_user_context": shuffle_user_context,
            "token_space": token_space.to_dict(),
            "item_ids": train_dataset.item_ids,
            "best_step": total_steps,
        })
    elapsed_sec = float(time.perf_counter() - started)
    last_training_record = next((row for row in reversed(history) if "train_loss" in row), {})
    gpu_memory_peak = float(torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0
    result = {
        "variant": variant,
        "sid_mode": spec["sid_mode"],
        "soft_prefix": bool(spec["soft_prefix"]),
        "shuffle_user_context": shuffle_user_context,
        "train_examples": len(train_dataset),
        "valid_examples": len(valid_dataset),
        "total_steps": total_steps,
        "best_step": best_step or total_steps,
        "checkpoint_step": best_step or total_steps,
        "best_selection_score": best_score,
        "best_valid_loss": best_valid_loss,
        "train_loss": float(last_training_record.get("train_loss", 0.0)),
        "ranking_loss": float(last_training_record.get("ranking_loss", 0.0)),
        "prefix_loss": float(last_training_record.get("prefix_loss", 0.0)),
        "user_context_loss": float(last_training_record.get("user_context_loss", 0.0)),
        "steps_per_sec": total_steps / max(elapsed_sec, 1e-9),
        "GPU_memory_peak": gpu_memory_peak,
        "elapsed_sec": elapsed_sec,
        "history": history,
        "checkpoint": str(output_path),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }
    _write_json(metrics_path, result)
    return result
