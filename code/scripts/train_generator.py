"""Train the TIGER T5-style generator on Semantic ID sequences."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiger_rec.config import load_config
from tiger_rec.data.dataset import AmazonSequenceDataset, collate_sequence
from tiger_rec.models.generator import build_t5_model, save_generator
from tiger_rec.semantic_id.build import load_token_space
from tiger_rec.utils import resolve_device, set_seed, write_json


def run_epoch(model, loader, optimizer, scheduler, device, gradient_clip: float) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.detach().cpu())
        steps += 1
    return total_loss / max(steps, 1)


@torch.inference_mode()
def validate(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    for batch in loader:
        loss = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        ).loss
        total_loss += float(loss.detach().cpu())
        steps += 1
    return total_loss / max(steps, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = resolve_device("auto")
    train_dataset = AmazonSequenceDataset(cfg, "train")
    valid_dataset = AmazonSequenceDataset(cfg, "valid")
    if len(train_dataset) == 0:
        raise RuntimeError("training dataset is empty")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.generator.batch_size),
        shuffle=True,
        num_workers=int(cfg.generator.num_workers),
        collate_fn=lambda batch: collate_sequence(batch),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(cfg.generator.batch_size),
        shuffle=False,
        num_workers=int(cfg.generator.num_workers),
        collate_fn=lambda batch: collate_sequence(batch),
    )
    token_space = load_token_space(cfg)
    model = build_t5_model(token_space, cfg.generator).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.generator.learning_rate),
        weight_decay=float(cfg.generator.weight_decay),
    )
    total_steps = max(1, len(train_loader) * int(cfg.generator.epochs))
    warmup = min(int(cfg.generator.warmup_steps), max(1, total_steps // 10))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(warmup, 1)),
    )
    best_loss = math.inf
    stale = 0
    history = []
    checkpoint_path = cfg.paths.artifact(cfg.generator.checkpoint_name)
    for epoch in range(1, int(cfg.generator.epochs) + 1):
        started = time.perf_counter()
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, device, cfg.generator.gradient_clip)
        valid_loss = validate(model, valid_loader, device) if len(valid_dataset) else train_loss
        elapsed = time.perf_counter() - started
        row = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "elapsed_sec": elapsed}
        history.append(row)
        print(f"[generator] epoch={epoch:03d} train={train_loss:.4f} valid={valid_loss:.4f} time={elapsed:.1f}s")
        if valid_loss < best_loss:
            best_loss = valid_loss
            stale = 0
            save_generator(
                model,
                str(checkpoint_path),
                {
                    "architecture": cfg.generator.architecture,
                    "token_space": token_space.to_dict(),
                    "best_valid_loss": best_loss,
                    "history": history,
                    "constrained_decoder_recommended": True,
                },
            )
        else:
            stale += 1
            if stale >= int(cfg.generator.early_stopping_patience):
                print("[generator] early stopping")
                break
    write_json(cfg.paths.artifact("generator_training.json"), {"history": history, "best_valid_loss": best_loss})
    print(f"[generator] checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
