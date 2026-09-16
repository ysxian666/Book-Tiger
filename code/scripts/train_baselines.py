"""Train two-tower, GRU4Rec, SASRec, and BERT4Rec baselines."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiger_rec.config import load_config
from tiger_rec.retrieval.base import load_eval_examples
from tiger_rec.models.baselines import BERT4Rec, GRU4Rec, SASRec, TwoTower, sampled_softmax_loss
from tiger_rec.utils import resolve_device, set_seed, write_json


class ItemPrefixDataset(Dataset):
    def __init__(self, cfg, item_to_model: dict[str, int]):
        table = pq.read_table(cfg.paths.artifact("user_sequences.parquet"), columns=["item_seq"])
        sequences = table.column("item_seq").to_pylist()
        self.examples: list[tuple[list[int], int]] = []
        for sequence in sequences:
            values = [item_to_model[str(item)] for item in sequence if str(item) in item_to_model]
            for target_pos in range(1, len(values)):
                # Share the full sequence object and slice only when a batch is built.
                self.examples.append((values, target_pos))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        sequence, target_pos = self.examples[index]
        return torch.tensor(sequence[:target_pos], dtype=torch.long), torch.tensor(sequence[target_pos], dtype=torch.long)


class EvalItemDataset(Dataset):
    def __init__(self, cfg, item_to_model: dict[str, int], split: str):
        self.examples = []
        for example in load_eval_examples(cfg, split=split):
            history = [item_to_model[item] for item in example.history_items if item in item_to_model]
            target = item_to_model.get(example.target_item)
            if history and target is not None:
                self.examples.append((history, target))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        sequence, target = self.examples[index]
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def collate(batch, max_len: int):
    max_length = max(1, min(max_len, max(len(sequence) for sequence, _ in batch)))
    sequence_tensor = torch.zeros((len(batch), max_length), dtype=torch.long)
    lengths = torch.ones(len(batch), dtype=torch.long)
    targets = torch.zeros(len(batch), dtype=torch.long)
    for row, (sequence, target) in enumerate(batch):
        values = sequence[-max_length:]
        sequence_tensor[row, :len(values)] = values
        lengths[row] = max(1, len(values))
        targets[row] = target
    return sequence_tensor, lengths, targets


def build_model(name: str, num_items: int, cfg):
    kwargs = {
        "dim": int(cfg.baselines.dim),
        "num_layers": int(cfg.baselines.num_layers),
        "num_heads": int(cfg.baselines.num_heads),
        "dropout": float(cfg.baselines.dropout),
    }
    if name == "twotower":
        return TwoTower(num_items=num_items, dim=kwargs["dim"], dropout=kwargs["dropout"])
    if name == "gru4rec":
        return GRU4Rec(num_items=num_items, **kwargs)
    if name == "sasrec":
        return SASRec(num_items=num_items, max_len=int(cfg.baselines.max_history_items) + 2, **kwargs)
    if name == "bert4rec":
        return BERT4Rec(num_items=num_items, max_len=int(cfg.baselines.max_history_items) + 2, **kwargs)
    raise ValueError(f"unsupported model: {name}")


def train_one(cfg, name: str, train_loader, valid_loader, num_items: int, device):
    model = build_model(name, num_items, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.baselines.learning_rate), weight_decay=float(cfg.baselines.weight_decay))
    best_loss = math.inf
    stale = 0
    history = []

    def epoch_pass(loader, training: bool) -> float:
        model.train(training)
        total, count = 0.0, 0
        with torch.set_grad_enabled(training):
            for sequence, lengths, targets in loader:
                sequence = sequence.to(device)
                lengths = lengths.to(device)
                targets = targets.to(device)
                if name == "bert4rec" and training:
                    mask = (torch.rand_like(sequence, dtype=torch.float) < float(cfg.baselines.bert_mask_prob)) & sequence.ne(0)
                    sequence = torch.where(mask, torch.full_like(sequence, int(model.mask_token_id)), sequence)
                user_vectors = model.encode(sequence, lengths)
                negatives = torch.randint(1, num_items, (targets.shape[0], int(cfg.baselines.num_negatives)), device=device)
                negatives = torch.where(negatives.eq(targets.unsqueeze(1)), torch.ones_like(negatives), negatives)
                loss = sampled_softmax_loss(user_vectors, model, targets, negatives, float(cfg.baselines.temperature))
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                total += float(loss.detach().cpu())
                count += 1
        return total / max(count, 1)

    for epoch in range(1, int(cfg.baselines.epochs) + 1):
        train_loss = epoch_pass(train_loader, True)
        valid_loss = epoch_pass(valid_loader, False)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        print(f"[{name}] epoch={epoch:03d} train={train_loss:.4f} valid={valid_loss:.4f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            stale = 0
            checkpoint = {
                "model_name": name,
                "state_dict": model.state_dict(),
                "item_ids": item_ids,
                "item_to_model": item_to_model,
                "num_items": num_items,
                "dim": int(cfg.baselines.dim),
                "best_valid_loss": best_loss,
                "history": history,
            }
            torch.save(checkpoint, cfg.paths.artifact(f"baselines_{name}.pt"))
        else:
            stale += 1
            if stale >= int(cfg.baselines.early_stopping_patience):
                break
    return {"model": name, "best_valid_loss": best_loss, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = resolve_device("auto")
    item_ids = [str(value) for value in pq.read_table(cfg.paths.artifact("catalog.parquet"), columns=["item_id"]).column("item_id").to_pylist()]
    item_to_model = {item_id: index + 1 for index, item_id in enumerate(item_ids)}
    num_items = len(item_ids) + 2
    train_dataset = ItemPrefixDataset(cfg, item_to_model)
    valid_dataset = EvalItemDataset(cfg, item_to_model, "valid")
    train_loader = DataLoader(train_dataset, batch_size=int(cfg.baselines.batch_size), shuffle=True, collate_fn=lambda batch: collate(batch, int(cfg.baselines.max_history_items)))
    valid_loader = DataLoader(valid_dataset, batch_size=int(cfg.baselines.batch_size), shuffle=False, collate_fn=lambda batch: collate(batch, int(cfg.baselines.max_history_items)))
    requested = args.models or cfg.baselines.models
    report = {}
    for name in requested:
        if name in {"popularity", "itemcf"}:
            continue
        print(f"[baselines] training {name}")
        report[name] = train_one(cfg, name, train_loader, valid_loader, num_items, device)
    write_json(cfg.paths.artifact("baseline_training.json"), report)
    print(report)
