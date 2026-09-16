"""Traditional sequential recommendation baselines."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRU4Rec(nn.Module):
    def __init__(self, num_items: int, dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, dim, padding_idx=0)
        self.encoder = nn.GRU(dim, dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)

    def encode(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.item_embedding(sequence))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        return hidden[-1]

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.encode(sequence, lengths)


class SASRec(nn.Module):
    def __init__(self, num_items: int, dim: int = 128, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.2, max_len: int = 100):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim * 4, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def encode(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, length = sequence.shape
        positions = torch.arange(length, device=sequence.device).unsqueeze(0).expand(batch, -1)
        hidden = self.dropout(self.item_embedding(sequence) + self.position_embedding(positions))
        hidden = self.encoder(hidden, mask=self._causal_mask(length, sequence.device), src_key_padding_mask=sequence.eq(0))
        last_index = lengths.clamp_min(1).sub(1)
        return hidden[torch.arange(batch, device=sequence.device), last_index]

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.encode(sequence, lengths)


class BERT4Rec(nn.Module):
    def __init__(self, num_items: int, dim: int = 128, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.2, max_len: int = 100):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, dim, padding_idx=0)
        self.mask_token_id = num_items - 1
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.position_embedding = nn.Embedding(max_len + 1, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim * 4, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def encode(self, sequence: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        batch, _ = sequence.shape
        cls = self.cls_token.expand(batch, -1, -1)
        hidden = torch.cat([cls, self.item_embedding(sequence)], dim=1)
        positions = torch.arange(hidden.shape[1], device=sequence.device).unsqueeze(0).expand(batch, -1)
        hidden = self.dropout(hidden + self.position_embedding(positions))
        padding_mask = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=sequence.device), sequence.eq(0)], dim=1)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return hidden[:, 0]

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode(sequence, lengths)


class TwoTower(nn.Module):
    def __init__(self, num_items: int, dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, dim, padding_idx=0)
        self.user_mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.dropout = nn.Dropout(dropout)

    def encode(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.item_embedding(sequence)
        mask = sequence.ne(0).unsqueeze(-1).to(embedded.dtype)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.user_mlp(self.dropout(pooled))

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.encode(sequence, lengths)


def item_vectors(model: nn.Module, item_ids: torch.Tensor) -> torch.Tensor:
    return model.item_embedding(item_ids)


def sampled_softmax_loss(user_vectors: torch.Tensor, model: nn.Module, targets: torch.Tensor, negatives: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    candidate_ids = torch.cat([targets.unsqueeze(1), negatives], dim=1)
    candidate_vectors = item_vectors(model, candidate_ids)
    logits = torch.einsum("bd,bkd->bk", user_vectors, candidate_vectors) / temperature
    return nn.functional.cross_entropy(logits, torch.zeros(user_vectors.shape[0], dtype=torch.long, device=user_vectors.device))
