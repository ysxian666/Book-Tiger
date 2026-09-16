"""Multi-view residual-quantized VAE for hierarchical Semantic IDs.

The implementation follows TIGER's RQ-VAE idea and adds two project-specific
components: gated/cross-attention fusion of text and collaborative embeddings,
and inverse-frequency reconstruction weights for long-tail items.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class QuantizerOutput:
    quantized: torch.Tensor
    straight_through: torch.Tensor
    codes: torch.Tensor
    topk_codes: torch.Tensor
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    usage_loss: torch.Tensor
    hard_usage_loss: torch.Tensor
    joint_collision_loss: torch.Tensor
    entropy: torch.Tensor


class ResidualQuantizer(nn.Module):
    def __init__(
        self,
        num_levels: int,
        codebook_size: int,
        latent_dim: int,
        temperature: float = 0.1,
        usage_ema_decay: float = 0.9,
    ) -> None:
        super().__init__()
        self.num_levels = int(num_levels)
        self.codebook_size = int(codebook_size)
        self.latent_dim = int(latent_dim)
        self.temperature = float(temperature)
        self.usage_ema_decay = float(usage_ema_decay)
        self.register_buffer("usage_ema", torch.zeros(self.num_levels, self.codebook_size))
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, latent_dim) * (latent_dim ** -0.5))
            for _ in range(num_levels)
        ])

    def _distances(self, residual: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        # squared Euclidean distance, stable for standard embedding magnitudes
        return (
            residual.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * residual @ codebook.t()
        ).clamp_min_(0.0)

    def forward(self, z: torch.Tensor, topk: int = 1) -> QuantizerOutput:
        residual = z
        quantized_sum = torch.zeros_like(z)
        straight_sum = torch.zeros_like(z)
        codes: list[torch.Tensor] = []
        topk_codes: list[torch.Tensor] = []
        commitment_loss = z.new_zeros(())
        codebook_loss = z.new_zeros(())
        usage_loss = z.new_zeros(())
        hard_usage_loss = z.new_zeros(())
        joint_collision_loss = z.new_zeros(())
        entropy_sum = z.new_zeros(())
        hard_assignments: list[torch.Tensor] = []
        if self.training:
            self.usage_ema.mul_(self.usage_ema_decay)

        for level, codebook in enumerate(self.codebooks):
            distances = self._distances(residual, codebook)
            k = max(1, min(int(topk), self.codebook_size))
            nearest_dist, nearest_idx = torch.topk(distances, k=k, dim=1, largest=False)
            index = nearest_idx[:, 0]
            selected = codebook[index]
            quantized_sum = quantized_sum + selected
            straight_sum = straight_sum + residual + (selected - residual).detach()
            codes.append(index)
            topk_codes.append(nearest_idx)
            if self.training:
                self.usage_ema[level].index_add_(
                    0,
                    index.detach(),
                    torch.ones_like(index, dtype=self.usage_ema.dtype),
                )

            # Vector-quantization losses.
            commitment_loss = commitment_loss + F.mse_loss(residual, selected.detach())
            codebook_loss = codebook_loss + F.mse_loss(residual.detach(), selected)

            # Soft codebook usage regularizer. It avoids hard dead-code gradients
            # and is more stable than a pure entropy bonus with cold codebooks.
            logits = (-distances / max(self.temperature, 1e-6))
            probability = F.softmax(logits, dim=1)
            mean_probability = probability.mean(dim=0)
            uniform = torch.full_like(mean_probability, 1.0 / self.codebook_size)
            usage_loss = usage_loss + F.mse_loss(mean_probability, uniform)
            hard_assignment = F.one_hot(index, num_classes=self.codebook_size).to(probability.dtype)
            straight_through_hard = hard_assignment - probability.detach() + probability
            hard_assignments.append(straight_through_hard)
            hard_mean_probability = straight_through_hard.mean(dim=0)
            hard_usage_loss = hard_usage_loss + F.mse_loss(hard_mean_probability, uniform)
            entropy_sum = entropy_sum - (mean_probability * (mean_probability + 1e-8).log()).sum()

            residual = residual - selected.detach()

        if self.training and z.shape[0] > 1:
            same_sid_probability = torch.ones(
                z.shape[0], z.shape[0], device=z.device, dtype=z.dtype,
            )
            for assignment in hard_assignments:
                same_sid_probability = same_sid_probability * (assignment @ assignment.t())
            diagonal = torch.diagonal(same_sid_probability)
            off_diagonal_sum = same_sid_probability.sum() - diagonal.sum()
            joint_collision_loss = off_diagonal_sum / float(z.shape[0] * (z.shape[0] - 1))

        return QuantizerOutput(
            quantized=quantized_sum,
            straight_through=straight_sum,
            codes=torch.stack(codes, dim=1),
            topk_codes=torch.stack(topk_codes, dim=1),
            commitment_loss=commitment_loss / self.num_levels,
            codebook_loss=codebook_loss / self.num_levels,
            usage_loss=usage_loss / self.num_levels,
            hard_usage_loss=hard_usage_loss / self.num_levels,
            joint_collision_loss=joint_collision_loss,
            entropy=entropy_sum / self.num_levels,
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        decoded = torch.zeros(codes.shape[0], self.latent_dim, device=codes.device)
        for level, codebook in enumerate(self.codebooks):
            decoded = decoded + codebook[codes[:, level]]
        return decoded

class GatedMultiViewFusion(nn.Module):
    def __init__(
        self,
        text_dim: int,
        collab_dim: int,
        output_dim: int,
        fusion_type: str = "gate",
        gate_hidden_dim: int = 256,
        dropout: float = 0.1,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.fusion_type = fusion_type
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        if fusion_type == "content_only":
            self.collab_proj = None
            self.gate = None
            self.attention = None
        elif fusion_type == "collab_only":
            self.collab_proj = nn.Sequential(
                nn.Linear(collab_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
            self.gate = None
            self.attention = None
        else:
            self.collab_proj = nn.Sequential(
                nn.Linear(collab_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
            if fusion_type == "cross_attention":
                self.attention = nn.MultiheadAttention(
                    embed_dim=output_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.gate = None
            elif fusion_type == "gate":
                self.gate = nn.Sequential(
                    nn.Linear(output_dim * 2, gate_hidden_dim),
                    nn.GELU(),
                    nn.Linear(gate_hidden_dim, output_dim),
                    nn.Sigmoid(),
                )
                self.attention = None
            else:
                raise ValueError(f"unknown fusion_type: {fusion_type}")
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor, collaborative: torch.Tensor | None) -> torch.Tensor:
        text_feature = self.text_proj(text)
        if self.fusion_type == "content_only":
            return self.dropout(text_feature)
        if collaborative is None:
            if self.collab_proj is not None:
                collaborative = torch.zeros(
                    text.shape[0], self.collab_proj[0].in_features,
                    device=text.device, dtype=text.dtype,
                )
        assert collaborative is not None
        collab_feature = self.collab_proj(collaborative)
        if self.fusion_type == "collab_only":
            return self.dropout(collab_feature)
        if self.fusion_type == "gate":
            gate = self.gate(torch.cat([text_feature, collab_feature], dim=-1))
            fused = gate * text_feature + (1.0 - gate) * collab_feature
        elif self.fusion_type == "cross_attention":
            attended, _ = self.attention(
                query=text_feature.unsqueeze(1),
                key=collab_feature.unsqueeze(1),
                value=collab_feature.unsqueeze(1),
                need_weights=False,
            )
            fused = text_feature + attended.squeeze(1)
        else:
            raise RuntimeError("unreachable")
        return self.dropout(fused)


class MultiViewRQVAE(nn.Module):
    def __init__(
        self,
        text_dim: int,
        collab_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_levels: int,
        codebook_size: int,
        fusion_type: str = "gate",
        gate_hidden_dim: int = 256,
        dropout: float = 0.1,
        temperature: float = 0.1,
        usage_ema_decay: float = 0.9,
    ) -> None:
        super().__init__()
        self.fusion = GatedMultiViewFusion(
            text_dim=text_dim,
            collab_dim=collab_dim,
            output_dim=hidden_dim,
            fusion_type=fusion_type,
            gate_hidden_dim=gate_hidden_dim,
            dropout=dropout,
        )
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.quantizer = ResidualQuantizer(
            num_levels=num_levels,
            codebook_size=codebook_size,
            latent_dim=latent_dim,
            temperature=temperature,
            usage_ema_decay=usage_ema_decay,
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        text: torch.Tensor,
        collaborative: torch.Tensor | None,
        sample_weights: torch.Tensor | None = None,
        topk: int = 1,
    ) -> dict[str, torch.Tensor | QuantizerOutput]:
        fused = self.fusion(text, collaborative)
        latent = self.encoder(fused)
        quantized = self.quantizer(latent, topk=topk)
        reconstructed = self.decoder(quantized.straight_through)
        reconstruction = F.mse_loss(reconstructed, fused, reduction="none").mean(dim=-1)
        if sample_weights is None:
            reconstruction_loss = reconstruction.mean()
        else:
            weights = sample_weights.to(reconstruction.dtype)
            reconstruction_loss = (reconstruction * weights).sum() / weights.sum().clamp_min(1e-8)
        return {
            "fused": fused,
            "latent": latent,
            "quantized": quantized,
            "reconstruction": reconstructed,
            "reconstruction_loss": reconstruction_loss,
        }

    def encode_codes(
        self,
        text: torch.Tensor,
        collaborative: torch.Tensor | None = None,
        topk: int = 1,
    ) -> QuantizerOutput:
        fused = self.fusion(text, collaborative)
        latent = self.encoder(fused)
        return self.quantizer(latent, topk=topk)

def inverse_frequency_weights(
    counts: np.ndarray,
    power: float = 0.5,
    clip: float = 10.0,
) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    safe_counts = np.maximum(counts, 1.0)
    weights = (safe_counts.mean() / safe_counts) ** float(power)
    weights = np.clip(weights, 1.0 / float(clip), float(clip))
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def codebook_metrics(codes: np.ndarray, codebook_size: int) -> dict[str, float]:
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 2:
        raise ValueError("codes must be [num_items, num_levels]")
    utilization: dict[str, float] = {}
    perplexity: dict[str, float] = {}
    for level in range(codes.shape[1]):
        values, frequencies = np.unique(codes[:, level], return_counts=True)
        probability = frequencies.astype(np.float64) / frequencies.sum()
        utilization[f"level_{level}"] = float(len(values) / codebook_size)
        perplexity[f"level_{level}"] = float(np.exp(-(probability * np.log(probability + 1e-12)).sum()))
    return {
        "mean_utilization": float(np.mean(list(utilization.values()))),
        "mean_perplexity": float(np.mean(list(perplexity.values()))),
        **{f"utilization_{k}": v for k, v in utilization.items()},
        **{f"perplexity_{k}": v for k, v in perplexity.items()},
    }


@torch.no_grad()
def kmeans_initialize_codebooks(
    model: MultiViewRQVAE,
    text: torch.Tensor,
    collaborative: torch.Tensor | None,
    max_iter: int = 100,
    seed: int = 42,
) -> None:
    """Initialize each residual codebook with MiniBatchKMeans."""
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:  # pragma: no cover - dependency is in requirements
        raise RuntimeError("scikit-learn is required for kmeans initialization") from exc

    device = next(model.parameters()).device
    model.eval()
    fused = model.fusion(text.to(device), collaborative.to(device) if collaborative is not None else None)
    residual = model.encoder(fused).detach().cpu().numpy()
    for level, codebook in enumerate(model.quantizer.codebooks):
        n_clusters = min(model.quantizer.codebook_size, residual.shape[0])
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            max_iter=int(max_iter),
            n_init=3,
            random_state=seed + level,
            batch_size=min(4096, max(256, residual.shape[0] // 10)),
        )
        labels = kmeans.fit_predict(residual)
        centers = kmeans.cluster_centers_.astype(np.float32)
        if n_clusters < model.quantizer.codebook_size:
            padding = np.random.default_rng(seed + level).normal(
                0.0, 0.01, size=(model.quantizer.codebook_size - n_clusters, centers.shape[1])
            ).astype(np.float32)
            centers = np.concatenate([centers, padding], axis=0)
        codebook.copy_(torch.from_numpy(centers).to(codebook.device))
        residual = residual - kmeans.cluster_centers_[labels]
    model.train()


@torch.no_grad()
def reset_dead_codebooks(
    model: MultiViewRQVAE,
    latent: torch.Tensor,
    usage_counts: list[torch.Tensor] | None = None,
    threshold: float = 0.0,
    noise: float = 0.01,
    usage_ema: torch.Tensor | None = None,
) -> int:
    """Reinitialize codebook vectors whose EMA hard usage is at or below threshold."""
    if latent.numel() == 0:
        return 0
    latent = latent.detach()
    reset = 0
    for level, codebook in enumerate(model.quantizer.codebooks):
        if usage_ema is not None:
            if level >= usage_ema.shape[0]:
                continue
            usage = usage_ema[level]
        elif usage_counts is not None and level < len(usage_counts):
            usage = usage_counts[level]
        else:
            continue
        dead = torch.nonzero(usage <= float(threshold), as_tuple=False).flatten()
        if dead.numel() == 0:
            continue
        sample_index = torch.randint(0, latent.shape[0], (dead.numel(),), device=latent.device)
        replacement = latent[sample_index].to(device=codebook.device, dtype=codebook.dtype)
        if noise > 0:
            replacement = replacement + float(noise) * torch.randn_like(replacement)
        codebook.data[dead.to(codebook.device)] = replacement
        reset += int(dead.numel())
    return reset


@torch.no_grad()
def reset_dead_codebooks_error_aware(
    model: MultiViewRQVAE,
    latent: torch.Tensor,
    threshold: float = 0.0,
    noise: float = 0.01,
) -> dict[str, int]:
    """Revive dead codes with high-error residuals at each residual depth."""
    if latent.numel() == 0:
        return {"total": 0}
    residual = latent.detach()
    counts: dict[str, int] = {"total": 0}
    for level, codebook in enumerate(model.quantizer.codebooks):
        distances = model.quantizer._distances(residual, codebook)
        nearest_dist, nearest_idx = distances.min(dim=1)
        usage = torch.bincount(
            nearest_idx, minlength=model.quantizer.codebook_size
        ).to(distances.dtype)
        dead = torch.nonzero(usage <= float(threshold), as_tuple=False).flatten()
        if dead.numel() > 0:
            replacement_count = min(int(dead.numel()), int(residual.shape[0]))
            candidate_idx = torch.topk(
                nearest_dist, k=replacement_count, largest=True
            ).indices
            replacement = residual[candidate_idx]
            if noise > 0:
                scale = replacement.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
                replacement = replacement + float(noise) * scale * torch.randn_like(replacement)
            codebook.data[dead[:replacement_count].to(codebook.device)] = replacement.to(
                device=codebook.device, dtype=codebook.dtype,
            )
            counts[f"level_{level}"] = replacement_count
            counts["total"] += replacement_count
            distances = model.quantizer._distances(residual, codebook)
            _, nearest_idx = distances.min(dim=1)
        else:
            counts[f"level_{level}"] = 0
        residual = residual - codebook[nearest_idx].detach()
    return counts


def save_rqvae(model: MultiViewRQVAE, path: str, metadata: dict[str, Any]) -> None:
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_rqvae(path: str, map_location: str | torch.device = "cpu") -> tuple[MultiViewRQVAE, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    metadata = checkpoint["metadata"]
    model = MultiViewRQVAE(
        text_dim=int(metadata["text_dim"]),
        collab_dim=int(metadata["collab_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        latent_dim=int(metadata["latent_dim"]),
        num_levels=int(metadata["num_levels"]),
        codebook_size=int(metadata["codebook_size"]),
        fusion_type=str(metadata.get("fusion_type", "gate")),
        gate_hidden_dim=int(metadata.get("gate_hidden_dim", 256)),
        dropout=float(metadata.get("dropout", 0.1)),
        temperature=float(metadata.get("temperature", 0.1)),
        usage_ema_decay=float(metadata.get("usage_ema_decay", 0.9)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, metadata
