"""T5 generator with hashed-user and soft-prefix user-conditioned variants."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from tiger_rec.semantic_id.token_space import BOS, EOS, PAD, TokenSpace


class UserContextEncoder(nn.Module):
    def __init__(self, context_dim: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float, num_soft_tokens: int) -> None:
        super().__init__()
        self.num_soft_tokens = int(num_soft_tokens)
        self.input = nn.Linear(context_dim * 2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.queries = nn.Parameter(torch.randn(1, self.num_soft_tokens, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, long_context: torch.Tensor, short_context: torch.Tensor) -> torch.Tensor:
        context = torch.cat([long_context, short_context], dim=-1)
        hidden = self.input(context).unsqueeze(1)
        hidden = self.encoder(hidden)
        prefix = self.queries.expand(hidden.shape[0], -1, -1)
        return self.norm(prefix + hidden)


@dataclass
class BeamCandidate:
    token_ids: tuple[int, ...]
    score: float
    item_id: str | None


class TigerGenerator(nn.Module):
    def __init__(
        self,
        token_space: TokenSpace,
        generator_config: dict[str, Any],
        context_dim: int,
        user_context_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        from transformers import T5Config, T5ForConditionalGeneration

        self.token_space = token_space
        self.config_dict = dict(generator_config)
        self.context_dim = int(context_dim)
        self.use_soft_prefix = bool(user_context_config and user_context_config.get("enabled", False))
        self.num_soft_tokens = int(user_context_config["num_soft_tokens"]) if self.use_soft_prefix else 0
        t5_config = T5Config(
            vocab_size=token_space.vocab_size,
            d_model=int(generator_config["d_model"]),
            d_ff=int(generator_config["d_ff"]),
            d_kv=int(generator_config["d_model"]) // int(generator_config["num_heads"]),
            num_layers=int(generator_config["num_encoder_layers"]),
            num_decoder_layers=int(generator_config["num_decoder_layers"]),
            num_heads=int(generator_config["num_heads"]),
            dropout_rate=float(generator_config["dropout"]),
            pad_token_id=PAD,
            eos_token_id=EOS,
            decoder_start_token_id=BOS,
            relative_attention_num_buckets=32,
            tie_word_embeddings=True,
        )
        self.t5 = T5ForConditionalGeneration(t5_config)
        self.context_encoder = None
        self.target_projection = nn.Linear(self.context_dim, int(generator_config["d_model"]))
        if self.use_soft_prefix:
            assert user_context_config is not None
            self.context_encoder = UserContextEncoder(
                context_dim=self.context_dim,
                d_model=int(generator_config["d_model"]),
                num_layers=int(user_context_config["num_layers"]),
                num_heads=int(user_context_config["num_heads"]),
                d_ff=int(user_context_config["d_ff"]),
                dropout=float(user_context_config["dropout"]),
                num_soft_tokens=self.num_soft_tokens,
            )

    def context_prefix(
        self,
        long_context: torch.Tensor,
        short_context: torch.Tensor,
    ) -> torch.Tensor:
        if self.context_encoder is None:
            raise RuntimeError("model was built without soft-prefix user context")
        return self.context_encoder(long_context, short_context)

    def context_alignment_loss(
        self,
        long_context: torch.Tensor,
        short_context: torch.Tensor,
        target_context: torch.Tensor,
    ) -> torch.Tensor:
        prefix = self.context_prefix(long_context, short_context).mean(dim=1)
        target = self.target_projection(target_context)
        return 1.0 - F.cosine_similarity(prefix, target, dim=-1).mean()

    def _encoder_embeddings(
        self,
        input_ids: torch.Tensor,
        long_context: torch.Tensor | None = None,
        short_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.t5.get_input_embeddings()(input_ids)
        if self.use_soft_prefix:
            if long_context is None or short_context is None:
                raise ValueError("soft-prefix generator requires long_context and short_context")
            prefix = self.context_encoder(long_context, short_context)
            embeddings = embeddings.clone()
            embeddings[:, : self.num_soft_tokens, :] = prefix[:, : embeddings.shape[1], :]
        return embeddings

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        long_context: torch.Tensor | None = None,
        short_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_embeddings = self._encoder_embeddings(input_ids, long_context, short_context)
        output = self.t5.encoder(
            inputs_embeds=encoder_embeddings,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return output.last_hidden_state, attention_mask

    def decode_logits(
        self,
        encoder_hidden: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = self.t5.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        return self.t5.lm_head(output.last_hidden_state)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
        long_context: torch.Tensor | None = None,
        short_context: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if decoder_input_ids is None:
            decoder_input_ids = self.t5._shift_right(labels)
        encoder_embeddings = self._encoder_embeddings(input_ids, long_context, short_context)
        output = self.t5(
            inputs_embeds=encoder_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )
        return {"loss": output.loss, "logits": output.logits}

    def save(self, path: str, metadata: dict[str, Any]) -> None:
        torch.save({"state_dict": self.state_dict(), "metadata": metadata}, path)

    @classmethod
    def load(
        cls,
        path: str,
        token_space: TokenSpace,
        generator_config: dict[str, Any],
        context_dim: int,
        user_context_config: dict[str, Any] | None,
        map_location: str | torch.device = "cpu",
    ) -> tuple["TigerGenerator", dict[str, Any]]:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(token_space, generator_config, context_dim, user_context_config)
        model.load_state_dict(checkpoint["state_dict"])
        return model, checkpoint.get("metadata", {})
