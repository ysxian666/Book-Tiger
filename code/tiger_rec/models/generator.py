"""T5-style encoder-decoder for autoregressive next-Semantic-ID prediction."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tiger_rec.config import GeneratorConfig
from tiger_rec.semantic_id.token_space import BOS, EOS, PAD, TokenSpace


def build_t5_model(token_space: TokenSpace, config: GeneratorConfig) -> nn.Module:
    try:
        from transformers import T5Config, T5ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("transformers is required to build the TIGER generator") from exc
    t5_config = T5Config(
        vocab_size=token_space.vocab_size,
        d_model=int(config.d_model),
        d_ff=int(config.d_ff),
        d_kv=int(config.d_model) // int(config.num_heads),
        num_layers=int(config.num_layers),
        num_decoder_layers=int(config.num_layers),
        num_heads=int(config.num_heads),
        dropout_rate=float(config.dropout),
        pad_token_id=PAD,
        eos_token_id=EOS,
        decoder_start_token_id=BOS,
        relative_attention_num_buckets=32,
        tie_word_embeddings=True,
    )
    return T5ForConditionalGeneration(t5_config)


def save_generator(model: nn.Module, path: str, metadata: dict[str, Any]) -> None:
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_generator(path: str, token_space: TokenSpace, config: GeneratorConfig, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model = build_t5_model(token_space, config)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint.get("metadata", {})
