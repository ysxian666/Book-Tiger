from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from tiger_overnight.decoding import beam_search, build_trie
from tiger_overnight.generator_model import TigerGenerator
from tiger_rec.semantic_id.token_space import BOS, EOS, PAD, TokenSpace


def test_generator_forward_and_soft_prefix_are_finite():
    token_space = TokenSpace(
        num_levels=3,
        codebook_size=8,
        max_conflict_tokens=2,
        use_behavior_tokens=False,
        use_time_buckets=True,
        use_user_tokens=True,
        num_user_buckets=4,
        num_time_buckets=2,
        behavior_count=0,
    )
    generator_config = {
        "d_model": 16,
        "d_ff": 32,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "num_heads": 2,
        "dropout": 0.0,
    }
    user_context = {
        "enabled": True,
        "num_soft_tokens": 2,
        "num_layers": 1,
        "num_heads": 2,
        "d_model": 16,
        "d_ff": 32,
        "dropout": 0.0,
    }
    model = TigerGenerator(token_space, generator_config, context_dim=4, user_context_config=user_context)
    input_ids = torch.tensor([[PAD, PAD, BOS, token_space.user_token(0), token_space.sid_token(0, 1), token_space.time_token(0)]] * 2)
    attention = torch.ones_like(input_ids)
    labels = torch.tensor([[token_space.sid_token(0, 1), token_space.sid_token(1, 2), token_space.sid_token(2, 3), token_space.conflict_token(0), EOS]] * 2)
    output = model(
        input_ids=input_ids,
        attention_mask=attention,
        labels=labels,
        long_context=torch.randn(2, 4),
        short_context=torch.randn(2, 4),
    )
    assert torch.isfinite(output["loss"])
    hidden, mask = model.encode(input_ids, attention, torch.randn(2, 4), torch.randn(2, 4))
    assert hidden.shape[0] == 2
    assert mask.shape == attention.shape


def test_beam_search_four_token_decode_smoke():
    token_space = TokenSpace(
        num_levels=3,
        codebook_size=8,
        max_conflict_tokens=2,
        use_behavior_tokens=False,
        use_time_buckets=True,
        use_user_tokens=True,
        num_user_buckets=4,
        num_time_buckets=2,
        behavior_count=0,
    )
    generator_config = {"d_model": 16, "d_ff": 32, "num_encoder_layers": 1, "num_decoder_layers": 1, "num_heads": 2, "dropout": 0.0}
    model = TigerGenerator(token_space, generator_config, context_dim=4, user_context_config=None).eval()
    input_ids = torch.tensor([[BOS, token_space.user_token(0), token_space.sid_token(0, 1), token_space.time_token(0)]])
    attention = torch.ones_like(input_ids)
    sid = [token_space.sid_token(0, 1), token_space.sid_token(1, 2), token_space.sid_token(2, 3), token_space.conflict_token(0)]
    token_to_item = {tuple(sid): "item"}
    decoded = beam_search(
        model,
        input_ids,
        attention,
        torch.randn(1, 4),
        torch.randn(1, 4),
        token_space,
        token_to_item,
        [set()],
        beam_size=2,
        constrained_trie=build_trie({"item": sid}),
        max_steps=4,
    )
    assert len(decoded) == 1
    assert decoded[0].item_ids == ["item"]
