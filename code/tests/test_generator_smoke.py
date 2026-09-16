import torch

from tiger_rec.config import GeneratorConfig
from tiger_rec.decoding.constrained_beam import beam_search
from tiger_rec.models.generator import build_t5_model
from tiger_rec.semantic_id.token_space import EOS, TokenSpace
from tiger_rec.semantic_id.trie import SIDTrie


def test_t5_generator_and_constrained_decode_smoke():
    space = TokenSpace(num_levels=2, codebook_size=4, max_conflict_tokens=2)
    trie = SIDTrie()
    trie.insert(space.encode_sid([0, 1]), "item_a")
    trie.insert(space.encode_sid([2, 3]), "item_b")
    config = GeneratorConfig(
        architecture="t5-tiny",
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_decode_steps=4,
    )
    model = build_t5_model(space, config)
    input_ids = torch.tensor([[space.sid_token(0, 0), space.sid_token(1, 1)]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[space.sid_token(0, 2), space.sid_token(1, 3), EOS]])
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    assert torch.isfinite(loss)
    decoded = beam_search(
        model,
        input_ids,
        attention_mask,
        trie,
        space,
        num_beams=2,
        max_steps=4,
        constrained=True,
    )
    assert len(decoded) == 1
    assert all(beam.valid for beam in decoded[0])
