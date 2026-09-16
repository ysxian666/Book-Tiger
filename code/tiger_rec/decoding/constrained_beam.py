"""Beam decoding for standard TIGER generation and trie-constrained variants."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from tiger_rec.semantic_id.token_space import BOS, EOS, TokenSpace
from tiger_rec.semantic_id.trie import SIDTrie


@dataclass(frozen=True)
class DecodedBeam:
    token_ids: tuple[int, ...]
    score: float
    valid: bool
    item_ids: tuple[str, ...]


@dataclass(frozen=True)
class _BeamState:
    token_ids: tuple[int, ...]
    score: float
    done: bool = False


def _tokens_to_decoded(tokens: tuple[int, ...], score: float, trie: SIDTrie) -> DecodedBeam:
    clean = tuple(token for token in tokens if token != EOS)
    items = trie.items_for_tokens(clean)
    return DecodedBeam(token_ids=clean, score=float(score), valid=bool(items), item_ids=tuple(items))

@torch.inference_mode()
def beam_search(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    trie: SIDTrie,
    token_space: TokenSpace,
    num_beams: int = 20,
    max_steps: int = 8,
    length_penalty: float = 1.0,
    constrained: bool = True,
) -> list[list[DecodedBeam]]:
    """Decode a batch and return ``num_beams`` beams per user."""
    model.eval()
    device = input_ids.device
    batch_size = input_ids.shape[0]
    beams = max(1, int(num_beams))
    steps = int(max_steps) if constrained else max(int(max_steps), int(token_space.max_sid_length) + 1)
    encoder_outputs = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    states: list[list[_BeamState]] = [[_BeamState(tuple(), 0.0, False)] for _ in range(batch_size)]
    all_code_tokens = token_space.all_sid_code_tokens()

    for _ in range(steps):
        flat_decoder_ids: list[list[int]] = []
        state_map: list[tuple[int, int]] = []
        for batch_index, batch_states in enumerate(states):
            for state_index, state in enumerate(batch_states):
                if state.done:
                    continue
                flat_decoder_ids.append([BOS, *state.token_ids])
                state_map.append((batch_index, state_index))
        if not flat_decoder_ids:
            break
        decoder_input_ids = torch.tensor(flat_decoder_ids, device=device, dtype=torch.long)
        indices = torch.tensor([item[0] for item in state_map], device=device, dtype=torch.long)
        expanded_hidden = encoder_outputs.last_hidden_state.index_select(0, indices)
        expanded_encoder = type(encoder_outputs)(last_hidden_state=expanded_hidden)
        logits = model(encoder_outputs=expanded_encoder, decoder_input_ids=decoder_input_ids).logits[:, -1, :]
        log_probs = F.log_softmax(logits.float(), dim=-1)

        next_states: list[list[_BeamState]] = [[] for _ in range(batch_size)]
        for row, (batch_index, state_index) in enumerate(state_map):
            state = states[batch_index][state_index]
            if constrained:
                node = trie.node_for_prefix(state.token_ids)
                if node is None:
                    continue
                allowed = set(node.children)
                if node.terminal:
                    allowed.add(EOS)
            else:
                allowed = set(int(token) for token in all_code_tokens)
                allowed.add(EOS)
            for token in allowed:
                token_score = float(log_probs[row, token].detach().cpu())
                new_tokens = state.token_ids if token == EOS else state.token_ids + (int(token),)
                next_states[batch_index].append(_BeamState(new_tokens, state.score + token_score, token == EOS))
        for batch_index, batch_states in enumerate(states):
            for state in batch_states:
                if state.done:
                    next_states[batch_index].append(state)
            next_states[batch_index] = sorted(next_states[batch_index], key=lambda item: item.score, reverse=True)[:beams]
        states = next_states

    results: list[list[DecodedBeam]] = []
    for batch_index in range(batch_size):
        decoded: list[DecodedBeam] = []
        for state in states[batch_index]:
            score = state.score / max(len(state.token_ids), 1) ** float(length_penalty)
            decoded.append(_tokens_to_decoded(state.token_ids, score, trie))
        decoded = sorted(decoded, key=lambda item: item.score, reverse=True)
        unique: list[DecodedBeam] = []
        seen: set[str] = set()
        for beam in decoded:
            key = beam.item_ids[0] if beam.item_ids else f"invalid:{beam.token_ids}"
            if key in seen:
                continue
            seen.add(key)
            unique.append(beam)
        results.append(unique[:beams])
    return results
