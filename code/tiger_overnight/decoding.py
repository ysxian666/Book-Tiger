"""Standard/trie beam search and exhaustive SID scoring."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F

from tiger_rec.semantic_id.token_space import BOS, TokenSpace


@dataclass
class DecodedUser:
    item_ids: list[str]
    scores: list[float]
    token_ids: list[tuple[int, ...]]
    invalid_beams: int
    total_beams: int


def build_trie(tokens_by_item: dict[str, list[int]]) -> dict[tuple[int, ...], set[int]]:
    trie: dict[tuple[int, ...], set[int]] = defaultdict(set)
    for token_ids in tokens_by_item.values():
        prefix: tuple[int, ...] = ()
        for token in token_ids:
            trie[prefix].add(int(token))
            prefix = prefix + (int(token),)
    return dict(trie)


def _allowed_step_tokens(token_space: TokenSpace, step: int) -> list[int]:
    if step < token_space.num_levels:
        start = token_space.sid_base + step * token_space.codebook_size
        return list(range(start, start + token_space.codebook_size))
    return [token_space.conflict_token(index) for index in range(token_space.max_conflict_tokens)]


@torch.inference_mode()
def beam_search(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    long_context: torch.Tensor,
    short_context: torch.Tensor,
    token_space: TokenSpace,
    token_to_item: dict[tuple[int, ...], str],
    history_item_ids: list[set[str]],
    beam_size: int = 20,
    constrained_trie: dict[tuple[int, ...], set[int]] | None = None,
    max_steps: int = 4,
) -> list[DecodedUser]:
    device = attention_mask.device
    encoder_hidden, encoder_mask = model.encode(input_ids, attention_mask, long_context, short_context)
    batch_size = int(input_ids.shape[0])
    decoder_inputs = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    scores = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    sequences: list[list[tuple[int, ...]]] = [[()] for _ in range(batch_size)]

    for step in range(int(max_steps)):
        beam_count = int(scores.shape[1])
        flat_indices = torch.arange(batch_size, device=device).repeat_interleave(beam_count)
        flat_decoder = decoder_inputs.reshape(batch_size * beam_count, -1)
        flat_hidden = encoder_hidden[flat_indices]
        flat_mask = encoder_mask[flat_indices]
        logits = model.decode_logits(flat_hidden, flat_mask, flat_decoder)[:, -1, :].float()
        next_inputs: list[torch.Tensor] = []
        next_scores: list[torch.Tensor] = []
        next_sequences: list[list[tuple[int, ...]]] = []
        for batch_index in range(batch_size):
            allowed_rows: list[list[int]] = []
            for beam_index in range(beam_count):
                prefix = sequences[batch_index][beam_index]
                if constrained_trie is None:
                    allowed_rows.append(_allowed_step_tokens(token_space, step))
                else:
                    allowed_rows.append(sorted(constrained_trie.get(prefix, set())))
            candidates: list[tuple[float, tuple[int, ...], int]] = []
            for beam_index, allowed in enumerate(allowed_rows):
                if not allowed:
                    continue
                allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
                row_index = batch_index * beam_count + beam_index
                token_log_probs = F.log_softmax(logits[row_index], dim=-1)[allowed_tensor]
                parent_score = float(scores[batch_index, beam_index].detach().cpu())
                values, order = torch.topk(token_log_probs, k=min(beam_size, token_log_probs.numel()), largest=True)
                for value, allowed_pos in zip(values.detach().cpu().tolist(), order.detach().cpu().tolist()):
                    token = int(allowed[int(allowed_pos)])
                    candidates.append((parent_score + float(value), sequences[batch_index][beam_index] + (token,), beam_index))
            candidates.sort(key=lambda value: value[0], reverse=True)
            seen: set[tuple[int, ...]] = set()
            chosen: list[tuple[float, tuple[int, ...], int]] = []
            for candidate in candidates:
                if candidate[1] in seen:
                    continue
                seen.add(candidate[1])
                chosen.append(candidate)
                if len(chosen) >= beam_size:
                    break
            if not chosen:
                raise RuntimeError(f"beam search produced no candidates at step {step}")
            user_scores = torch.tensor([value[0] for value in chosen], dtype=torch.float32, device=device)
            user_decoders = []
            for _score, sequence, _parent in chosen:
                user_decoders.append(sequence)
            next_scores.append(user_scores)
            next_sequences.append([value[1] for value in chosen])
            next_inputs.append(torch.tensor(user_decoders, dtype=torch.long, device=device))
        scores = torch.stack(next_scores, dim=0)
        sequences = next_sequences
        decoder_inputs = torch.stack(next_inputs, dim=0)

    output: list[DecodedUser] = []
    for batch_index in range(batch_size):
        valid_items: list[str] = []
        valid_scores: list[float] = []
        valid_tokens: list[tuple[int, ...]] = []
        invalid = 0
        seen_items: set[str] = set()
        for score, token_ids in zip(scores[batch_index].detach().cpu().tolist(), sequences[batch_index]):
            item_id = token_to_item.get(tuple(int(value) for value in token_ids))
            if item_id is None:
                invalid += 1
                continue
            if item_id in history_item_ids[batch_index] or item_id in seen_items:
                continue
            seen_items.add(item_id)
            valid_items.append(item_id)
            valid_scores.append(float(score))
            valid_tokens.append(tuple(int(value) for value in token_ids))
        output.append(DecodedUser(
            item_ids=valid_items,
            scores=valid_scores,
            token_ids=valid_tokens,
            invalid_beams=invalid,
            total_beams=int(scores.shape[1]),
        ))
    return output


@torch.inference_mode()
def exhaustive_rank(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    long_context: torch.Tensor,
    short_context: torch.Tensor,
    candidate_token_ids: np.ndarray,
    candidate_item_ids: list[str],
    history_item_ids: list[set[str]],
    top_k: int = 20,
    candidate_batch_size: int = 2048,
) -> list[DecodedUser]:
    device = attention_mask.device
    encoder_hidden, encoder_mask = model.encode(input_ids, attention_mask, long_context, short_context)
    batch_size = int(input_ids.shape[0])
    candidates = torch.from_numpy(candidate_token_ids.astype(np.int64))
    output: list[DecodedUser] = []
    for batch_index in range(batch_size):
        all_scores: list[torch.Tensor] = []
        for start in range(0, candidates.shape[0], int(candidate_batch_size)):
            end = min(start + int(candidate_batch_size), candidates.shape[0])
            candidate_chunk = candidates[start:end].to(device)
            count = int(candidate_chunk.shape[0])
            hidden = encoder_hidden[batch_index:batch_index + 1].expand(count, -1, -1)
            mask = encoder_mask[batch_index:batch_index + 1].expand(count, -1)
            decoder_input = torch.cat([
                torch.full((count, 1), BOS, dtype=torch.long, device=device),
                candidate_chunk[:, :-1],
            ], dim=1)
            logits = model.decode_logits(hidden, mask, decoder_input).float()
            log_probs = F.log_softmax(logits, dim=-1)
            gathered = log_probs.gather(2, candidate_chunk.unsqueeze(-1)).squeeze(-1)
            all_scores.append(gathered.sum(dim=1))
        scores = torch.cat(all_scores, dim=0).detach().cpu().numpy()
        order = np.argsort(-scores, kind="stable")
        seen: set[str] = set()
        items: list[str] = []
        valid_scores: list[float] = []
        tokens: list[tuple[int, ...]] = []
        for index in order:
            item_id = candidate_item_ids[int(index)]
            if item_id in history_item_ids[batch_index] or item_id in seen:
                continue
            seen.add(item_id)
            items.append(item_id)
            valid_scores.append(float(scores[int(index)]))
            tokens.append(tuple(int(value) for value in candidate_token_ids[int(index)]))
            if len(items) >= int(top_k):
                break
        output.append(DecodedUser(
            item_ids=items,
            scores=valid_scores,
            token_ids=tokens,
            invalid_beams=0,
            total_beams=int(candidates.shape[0]),
        ))
    return output
