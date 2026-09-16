"""Frequency-aware, semantic-aware collision handling for RQ-VAE codes.

The collision resolver supports two SID length policies:

``variable``
    The current behavior: ordinary items use the ``L`` base codewords from the
    RQ-VAE, while only colliding items append a conflict token.  The effective
    SID length is therefore variable (``L`` or ``L + 1``).

``fixed``
    Every item receives the same ``L + 1`` SID length.  Non-colliding items use
    conflict token 0 and colliding items consume subsequent conflict tokens.
    The RQ-VAE itself still has exactly ``L`` codebook levels; the extra token
    is an explicit, uniformly present collision-resolution level.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tiger_rec.semantic_id.token_space import TokenSpace

VALID_LENGTH_MODES = {"variable", "fixed"}


@dataclass
class SIDAssignment:
    item_id: str
    base_codes: list[int]
    suffix_index: int | None
    token_ids: list[int]
    assignment: str  # anchor | semantic_reallocation | suffix
    train_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CollisionReport:
    items: int
    unique_base_codes: int
    duplicate_groups: int
    duplicate_items: int
    base_collision_rate: float
    semantic_reallocated: int
    suffix_resolved: int
    unresolved: int
    max_suffix_used: int
    length_mode: str
    fixed_suffix_tokens: int
    min_sid_length: int
    mean_sid_length: float
    max_sid_length: int
    reallocation_mode: str
    resolution_seconds: float


def resolve_semantic_id_collisions(
    item_ids: list[str],
    base_codes: np.ndarray,
    token_space: TokenSpace,
    train_counts: np.ndarray | None = None,
    topk_codes: np.ndarray | None = None,
    max_suffix_tokens: int | None = None,
    length_mode: str = "variable",
    reallocation_mode: str = "last",
) -> tuple[dict[str, SIDAssignment], CollisionReport]:
    """Resolve exact-code collisions under a fixed or variable SID length policy.

    Args:
        base_codes: ``[num_items, num_levels]`` integer codewords.
        topk_codes: Optional ``[num_items, num_levels, k]`` nearest codewords.
            Only the last level is used for alternative-base reallocation.
        train_counts: Optional item frequencies used to choose the anchor.
        length_mode: ``variable`` preserves the current L/L+1 behavior.  ``fixed``
            makes every item use exactly ``num_levels + 1`` tokens.
    """
    started = time.perf_counter()
    length_mode = str(length_mode).lower()
    if length_mode not in VALID_LENGTH_MODES:
        raise ValueError(f"length_mode must be one of {sorted(VALID_LENGTH_MODES)}, got {length_mode!r}")
    reallocation_mode = str(reallocation_mode).lower()
    if reallocation_mode not in {"last", "all"}:
        raise ValueError(f"reallocation_mode must be 'last' or 'all', got {reallocation_mode!r}")

    base_codes = np.asarray(base_codes, dtype=np.int64)
    if base_codes.ndim != 2:
        raise ValueError("base_codes must be a 2-D array")
    if base_codes.shape[0] != len(item_ids):
        raise ValueError("item_ids and base_codes disagree on size")
    if base_codes.shape[1] != token_space.num_levels:
        raise ValueError("base_codes does not match token space levels")

    counts = (
        np.asarray(train_counts, dtype=np.int64)
        if train_counts is not None
        else np.ones(len(item_ids), dtype=np.int64)
    )
    if counts.shape[0] != len(item_ids):
        raise ValueError("train_counts and item_ids disagree on size")

    max_suffix = token_space.max_conflict_tokens if max_suffix_tokens is None else int(max_suffix_tokens)
    max_suffix = min(max_suffix, token_space.max_conflict_tokens)
    if length_mode == "fixed" and max_suffix < 1:
        raise ValueError("fixed length_mode requires at least one conflict token")

    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, code in enumerate(base_codes):
        groups[tuple(int(v) for v in code)].append(index)

    # Reserve every original code. A reallocated item may not steal another
    # item's base code because that would only move the collision elsewhere.
    reserved = set(groups)
    assigned_base_codes = set(reserved)
    assigned_token_sequences: set[tuple[int, ...]] = set()
    assignments: dict[str, SIDAssignment] = {}
    semantic_reallocated = 0
    suffix_resolved = 0
    fixed_suffix_tokens = 0
    max_suffix_used = -1

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    for base_code, indices in ordered_groups:
        indices = sorted(indices, key=lambda idx: (-int(counts[idx]), item_ids[idx]))
        anchor = indices[0]
        anchor_suffix = 0 if length_mode == "fixed" else None
        anchor_token_ids = token_space.encode_sid(base_code, suffix_index=anchor_suffix)
        anchor_assignment = SIDAssignment(
            item_id=item_ids[anchor],
            base_codes=list(base_code),
            suffix_index=anchor_suffix,
            token_ids=anchor_token_ids,
            assignment="anchor",
            train_count=int(counts[anchor]),
        )
        assignments[anchor_assignment.item_id] = anchor_assignment
        assigned_token_sequences.add(tuple(anchor_assignment.token_ids))
        if anchor_suffix is not None:
            fixed_suffix_tokens += 1
            max_suffix_used = max(max_suffix_used, anchor_suffix)

        # In fixed mode suffix token 0 is already occupied by the anchor.
        next_suffix = 1 if length_mode == "fixed" else 0
        for index in indices[1:]:
            target = tuple(int(v) for v in base_codes[index])
            chosen_codes: list[int] | None = None
            if topk_codes is not None:
                levels_to_search = (
                    range(len(target) - 1, -1, -1)
                    if reallocation_mode == "all"
                    else (len(target) - 1,)
                )
                for level in levels_to_search:
                    candidates = np.asarray(topk_codes[index, level], dtype=np.int64)
                    for candidate in candidates:
                        candidate = int(candidate)
                        if candidate == target[level]:
                            continue
                        alternative = list(target)
                        alternative[level] = candidate
                        alternative = tuple(alternative)
                        if alternative in reserved:
                            continue
                        if alternative in assigned_base_codes:
                            continue
                        chosen_codes = list(alternative)
                        break
                    if chosen_codes is not None:
                        break

            if chosen_codes is not None:
                reallocated_suffix = 0 if length_mode == "fixed" else None
                tokens = token_space.encode_sid(chosen_codes, suffix_index=reallocated_suffix)
                if tuple(tokens) not in assigned_token_sequences:
                    assignments[item_ids[index]] = SIDAssignment(
                        item_id=item_ids[index],
                        base_codes=chosen_codes,
                        suffix_index=reallocated_suffix,
                        token_ids=tokens,
                        assignment="semantic_reallocation",
                        train_count=int(counts[index]),
                    )
                    assigned_token_sequences.add(tuple(tokens))
                    assigned_base_codes.add(tuple(chosen_codes))
                    semantic_reallocated += 1
                    if reallocated_suffix is not None:
                        fixed_suffix_tokens += 1
                        max_suffix_used = max(max_suffix_used, reallocated_suffix)
                    continue

            if next_suffix >= max_suffix:
                raise RuntimeError(
                    f"not enough conflict tokens for item {item_ids[index]}; "
                    f"increase sid.max_conflict_tokens"
                )
            suffix_index = next_suffix
            next_suffix += 1
            tokens = token_space.encode_sid(target, suffix_index=suffix_index)
            assignments[item_ids[index]] = SIDAssignment(
                item_id=item_ids[index],
                base_codes=list(target),
                suffix_index=suffix_index,
                token_ids=tokens,
                assignment="suffix",
                train_count=int(counts[index]),
            )
            assigned_token_sequences.add(tuple(tokens))
            suffix_resolved += 1
            max_suffix_used = max(max_suffix_used, suffix_index)

    duplicate_groups = sum(1 for indices in groups.values() if len(indices) > 1)
    duplicate_items = len(item_ids) - len(groups)
    lengths = np.asarray([len(assignment.token_ids) for assignment in assignments.values()], dtype=np.int64)
    report = CollisionReport(
        items=len(item_ids),
        unique_base_codes=len(groups),
        duplicate_groups=duplicate_groups,
        duplicate_items=duplicate_items,
        base_collision_rate=duplicate_items / max(len(item_ids), 1),
        semantic_reallocated=semantic_reallocated,
        suffix_resolved=suffix_resolved,
        unresolved=0,
        max_suffix_used=max_suffix_used,
        length_mode=length_mode,
        fixed_suffix_tokens=fixed_suffix_tokens,
        min_sid_length=int(lengths.min()) if lengths.size else 0,
        mean_sid_length=float(lengths.mean()) if lengths.size else 0.0,
        max_sid_length=int(lengths.max()) if lengths.size else 0,
        reallocation_mode=reallocation_mode,
        resolution_seconds=float(time.perf_counter() - started),
    )
    if len(assignments) != len(item_ids):
        raise AssertionError("collision resolver did not assign every item")
    return assignments, report
