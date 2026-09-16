import numpy as np

from tiger_rec.semantic_id.collisions import resolve_semantic_id_collisions
from tiger_rec.semantic_id.token_space import TokenSpace


def test_collision_resolution_is_unique():
    # Two items collide at [1,2]. The second can move to [1,3].
    base = np.asarray([[1, 2], [1, 2], [4, 5]], dtype=np.int64)
    topk = np.asarray([
        [[0, 1], [1, 2]],
        [[0, 1], [3, 2]],
        [[0, 1], [5, 6]],
    ], dtype=np.int64)
    assignments, report = resolve_semantic_id_collisions(
        ["a", "b", "c"],
        base,
        TokenSpace(num_levels=2, codebook_size=8, max_conflict_tokens=4),
        train_counts=np.asarray([10, 5, 1]),
        topk_codes=topk,
    )
    sequences = {tuple(row.token_ids) for row in assignments.values()}
    assert len(sequences) == 3
    assert report.unresolved == 0
    assert assignments["a"].assignment == "anchor"



def test_fixed_length_mode_makes_every_sid_same_length():
    base = np.asarray([[1, 2], [1, 2], [4, 5]], dtype=np.int64)
    topk = np.asarray([
        [[0, 1], [1, 2]],
        [[0, 1], [3, 2]],
        [[0, 1], [5, 6]],
    ], dtype=np.int64)
    assignments, report = resolve_semantic_id_collisions(
        ["a", "b", "c"],
        base,
        TokenSpace(num_levels=2, codebook_size=8, max_conflict_tokens=4),
        train_counts=np.asarray([10, 5, 1]),
        topk_codes=topk,
        length_mode="fixed",
    )
    lengths = {len(row.token_ids) for row in assignments.values()}
    assert lengths == {3}
    assert report.length_mode == "fixed"
    assert report.min_sid_length == report.mean_sid_length == report.max_sid_length == 3
    assert report.fixed_suffix_tokens == 3


def test_collision_resolution_can_search_all_levels():
    # Last-level alternatives are reserved, so the duplicate must move level 0.
    base = np.asarray([[1, 2], [1, 2], [1, 3]], dtype=np.int64)
    topk = np.asarray([
        [[1, 2], [1, 2]],
        [[2, 1], [2, 2]],
        [[1, 3], [1, 3]],
    ], dtype=np.int64)
    assignments, report = resolve_semantic_id_collisions(
        ["a", "b", "c"],
        base,
        TokenSpace(num_levels=2, codebook_size=8, max_conflict_tokens=4),
        train_counts=np.asarray([10, 5, 1]),
        topk_codes=topk,
        length_mode="fixed",
        reallocation_mode="all",
    )
    assert assignments["b"].base_codes == [2, 2]
    assert report.semantic_reallocated == 1
    assert report.suffix_resolved == 0
    assert report.reallocation_mode == "all"
