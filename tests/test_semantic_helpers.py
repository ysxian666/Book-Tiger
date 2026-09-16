from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from tiger_overnight.semantic_ids import _candidate_bases
from tiger_rec.semantic_id.token_space import TokenSpace


def test_candidate_base_generation_is_bounded_and_unique():
    base = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    topk = np.array([
        [[1, 7, 8], [2, 9, 10], [3, 11, 12]],
        [[4, 13, 14], [5, 15, 16], [6, 17, 18]],
    ], dtype=np.int64)
    candidates = _candidate_bases(base, topk, limit=4, top_codes_per_level=3)
    assert len(candidates) == 2
    assert all(len(values) <= 4 for values in candidates)
    assert all(len({tuple(value) for value in values}) == len(values) for values in candidates)


def test_fixed_sid_has_four_tokens():
    space = TokenSpace(num_levels=3, codebook_size=256, max_conflict_tokens=32, use_behavior_tokens=False, use_time_buckets=True)
    tokens = space.encode_sid([1, 2, 3], suffix_index=0)
    assert len(tokens) == 4
    assert space.decode_sid_tokens(tokens) == ([1, 2, 3], 0)
