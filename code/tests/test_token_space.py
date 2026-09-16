import pytest

from tiger_rec.semantic_id.token_space import BOS, EOS, PAD, TokenSpace


def test_token_space_roundtrip():
    space = TokenSpace(num_levels=3, codebook_size=16, max_conflict_tokens=4)
    tokens = space.encode_sid([1, 5, 15], suffix_index=2)
    codes, suffix = space.decode_sid_tokens(tokens)
    assert codes == [1, 5, 15]
    assert suffix == 2
    assert space.vocab_size > tokens[-1]


def test_special_tokens_are_distinct():
    space = TokenSpace(num_levels=3, codebook_size=16)
    assert PAD == 0 and BOS == 1 and EOS == 2
    assert space.sid_token(0, 0) == 3


def test_user_tokens_do_not_shift_sid_decoding():
    space = TokenSpace(
        num_levels=3, codebook_size=16, max_conflict_tokens=4,
        use_user_tokens=True, num_user_buckets=8,
    )
    tokens = space.encode_sid([1, 5, 15], suffix_index=2)
    codes, suffix = space.decode_sid_tokens(tokens)
    assert codes == [1, 5, 15]
    assert suffix == 2
    assert space.user_token(7) == space.user_base + 7
    assert space.user_token(7) < space.behavior_base < space.time_base < space.vocab_size
