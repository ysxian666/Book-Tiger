"""Integer token space shared by Semantic IDs and the sequence generator."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

PAD = 0
BOS = 1
EOS = 2
SPECIAL_TOKENS = 3


@dataclass(frozen=True)
class TokenSpace:
    num_levels: int
    codebook_size: int
    max_conflict_tokens: int = 16
    use_behavior_tokens: bool = True
    use_time_buckets: bool = True
    use_user_tokens: bool = False
    num_user_buckets: int = 512
    num_time_buckets: int = 16
    behavior_count: int = 3

    @property
    def sid_base(self) -> int:
        return SPECIAL_TOKENS

    @property
    def conflict_base(self) -> int:
        return self.sid_base + self.num_levels * self.codebook_size

    @property
    def user_base(self) -> int:
        return self.conflict_base + self.max_conflict_tokens

    @property
    def behavior_base(self) -> int:
        return self.user_base + (self.num_user_buckets if self.use_user_tokens else 0)

    @property
    def time_base(self) -> int:
        return self.behavior_base + (self.behavior_count if self.use_behavior_tokens else 0)

    @property
    def vocab_size(self) -> int:
        return self.time_base + (self.num_time_buckets if self.use_time_buckets else 0)

    @property
    def max_sid_length(self) -> int:
        return self.num_levels + (1 if self.max_conflict_tokens > 0 else 0)

    def sid_token(self, level: int, code: int) -> int:
        self.validate_level(level)
        if not 0 <= code < self.codebook_size:
            raise ValueError(f"code out of range: {code}")
        return self.sid_base + level * self.codebook_size + code

    def conflict_token(self, suffix_index: int) -> int:
        if not 0 <= suffix_index < self.max_conflict_tokens:
            raise ValueError(f"suffix index out of range: {suffix_index}")
        return self.conflict_base + suffix_index

    def user_token(self, bucket: int) -> int:
        if not self.use_user_tokens:
            raise ValueError("user tokens are disabled")
        if not 0 <= bucket < self.num_user_buckets:
            raise ValueError(f"user bucket out of range: {bucket}")
        return self.user_base + bucket

    def behavior_token(self, behavior: int) -> int:
        if not self.use_behavior_tokens:
            raise ValueError("behavior tokens are disabled")
        if not 0 <= behavior < self.behavior_count:
            raise ValueError(f"behavior out of range: {behavior}")
        return self.behavior_base + behavior

    def time_token(self, bucket: int) -> int:
        if not self.use_time_buckets:
            raise ValueError("time tokens are disabled")
        if not 0 <= bucket < self.num_time_buckets:
            raise ValueError(f"time bucket out of range: {bucket}")
        return self.time_base + bucket

    def encode_base_codes(self, codes: Iterable[int]) -> list[int]:
        codes = list(codes)
        if len(codes) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} codes, received {len(codes)}")
        return [self.sid_token(level, int(code)) for level, code in enumerate(codes)]

    def encode_sid(self, base_codes: Iterable[int], suffix_index: int | None = None) -> list[int]:
        tokens = self.encode_base_codes(base_codes)
        if suffix_index is not None:
            tokens.append(self.conflict_token(int(suffix_index)))
        return tokens

    def decode_sid_tokens(self, tokens: Iterable[int]) -> tuple[list[int], int | None]:
        tokens = list(tokens)
        base_tokens = tokens[:self.num_levels]
        if len(base_tokens) != self.num_levels:
            raise ValueError("incomplete Semantic ID")
        codes = []
        for level, token in enumerate(base_tokens):
            low = self.sid_base + level * self.codebook_size
            high = low + self.codebook_size
            if not low <= token < high:
                raise ValueError(f"token {token} is not a level-{level} Semantic ID token")
            codes.append(token - low)
        suffix = None
        if len(tokens) == self.num_levels + 1:
            token = tokens[-1]
            if not self.conflict_base <= token < self.user_base:
                raise ValueError(f"invalid conflict suffix token: {token}")
            suffix = token - self.conflict_base
        elif len(tokens) != self.num_levels:
            raise ValueError(f"invalid Semantic ID length: {len(tokens)}")
        return codes, suffix

    def all_sid_code_tokens(self) -> list[int]:
        tokens: list[int] = []
        for level in range(self.num_levels):
            start = self.sid_base + level * self.codebook_size
            tokens.extend(range(start, start + self.codebook_size))
        return tokens

    def validate_level(self, level: int) -> None:
        if not 0 <= level < self.num_levels:
            raise ValueError(f"level out of range: {level}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "TokenSpace":
        return cls(**value)
