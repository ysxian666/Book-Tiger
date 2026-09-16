"""Prefix trie used for valid Semantic ID lookup and constrained decoding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    item_ids: list[str] = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        return bool(self.item_ids)


class SIDTrie:
    def __init__(self) -> None:
        self.root = TrieNode()
        self.max_depth = 0

    def insert(self, token_ids: Iterable[int], item_id: str) -> None:
        tokens = [int(token) for token in token_ids]
        if not tokens:
            raise ValueError("empty Semantic ID is not allowed")
        node = self.root
        for token in tokens:
            node = node.children.setdefault(token, TrieNode())
        node.item_ids.append(item_id)
        self.max_depth = max(self.max_depth, len(tokens))

    def node_for_prefix(self, prefix: Iterable[int]) -> TrieNode | None:
        node = self.root
        for token in prefix:
            node = node.children.get(int(token))
            if node is None:
                return None
        return node

    def allowed_next(self, prefix: Iterable[int], include_eos: bool = False) -> list[int]:
        node = self.node_for_prefix(prefix)
        if node is None:
            return []
        allowed = sorted(node.children)
        if include_eos and node.terminal:
            allowed.append(-1)  # EOS sentinel, resolved by the decoder.
        return allowed

    def is_terminal(self, tokens: Iterable[int]) -> bool:
        node = self.node_for_prefix(tokens)
        return bool(node and node.terminal)

    def items_for_tokens(self, tokens: Iterable[int]) -> list[str]:
        node = self.node_for_prefix(tokens)
        return list(node.item_ids) if node else []

    def contains(self, tokens: Iterable[int]) -> bool:
        return bool(self.items_for_tokens(tokens))

    def __len__(self) -> int:
        count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            count += len(node.item_ids)
            stack.extend(node.children.values())
        return count
