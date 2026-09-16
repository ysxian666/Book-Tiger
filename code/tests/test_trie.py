from tiger_rec.semantic_id.trie import SIDTrie


def test_trie_constraints():
    trie = SIDTrie()
    trie.insert([3, 20, 37], "a")
    trie.insert([3, 20, 38], "b")
    assert trie.allowed_next([]) == [3]
    assert trie.allowed_next([3, 20]) == [37, 38]
    assert trie.items_for_tokens([3, 20, 37]) == ["a"]
    assert not trie.is_terminal([3, 20])
