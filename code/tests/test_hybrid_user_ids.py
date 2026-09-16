from tiger_rec.retrieval.base import RetrievalBatch
from tiger_rec.retrieval.hybrid import HybridRetriever


class UserAwareRetriever:
    name = 'user-aware'
    supports_user_ids = True

    def retrieve(self, histories, k, user_ids=None):
        assert user_ids == ['u1']
        return RetrievalBatch(items=[['i1']], scores=[[1.0]], latency_ms=1.0)


def test_hybrid_propagates_user_ids():
    retriever = HybridRetriever([UserAwareRetriever()])
    output = retriever.retrieve([['h1']], 1, user_ids=['u1'])
    assert output.items == [['i1']]
