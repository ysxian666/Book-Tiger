import math

import pytest

from tiger_rec.evaluation.metrics import ndcg_at_k, recall_at_k


def test_recall_and_ndcg():
    recommendations = [["a", "b", "c"], ["x", "y", "z"]]
    targets = ["b", "z"]
    assert recall_at_k(recommendations, targets, 2) == pytest.approx(0.5)
    assert ndcg_at_k(recommendations, targets, 3) == pytest.approx((1 / math.log2(3) + 1 / 2) / 2)
