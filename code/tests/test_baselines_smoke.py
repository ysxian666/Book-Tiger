import torch

from tiger_rec.models.baselines import BERT4Rec, GRU4Rec, SASRec, TwoTower, sampled_softmax_loss


def test_sequential_baselines_forward():
    models = [
        GRU4Rec(12, dim=8, num_layers=1, dropout=0.0),
        SASRec(12, dim=8, num_layers=1, num_heads=2, dropout=0.0, max_len=6),
        BERT4Rec(12, dim=8, num_layers=1, num_heads=2, dropout=0.0, max_len=6),
        TwoTower(12, dim=8, dropout=0.0),
    ]
    sequence = torch.tensor([[1, 2, 3], [4, 5, 0]])
    lengths = torch.tensor([3, 2])
    targets = torch.tensor([4, 6])
    negatives = torch.tensor([[7, 8], [9, 10]])
    for model in models:
        vectors = model.encode(sequence, lengths)
        loss = sampled_softmax_loss(vectors, model, targets, negatives)
        assert vectors.shape == (2, 8)
        assert torch.isfinite(loss)
