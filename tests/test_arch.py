import torch

from ao3_disco_ai.arch import build_interactions


def test_build_interactions():
    embeddings = [
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[2.0, 1.0]]),
    ]
    result = build_interactions(embeddings)
    assert result.shape == torch.Size([1, 3])
    assert result[:, 0] == torch.tensor([0.0])
    assert result[:, 1] == torch.tensor([1.0])
    assert result[:, 2] == torch.tensor([2.0])
