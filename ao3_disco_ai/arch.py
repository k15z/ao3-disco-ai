from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


def build_interactions(embeddings: List[torch.tensor]) -> torch.tensor:
    """Compute interactions between embeddings.

    Input: List of N embeddings of shape (batch_size, embedding_dim)
    Output: (batch_size, N*(N-1)/2)
    """
    interactions = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            interactions.append(
                torch.sum(embeddings[i] * embeddings[j], axis=1, keepdim=True)
            )
    return torch.cat(interactions, axis=1)


class SparseArch(nn.Module):
    def __init__(self, cardinality, max_hash_size, embedding_dim):
        super().__init__()
        hidden_dims = max(16, int(1.6 * np.sqrt(cardinality)))
        self._embed = nn.EmbeddingBag(
            min(cardinality, max_hash_size),
            hidden_dims,
            max_norm=1.0,
            mode="sum",
        )
        self._proj = nn.Linear(hidden_dims, embedding_dim)

    def forward(self, id_list, offsets):
        return self._proj(self._embed(id_list, offsets))


class SparseNNV0(nn.Module):
    def __init__(
        self,
        dense_meta,
        sparse_meta,
        embedding_dims,
        max_hash_size,
        use_interactions,
    ):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.max_hash_size = max_hash_size
        self.use_interactions = use_interactions
        self.dense_arch = nn.Linear(len(dense_meta), self.embedding_dims)
        self.sparse_archs = {}
        for name, cardinality in sparse_meta.items():
            self.sparse_archs[name] = SparseArch(
                cardinality, self.max_hash_size, self.embedding_dims
            )
        self.sparse_archs = nn.ModuleDict(self.sparse_archs)

        pre_over_dims = self.embedding_dims * (len(self.sparse_archs) + 1)
        if self.use_interactions:
            # For N embeddings, the # of interactions is N choose 2.
            pre_over_dims += (
                (len(self.sparse_archs) + 1) * len(self.sparse_archs)
            ) // 2

        self.over_arch = nn.Linear(pre_over_dims, self.embedding_dims)

    def forward(self, dense: np.ndarray, sparse: Dict[str, List[List[int]]]):
        device = self.dense_arch.weight.device
        embeddings = [self.dense_arch(torch.tensor(dense).to(device))]
        for name, sparse_arch in self.sparse_archs.items():
            id_list, offsets = [], []
            for row in sparse[name]:
                offsets.append(len(id_list))
                id_list.extend([x % self.max_hash_size for x in row])
            id_list, offsets = torch.IntTensor(id_list).to(device), torch.IntTensor(
                offsets
            ).to(device)
            embeddings.append(sparse_arch(id_list, offsets))
        if self.use_interactions:
            embeddings.append(build_interactions(embeddings))
        return self.over_arch(torch.cat(embeddings, dim=1))
