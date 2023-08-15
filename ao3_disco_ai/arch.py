from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ao3_disco_ai.structs import DenseMeta, SparseMeta


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
    def __init__(self, cardinality: int, max_hash_size: int, embedding_dim: int):
        super().__init__()
        self.max_hash_size = max_hash_size
        hidden_dims = max(16, int(1.6 * np.sqrt(cardinality)))
        self._embed = nn.EmbeddingBag(
            min(cardinality, max_hash_size),
            hidden_dims,
            max_norm=1.0,
            mode="sum",
        )
        self._proj = nn.Linear(hidden_dims, embedding_dim)

    def forward(self, id_list: torch.tensor, offsets: torch.tensor) -> torch.tensor:
        return self._proj(self._embed(id_list % self.max_hash_size, offsets))


class SparseNNV0(nn.Module):
    def __init__(
        self,
        dense_meta: DenseMeta,
        sparse_meta: SparseMeta,
        embedding_dims: int,
        max_hash_size: int,
        use_interactions: bool,
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

    def forward(
        self, dense: torch.tensor, sparse: Dict[str, Tuple[torch.tensor, torch.tensor]]
    ):
        embeddings = [self.dense_arch(dense)]
        for name, sparse_arch in self.sparse_archs.items():
            id_list, offsets = sparse[name]
            embeddings.append(sparse_arch(id_list, offsets))
        if self.use_interactions:
            embeddings.append(build_interactions(embeddings))
        return self.over_arch(torch.cat(embeddings, dim=1))
