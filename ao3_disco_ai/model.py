from random import random

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from ao3_disco_ai.arch import SparseNNV0


class DiscoModel(pl.LightningModule):
    def __init__(
        self,
        feature_store,
        embedding_dims: int,
        max_hash_size: int,
        use_interactions: bool,
        learning_rate: float,
        similarity_loss_scale: float,
    ):
        super().__init__()
        self.arch = SparseNNV0(
            *feature_store.metadata(),
            embedding_dims=embedding_dims,
            max_hash_size=max_hash_size,
            use_interactions=use_interactions,
        )
        self.feature_store = feature_store
        self.save_hyperparameters(ignore=["feature_store"])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch):
        dense, sparse, indices, y_true = self._batch_to_xy(batch)
        embeddings = self.arch(torch.FloatTensor(dense), sparse)

        # Ranking loss
        loss = 0.0
        for (anchor_idx, start_idx, end_idx), y in zip(indices, y_true):
            y_pred = 10.0 * F.cosine_similarity(
                embeddings[anchor_idx], embeddings[start_idx:end_idx]
            )
            y_true = F.softmax(torch.FloatTensor(y).to(y_pred.device), dim=0)
            y_pred = F.softmax(y_pred, dim=0)
            loss += -torch.sum(y_true * torch.log(y_pred))
        loss /= len(indices)

        self.log("train_loss", loss)

        # Sparse regularization loss
        if self.hparams.similarity_loss_scale > 0:
            sparse_noisy = {}
            for name in sparse:
                sparse_noisy[name] = []
                for row in sparse[name]:
                    # Randomly drop some tags
                    row = [x for x in row if random() < 0.5]
                    sparse_noisy[name].append(row)
            embeddings_noisy = self.arch(
                torch.FloatTensor(dense).to(y_pred.device), sparse_noisy
            )
            regularization_loss = -F.cosine_similarity(
                embeddings, embeddings_noisy
            ).mean()
            self.log("regularization_loss", regularization_loss)
            loss = loss + self.hparams.similarity_loss_scale * regularization_loss

        return loss

    def validation_step(self, batch, batch_idx):
        dense, sparse, indices, y_true = self._batch_to_xy(batch)
        embeddings = self.arch(torch.FloatTensor(dense), sparse)

        loss = 0.0
        for (anchor_idx, start_idx, end_idx), y in zip(indices, y_true):
            y_pred = 10.0 * F.cosine_similarity(
                embeddings[anchor_idx], embeddings[start_idx:end_idx]
            )
            y_true = F.softmax(torch.FloatTensor(y).to(y_pred.device), dim=0)
            y_pred = F.softmax(y_pred, dim=0)
            loss += -torch.sum(y_true * torch.log(y_pred))
        loss /= len(indices)

        self.log("val_loss", loss, batch_size=len(batch))
        self.log("hp_metric", loss, batch_size=len(batch))

    def _batch_to_xy(self, batch):
        """Transform the batch into dense/sparse/y_true representation.

        Each batch is a list of objects where each object has the anchor work plus a
        list of candidates with scores. This transforms the batch into a set of
        dense and sparse features which can be passed through an embedding arch. The
        tensors are ordered as:

            [row_1:anchor]
            [row_1:candidate_1]
            [row_1:candidate_2]
            ...
            [row_2:anchor]
            [row_2:candidate_1]
            ...

        The indices are a list of tuples where each tuple contains the index of the
        anchor plus the start/end index for the candidates. This allows the embeddings
        to be fetched and compared against y_true, where each row in y_true indicates
        the scores for each candidate.
        """
        work_ids, indices, y_true = [], [], []
        for row in batch:
            candidates, scores = zip(*row["candidates"].items())

            anchor_idx = len(work_ids)
            start_idx = len(work_ids) + 1
            end_idx = len(work_ids) + len(candidates) + 1
            indices.append((anchor_idx, start_idx, end_idx))

            work_ids.append(row["work"])
            work_ids.extend(candidates)

            y_true.append(scores)
        dense, sparse = self.feature_store.get_features(work_ids)
        return dense, sparse, indices, y_true
