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
        dense, sparse, indices, y_true = batch
        embeddings = self.arch(dense, sparse)

        # Ranking loss
        loss = 0.0
        for (anchor_idx, start_idx, end_idx), y in zip(indices, y_true):
            y_pred = 10.0 * F.cosine_similarity(
                embeddings[anchor_idx], embeddings[start_idx:end_idx]
            )
            y_true = F.softmax(y, dim=0)
            y_pred = F.softmax(y_pred, dim=0)
            loss += -torch.sum(y_true * torch.log(y_pred))
        loss /= len(indices)

        self.log("train_loss", loss)

        # Sparse regularization loss
        if self.hparams.similarity_loss_scale > 0:
            sparse_noisy = {}
            for name in sparse:
                id_list, offsets = sparse[name]
                zero_mask = torch.rand(len(id_list)) < 0.25  # Corrupt 25% of the tags.
                random_value = torch.randint(
                    int(torch.min(id_list)), int(torch.max(id_list)), (len(id_list),)
                )
                id_list = id_list * (~zero_mask) + random_value * zero_mask
                sparse_noisy[name] = (id_list, offsets)
            embeddings_noisy = self.arch(dense, sparse_noisy)
            regularization_loss = -F.cosine_similarity(
                embeddings, embeddings_noisy
            ).mean()
            self.log("regularization_loss", regularization_loss)
            loss = loss + self.hparams.similarity_loss_scale * regularization_loss

        return loss

    def validation_step(self, batch, batch_idx):
        dense, sparse, indices, y_true = batch
        embeddings = self.arch(dense, sparse)

        loss = 0.0
        for (anchor_idx, start_idx, end_idx), y in zip(indices, y_true):
            y_pred = 10.0 * F.cosine_similarity(
                embeddings[anchor_idx], embeddings[start_idx:end_idx]
            )
            y_true = F.softmax(y, dim=0)
            y_pred = F.softmax(y_pred, dim=0)
            loss += -torch.sum(y_true * torch.log(y_pred))
        loss /= len(indices)

        self.log("val_loss", loss, batch_size=len(batch))
        self.log("hp_metric", loss, batch_size=len(batch))
