import os
import pickle
from glob import glob

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset

from ao3_disco_ai.utils import convert_id_lists


class DiscoDataset(Dataset):
    def __init__(self, data):
        super(Dataset).__init__()
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class DiscoDataModule(pl.LightningDataModule):
    def __init__(self, dev: bool, batch_size: int):
        super().__init__()
        data_dir = "data/dev" if dev else "data/prod"
        data_dir = glob(os.path.join(data_dir, "version_*"))[-1]
        self.save_hyperparameters({"dataset": data_dir})

        with open(os.path.join(data_dir, "features.pkl"), "rb") as fin:
            self.feature_store = pickle.load(fin)
        with open(os.path.join(data_dir, "train.pkl"), "rb") as fin:
            self.train_rows = pickle.load(fin)
        with open(os.path.join(data_dir, "val.pkl"), "rb") as fin:
            self.val_rows = pickle.load(fin)

        self.save_hyperparameters()

    def setup(self, stage: str):
        self.test = DiscoDataset(self.train_rows)
        self.val = DiscoDataset(self.val_rows)

    def train_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            collate_fn=self._batch_to_xy,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            collate_fn=self._batch_to_xy,
            num_workers=0,
        )

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

            y_true.append(torch.FloatTensor(scores))
        dense, sparse = self.feature_store.get_features(work_ids)

        pt_dense = torch.FloatTensor(dense)
        pt_sparse = {}
        for name in sparse:
            pt_sparse[name] = convert_id_lists(sparse[name])

        return pt_dense, pt_sparse, indices, y_true
