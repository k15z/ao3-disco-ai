import pickle

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from ao3_disco_ai.feature import FeatureStore


class DiscoDataset(Dataset):
    def __init__(self, data):
        super(Dataset).__init__()
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


class DiscoDataModule(pl.LightningDataModule):
    def __init__(self, small_world: bool = True, batch_size: int = 100):
        super().__init__()
        with open(
            "data/small-world.pkl" if small_world else "data/dataset.pkl", "rb"
        ) as fin:
            work_to_json, self.train_rows, self.val_rows = pickle.load(fin)
        self.feature_store = FeatureStore(work_to_json)
        self.save_hyperparameters()

    def setup(self, stage: str):
        self.test = DiscoDataset(self.train_rows)
        self.val = DiscoDataset(self.val_rows)

    def train_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            collate_fn=lambda x: x,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.hparams.batch_size, collate_fn=lambda x: x
        )
