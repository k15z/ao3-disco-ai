import logging
import os
import pickle
import shutil
from datetime import datetime
from glob import glob
from typing import Optional

import lightning.pytorch as pl
import typer
import yaml
from lightning.pytorch.loggers import TensorBoardLogger

from ao3_disco_ai.data import DiscoDataModule
from ao3_disco_ai.feature import FeatureStore
from ao3_disco_ai.model import DiscoModel
from ao3_disco_ai.wrapper import ModelWrapper

logging.basicConfig(level=logging.DEBUG)

app = typer.Typer()


@app.command()
def clean():
    print("Are you sure? This will result in data loss. (yes/no)")
    if input() != "yes":
        return
    shutil.rmtree("data/dev", ignore_errors=True)
    shutil.rmtree("data/prod", ignore_errors=True)
    shutil.rmtree("lightning_logs", ignore_errors=True)


@app.command()
def preprocess(dev: bool = False):
    # Determine the output dir (create if needed).
    version_num = 0
    data_dir = "data/dev" if dev else "data/prod"
    if os.path.exists(data_dir):
        for path in glob(os.path.join(data_dir, "version_*")):
            version_num = int(path.split("_")[-1]) + 1
    data_dir = os.path.join(data_dir, f"version_{version_num}")
    os.makedirs(data_dir, exist_ok=True)

    # Prepare the dataset.
    work_to_json, train_rows, val_rows = pickle.load(open("data/dataset.pkl", "rb"))
    logging.info("Found %d works.", len(work_to_json))
    logging.info("Found %d train rows.", len(train_rows))
    logging.info("Found %d validation rows.", len(val_rows))
    if dev:
        train_rows = train_rows[:1000]
        val_rows = val_rows[:1000]
        valid_work_ids = set([row["work"] for row in train_rows + val_rows])
        for row in train_rows + val_rows:
            valid_work_ids.update(row["candidates"].keys())
        work_to_json = {k: v for k, v in work_to_json.items() if k in valid_work_ids}
        logging.info("Pruned to %d works.", len(work_to_json))
        logging.info("Pruned to %d train rows.", len(train_rows))
        logging.info("Pruned to %d validation rows.", len(val_rows))

    # Prepare the features.
    features = FeatureStore(work_to_json)

    with open(os.path.join(data_dir, "features.pkl"), "wb") as fout:
        pickle.dump(features, fout)
    with open(os.path.join(data_dir, "train.pkl"), "wb") as fout:
        pickle.dump(train_rows, fout)
    with open(os.path.join(data_dir, "val.pkl"), "wb") as fout:
        pickle.dump(val_rows, fout)
    with open(os.path.join(data_dir, "works.pkl"), "wb") as fout:
        pickle.dump(work_to_json, fout)
    with open(os.path.join(data_dir, "metadata.yaml"), "wt") as fout:
        yaml.dump(
            {
                "version": version_num,
                "created_at": datetime.now().isoformat(),
                "num_works": len(work_to_json),
                "num_train_rows": len(train_rows),
                "num_val_rows": len(val_rows),
            },
            fout,
        )


@app.command()
def train(
    experiment_name: str = None,
    dev: bool = False,
    batch_size: int = 100,
    embedding_dims: int = 128,
    max_hash_size: int = 1000,
    use_interactions: bool = False,
    learning_rate: float = 1e-3,
    similarity_loss_scale: float = 0.1,
):
    data = DiscoDataModule(dev, batch_size)
    model = DiscoModel(
        data.feature_store,
        embedding_dims=embedding_dims,
        max_hash_size=max_hash_size,
        use_interactions=use_interactions,
        learning_rate=learning_rate,
        similarity_loss_scale=similarity_loss_scale,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=10,
        logger=TensorBoardLogger(save_dir="lightning_logs", name=experiment_name),
        # TODO: Investigate adding data parallelism on CPU again.
        # devices=4,
        # strategy="ddp",
    )
    trainer.fit(model=model, datamodule=data)


@app.command()
def export(
    model_dir: str,
    output_path: str = "wrapped_model.pkl",
    checkpoint: Optional[str] = None,
):
    if not checkpoint:
        checkpoint = sorted(glob(os.path.join(model_dir, "checkpoints/*.ckpt")))[-1]
        checkpoint = checkpoint.split("/")[-1]
        logging.info("Checkpoint not specified, using %s.", checkpoint)
    wrapper = ModelWrapper.build(model_dir, checkpoint)
    pickle.dump(wrapper, open(output_path, "wb"))


if __name__ == "__main__":
    app()
