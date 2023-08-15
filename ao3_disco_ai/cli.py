import logging

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
import typer

from ao3_disco_ai.data import DiscoDataModule
from ao3_disco_ai.model import DiscoModel

logging.basicConfig(level=logging.INFO)

app = typer.Typer()


@app.command()
def train(
    experiment_name: str = None,
    small_world: bool = True,
    batch_size: int = 100,
    embedding_dims: int = 128,
    max_hash_size: int = 1000,
    use_interactions: bool = False,
    learning_rate: float = 1e-3,
    similarity_loss_scale: float = 0.1,
):
    data = DiscoDataModule(small_world, batch_size)
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
        logger=TensorBoardLogger(save_dir="lightning_logs", name=experiment_name)
    )
    trainer.fit(model=model, datamodule=data)


if __name__ == "__main__":
    app()
