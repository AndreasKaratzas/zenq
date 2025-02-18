from typing import Tuple

import pytorch_lightning as pl
from generator import DataConfig, SplineDataModule
from modules import CubicSpline, SplineConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def train_spline_model(
    spline_config: SplineConfig = SplineConfig(),
    data_config: DataConfig = DataConfig(),
    max_epochs: int = 1000,
    gpus: int = 1,
    log_dir: str = "logs",
) -> Tuple[CubicSpline, SplineDataModule]:
    """Train the spline model with the given configuration."""

    # Initialize model and data
    model = CubicSpline(spline_config)
    datamodule = SplineDataModule(data_config)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="spline-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        ),
        EarlyStopping(monitor="val_loss", patience=20, mode="min"),
    ]

    # Setup logger
    logger = TensorBoardLogger(log_dir, name="spline_model")

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else None,
        callbacks=callbacks,
        logger=logger,
        precision=16 if gpus > 0 else 32,
    )

    # Train and test
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

    return model, datamodule
