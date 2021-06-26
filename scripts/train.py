from typing import List, Optional

import hydra
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase

from reddit_post_classification.model import LitModel
from reddit_post_classification.utils import (
    extras,
    log_artifacts,
    log_hyperparams,
)


def main(cfg: DictConfig) -> None:
    if "seed" in cfg:
        seed_everything(cfg.seed, workers=True)

    # Set up datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Set up model
    model: nn.Module = hydra.utils.instantiate(
        cfg.model,
        num_labels=len(datamodule.labels),  # type: ignore
        vocab_size=len(datamodule.tokenizer),  # type: ignore
        padding_idx=datamodule.tokenizer.pad_index,  # type: ignore
    )

    lit_model = LitModel(
        model=model,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
    )

    # Set up callbacks
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for callbacks_cfg in cfg["callbacks"].values():
            if "_target_" in callbacks_cfg:
                callbacks.append(hydra.utils.instantiate(callbacks_cfg))

    # Set up logger
    logger: Optional[LightningLoggerBase] = None
    if "logger" in cfg:
        logger = hydra.utils.instantiate(cfg.logger)

    # Set up trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Log hyperparamaeters
    if logger:
        log_hyperparams(cfg=cfg, lit_model=lit_model, trainer=trainer)

    # Train model
    trainer.fit(lit_model, datamodule=datamodule)

    # Evaluate model
    trainer.test(lit_model, datamodule=datamodule)

    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            log_artifacts(datamodule, trainer, callback)

    return trainer.callback_metrics[cfg.monitor]


@hydra.main(config_path="../configs", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    extras(cfg)
    return main(cfg)


if __name__ == "__main__":
    hydra_entry()
