import json
import logging
import tempfile
import warnings
from argparse import Namespace

import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase, WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    log = logging.getLogger(name)
    log.setLevel(level)

    # This ensures all logging levels get marked with the rank zero decorator;
    # otherwise, logs would get multiplied for each GPU process in multi-GPU
    # setup.
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(log, level, rank_zero_only(getattr(log, level)))

    return log


def extras(cfg: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file.

    Options include:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    - forcing multi-gpu friendly configuration
    Modifies DictConfig in place.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(cfg, False)

    # disable python warnings if <cfg.ignore_warnings=True>
    if cfg.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <cfg.trainer.fast_dev_run=True> if <cfg.debug=True>
    if cfg.get("debug"):
        log.info("Running in debug mode! <cfg.debug=True>")
        cfg.trainer.fast_dev_run = True

    # force debugger friendly configuration if <cfg.trainer.fast_dev_run=True>
    if cfg.trainer.get("fast_dev_run"):
        log.info(
            "Forcing debugger friendly configuration! "
            "<cfg.trainer.fast_dev_run=True>"
        )
        # Debuggers don't like GPUs or multiprocessing
        if cfg.trainer.get("gpus"):
            cfg.trainer.gpus = 0
        if cfg.datamodule.get("pin_memory"):
            cfg.datamodule.pin_memory = False
        if cfg.datamodule.get("num_workers"):
            cfg.datamodule.num_workers = 0

    # force multi-gpu friendly configuration if <cfg.trainer.accelerator=ddp>
    accelerator = cfg.trainer.get("accelerator")
    if accelerator in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info(
            "Forcing ddp friendly configuration! "
            f"<cfg.trainer.accelerator={accelerator}>"
        )
        if cfg.datamodule.get("num_workers"):
            cfg.datamodule.num_workers = 0
        if cfg.datamodule.get("pin_memory"):
            cfg.datamodule.pin_memory = False

    # disable adding new keys to config
    OmegaConf.set_struct(cfg, True)


def _empty(*args, **kwargs) -> None:
    """A dummy function."""
    pass


@rank_zero_only
def log_hyperparams(
    cfg: DictConfig,
    lit_model: LightningModule,
    trainer: Trainer,
) -> None:
    """Log hyperparameters."""
    hparams = {}

    # Save Hydra configs
    hparams["trainer"] = cfg.trainer
    hparams["model"] = cfg.model
    hparams["optimizer"] = cfg.optimizer
    hparams["scheduler"] = cfg.scheduler
    hparams["datamodule"] = cfg.datamodule
    if "callbacks" in cfg:
        hparams["callbacks"] = cfg.callbacks

    # Save number of model parameters
    hparams["model/params_total"] = sum(
        p.numel() for p in lit_model.parameters()
    )
    hparams["model/params_trainable"] = sum(
        p.numel() for p in lit_model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in lit_model.parameters() if not p.requires_grad
    )

    # Send hparams to loggers
    trainer.logger.log_hyperparams(Namespace(**hparams))

    # Disable logging any more hyperparameters for all loggers
    # This is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = _empty  # type: ignore


def log_artifacts(
    trainer: Trainer,
    datamodule: LightningDataModule,
    model_checkpoint: ModelCheckpoint,
) -> None:
    """Log artifacts."""
    if isinstance(trainer.logger, WandbLogger):
        wandb.save(model_checkpoint.best_model_path)
        wandb.save(str(datamodule.tokenizer_filename))  # type: ignore
        with tempfile.TemporaryDirectory() as temp_dir:
            label_filename = f"{temp_dir}/subreddit_names.json"
            with open(label_filename, "w") as fp:
                json.dump(list(datamodule.labels), fp)  # type: ignore
            wandb.save(label_filename)
            finish(trainer.logger)


def finish(logger: LightningLoggerBase) -> None:
    """Makes sure everything closed properly."""
    if isinstance(logger, WandbLogger):
        wandb.finish()
