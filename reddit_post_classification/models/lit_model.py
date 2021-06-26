from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import Accuracy


class LitModel(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict(
            {
                "acc": Accuracy(),
                # "prec": Precision(),
                # "recall": Recall(),
            }
        )

    def forward(self, token_ids):
        logits = self.model(token_ids)
        probs = torch.softmax(logits, dim=1)
        return probs

    def training_step(self, batch, batch_idx):
        logits = self.model(batch["token_ids"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("train/loss", loss.item(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.common_step("val", batch)

    def test_step(self, batch, batch_idx):
        return self.common_step("test", batch)

    def common_step(self, mode: str, batch: Any) -> Tensor:
        logits = self.model(batch["token_ids"])
        loss = self.loss_fn(logits, batch["labels"])
        probs = torch.softmax(logits, dim=1)
        metric_dict = self.compute_metrics(probs, batch["labels"], mode)
        self.log_dict(metric_dict, on_epoch=True, prog_bar=True)
        self.log(f"{mode}/loss", loss.item(), on_epoch=True, prog_bar=True)
        return loss

    def compute_metrics(
        self, probs: Tensor, labels: Tensor, mode: str
    ) -> Dict:
        return {
            f"{mode}/{name}": metric(probs, labels).item()
            for name, metric in self.metrics.items()
        }

    def configure_optimizers(self) -> Dict:
        optimizer = hydra.utils.instantiate(
            self.optimizer_cfg, params=self.model.parameters()
        )
        scheduler = hydra.utils.instantiate(
            self.scheduler_cfg,
            optimizer=optimizer,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }
