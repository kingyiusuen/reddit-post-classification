import math
from typing import Any, Dict, List, Optional

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import Accuracy


class CNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 128,
        kernel_sizes: List[int] = [2, 4, 6, 8, 10],
        num_kernels: int = 128,
        dropout: float = 0.2,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.convs = nn.ModuleList(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_kernels,
                kernel_size=kernel_size,
            )
            for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            in_features=num_kernels * len(kernel_sizes),
            out_features=num_labels,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """Forward pass.

        Args:
            token_ids: (batch_size, seq_len). Indices of input sequence tokens
                in the vocabulary.

        Returns:
            logits: (batch_size, num_labels). Log odds of the class labels.
        """
        seq_len = token_ids.shape[1]

        embedded = self.embeddings(token_ids)
        # (batch_size, seq_len, embedding_dim)

        embedded = embedded.permute(0, 2, 1)
        # (batch_size, embedding_dim, seq_len)

        x = []
        for conv, kernel_size in zip(self.convs, self.kernel_sizes):
            padding = conv.stride[0] * (seq_len - 1) - seq_len + kernel_size
            padding /= 2
            padding_left = int(padding)
            padding_right = int(math.ceil(padding))

            _x = F.pad(embedded, (padding_left, padding_right))

            _x = F.relu(conv(_x))
            # (batch_size, num_kernels, seq_len)

            _x = F.max_pool1d(_x, _x.shape[2]).squeeze(2)
            # (batch_size, num_kernels)

            x.append(_x)

        x = self.dropout(torch.cat(x, dim=1))
        # (batch_size, num_kernels * len(kernel_sizes))

        logits = self.fc(x)
        # (batch_size, num_labels)

        return logits


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
        self.log_dict(metric_dict, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            f"{mode}/loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def setup(self, stage: Optional[str] = None):
        self.configure_metrics()

    def configure_metrics(self) -> None:
        self.metrics = nn.ModuleDict(
            {
                "acc": Accuracy(),
                # "prec": Precision(),
                # "recall": Recall(),
            }
        )

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
