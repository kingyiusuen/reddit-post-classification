import math
from typing import Any, Dict, List, Optional

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import Accuracy


class LitModel(LightningModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig,
    ) -> None:
        """A class that specifies the training loop.

        Args:
            model_cfg (DictConfig): Model configurations.
            optimizer_cfg (DictConfig): Optimizer configurations.
            scheduler_cfg (DictConfig): Scheduler configurations.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = hydra.utils.instantiate(model_cfg)
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
        """A forward pass for inference."""
        logits = self.model(token_ids)
        probs = torch.softmax(logits, dim=1)
        return probs

    def training_step(self, batch, batch_idx):
        """A step in the training loop."""
        logits = self.model(batch["token_ids"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("train/loss", loss.item(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """A step in the validation loop."""
        return self.common_step("val", batch)

    def test_step(self, batch, batch_idx):
        """A step in the test loop."""
        return self.common_step("test", batch)

    def common_step(self, mode: str, batch: Any) -> Tensor:
        """A common step shared between validation and test steps."""
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
        """Compute metrics given predictions and ground truth labels.

        Args:
            probs (Tensor): (batch_size, num_labels).
            labels (Tensor): (batch_size).
            mode (str): {'val', 'test'}.
        """
        return {
            f"{mode}/{name}": metric(probs, labels).item()
            for name, metric in self.metrics.items()
        }

    def configure_optimizers(self) -> Dict:
        """Configure optimizer(s) and learning-rate scheduler(s)."""
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


class CNN(nn.Module):
    def __init__(
        self,
        num_labels: int,
        vocab_size: int,
        embedding_dim: int = 128,
        kernel_sizes: List[int] = [2, 4, 6, 8, 10],
        num_kernels: int = 128,
        dropout: float = 0.2,
        padding_idx: Optional[int] = None,
    ) -> None:
        """1-D Convoluational Neural Netowrk.

        Args:
            num_labels (int): Number of classes.
            vocab_size (int): Size of the dictionary of embeddings.
            embedding_dim (int): The size of each embedding vector.
            kernel_sizes (List[int]): A list of kernel sizes.
            num_kernels (int): Number of kernels for each kernel size.
            dropout (float): Probability of an element to be zeroed.
            padding_idx (Optional[int]): The index of the padding token.
        """
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


rnn_dict = {
    "rnn": nn.RNN,
    "gru": nn.GRU,
    "lstm": nn.LSTM,
}


class RNN(nn.Module):
    def __init__(
        self,
        num_labels: int,
        vocab_size: int,
        embedding_dim: int = 128,
        rnn_type: str = "RNN",
        rnn_hidden_dim: int = 128,
        rnn_dropout: float = 0.2,
        rnn_num_layers: int = 2,
        bidirectional: bool = True,
        padding_idx: Optional[int] = None,
    ) -> None:
        """Recurrent Neural Netowrk.

        Args:
            num_labels (int): Number of classes.
            vocab_size (int): Size of the dictionary of embeddings.
            embedding_dim (int): The size of each embedding vector.
            rnn_type (str): The type of RNN cell to use. Accepts the following
                values: `RNN`, `LSTM`, `GRU` (case-insensitive).
            rnn_hidden_dim (int): The number of features in the hidden state.
            rnn_dropout (float): Probability of an element to be zeroed in a
                RNN layer.
            rnn_num_layers (int): Number of recurrent layers.
            bidirectional (bool): Use a bidirectional RNN or not.
            padding_idx (Optional[int]): The index of the padding token.
        """
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.rnn_type = rnn_type.lower()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.rnn = rnn_dict[self.rnn_type](
            input_size=embedding_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.fc = nn.Linear(
            in_features=rnn_hidden_dim * self.num_directions,
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
        embedded = self.embeddings(token_ids)
        # (batch_size, seq_len, embedding_dim)

        if self.rnn_type == "lstm":
            _, (hidden, _) = self.rnn(embedded)
        else:
            _, hidden = self.rnn(embedded)
        # (num_directions * num_layers, batch_size, rnn_hidden_dim)

        if self.num_directions == 2:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        # (batch_size, rnn_hidden_dim * num_directions)

        logits = self.fc(hidden)
        # (batch_size, num_labels)

        return logits
