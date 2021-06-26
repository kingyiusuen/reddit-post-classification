import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
