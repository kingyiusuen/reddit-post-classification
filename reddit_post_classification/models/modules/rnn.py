from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


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
    ):
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

        if self.num_directions:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        # (batch_size, rnn_hidden_dim * num_directions)

        logits = self.fc(hidden)
        # (batch_size, num_labels)

        return logits
