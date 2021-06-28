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
    """Recurrent Neural Netowrk.

    Args:
        num_labels: Number of classes.
        vocab_size: Size of the dictionary of embeddings.
        embedding_dim: The size of each embedding vector.
        rnn_type: The type of RNN cell to use. Accepts the following values:
            `RNN`, `LSTM`, `GRU` (case-insensitive).
        rnn_hidden_dim: The number of features in the hidden state.
        rnn_dropout: Probability of an element to be zeroed in a RNN layer.
        rnn_num_layers: Number of recurrent layers.
        bidirectional: Use a bidirectional RNN or not.
        padding_dix: The index of the padding token.
    """

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
