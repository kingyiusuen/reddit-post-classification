import pytest
import torch

from reddit_post_classification.models.modules import RNN


@pytest.fixture(params=["RNN", "GRU", "LSTM"])
def params(request):
    return {
        "vocab_size": 100,
        "num_labels": 3,
        "embedding_dim": 64,
        "rnn_type": request.param,
        "rnn_hidden_dim": 64,
        "rnn_dropout": 0.2,
        "rnn_num_layers": 2,
        "bidirectional": True,
        "padding_idx": 0,
    }


@pytest.fixture
def model(params):
    return RNN(**params)


def test_init(params, model):
    assert model.num_directions == 2
    assert model.rnn_type == params["rnn_type"].lower()
    assert model.embeddings.weight.shape == (
        params["vocab_size"],
        params["embedding_dim"],
    )
    assert model.embeddings.padding_idx == params["padding_idx"]
    assert model.rnn.hidden_size == params["rnn_hidden_dim"]
    assert model.rnn.dropout == params["rnn_dropout"]
    assert model.rnn.num_layers == params["rnn_num_layers"]
    assert model.rnn.bidirectional == params["bidirectional"]
    assert model.fc.weight.shape == (
        params["num_labels"],
        (params["rnn_hidden_dim"] * model.num_directions),
    )


def test_forward(params, model):
    batch_size = 8
    seq_len = 12
    x = torch.randint(0, params["vocab_size"], (batch_size, seq_len))
    logits = model.forward(x)
    assert logits.shape == (batch_size, params["num_labels"])
