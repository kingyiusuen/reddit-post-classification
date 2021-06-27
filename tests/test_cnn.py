import pytest
import torch

from reddit_post_classification.models.modules import CNN


@pytest.fixture
def params():
    return {
        "vocab_size": 100,
        "num_labels": 3,
        "embedding_dim": 64,
        "kernel_sizes": [1, 2, 3, 4],
        "num_kernels": 64,
        "dropout": 0.2,
        "padding_idx": 0,
    }


@pytest.fixture
def model(params):
    return CNN(**params)


def test_init(params, model):
    assert model.kernel_sizes == params["kernel_sizes"]
    assert model.embeddings.weight.shape == (
        params["vocab_size"],
        params["embedding_dim"],
    )
    assert model.embeddings.padding_idx == params["padding_idx"]
    assert len(model.convs) == len(params["kernel_sizes"])
    assert model.dropout.p == params["dropout"]
    assert model.fc.weight.shape == (
        params["num_labels"],
        (params["num_kernels"] * len(params["kernel_sizes"])),
    )


def test_forward(params, model):
    batch_size = 8
    seq_len = 12
    x = torch.randint(0, params["vocab_size"], (batch_size, seq_len))
    logits = model.forward(x)
    assert logits.shape == (batch_size, params["num_labels"])
