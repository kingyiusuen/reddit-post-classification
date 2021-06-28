from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient

from backend import api
from backend.api import app
from reddit_post_classification.data import Tokenizer
from reddit_post_classification.models import LitModel


client = TestClient(app)


def test_load_artifacts():
    api.load_artifacts()
    assert isinstance(api.tokenizer, Tokenizer)
    assert isinstance(api.model, LitModel)
    assert isinstance(api.labels, list)


def test_index():
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert response.json()["message"] == HTTPStatus.OK.phrase


@pytest.mark.parametrize(
    "title, selftext",
    [
        ("How do I become a data scientist?", "I love data science."),
        ("How do I become a machine learning engineer?", None),
    ],
)
def test_predict(title, selftext):
    data = {
        "title": title,
        "selftext": selftext,
    }
    response = client.post("/predict", json=data)
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == "POST"
    assert sum(
        pred["probability"] for pred in response.json()["data"]["predictions"]
    ) == pytest.approx(1, abs=1e-3)


@pytest.mark.parametrize(
    "title, selftext",
    [
        (None, None),
        (None, "This should not work"),
    ],
)
def test_empty_predict(title, selftext):
    data = {
        "title": title,
        "selftext": selftext,
    }
    response = client.post("/predict", json=data)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.request.method == "POST"
