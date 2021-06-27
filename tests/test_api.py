from http import HTTPStatus

from fastapi.testclient import TestClient
from pytest import approx

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


def test_predict():
    data = {
        "title": "How do I become a data scientist?",
        "selftext": "I love data science.",
    }
    response = client.post("/predict", json=data)
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == "POST"
    assert sum(
        pred["probability"] for pred in response.json()["data"]["predictions"]
    ) == approx(1, abs=1e-3)


def test_empty_predict():
    data = {
        "title": None,
        "selftext": None,
    }
    response = client.post("/predict", json=data)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.request.method == "POST"
