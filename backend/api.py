from http import HTTPStatus
from typing import Dict

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import OmegaConf

from backend.schemas import Post
from reddit_post_classification.data import Tokenizer
from reddit_post_classification.model import LitModel
from reddit_post_classification.utils import get_logger


logger = get_logger(__name__)


app = FastAPI(
    title="Reddit Post Classifier",
    description=(
        "Predict whether a post belongs to r/MachineLearning, r/statistics "
        "or r/datascience"
    ),
    version="0.1",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_artifacts():
    global tokenizer
    global model
    global labels
    cfg = OmegaConf.load("configs/config.yaml")
    labels = cfg.subreddit_names
    tokenizer = Tokenizer.load("artifacts/tokenizer.json")
    model = LitModel.load_from_checkpoint("artifacts/lit_model.ckpt")
    model.freeze()
    logger.info("Artifacts loaded successfully. Ready for inference.")


@app.get("/")
def index():
    """Health check."""
    return "Hello, World!"


@app.post("/predict")
def predict(request: Request, post: Post) -> Dict:
    """Predict which subreddit a post belongs to."""
    text = tokenizer.pretokenize(vars(post))  # type: ignore
    token_ids = tokenizer.encode(text)  # type: ignore
    token_ids = torch.tensor([token_ids])
    probs = model(token_ids).squeeze().tolist()  # type: ignore
    predictions = [
        {"probability": prob, "subreddit": label}
        for label, prob in zip(labels, probs)  # type: ignore
    ]
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": predictions},
    }
    return response
