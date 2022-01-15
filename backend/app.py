import logging
import pickle
import re
import string

from flask import Flask, jsonify, request
from flask_cors import CORS
from nltk.corpus import stopwords


logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = Flask(__name__)
CORS(app)

stop_words = set(stopwords.words("english"))


# Load saved model
with open("artifacts/model.pickle", "rb") as f:
    model = pickle.load(f)


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    translator = str.maketrans(
        string.punctuation, " " * len(string.punctuation)
    )
    text = text.translate(translator)
    text = re.sub(r"\s{1,}", " ", text)
    text = text.strip()
    text = text.lower()
    text = " ".join(word for word in text.split() if not word in stop_words)
    return text


@app.route("/")
def index():
    """Provide simple health check route."""
    return "Hello, world!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        raise ValueError("No data is received")
    logger.info(data["text"])
    text = clean_text(data["text"])
    probs = model.predict_proba([text])[0]
    return jsonify(
        {
            "predictions": {
                "r/MachineLearning": probs[0],
                "r/LearnMachineLearning": probs[1],
            }
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
