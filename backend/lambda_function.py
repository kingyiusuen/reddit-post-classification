import json
import logging
import pickle
import re
import string


logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
    return text


def lambda_handler(event, context):
    logger.info(event)

    try:
        logger.info(type(event))
        body = event["body"]
        logger.info(type(body))
        text = body["text"]
        text = clean_text(text)
        logger.info(text)
        probs = model.predict_proba([text])[0]
        return {
            "status_code": 200,
            "body": {
                "message": "Success",
                "predictions": {
                    "r/MachineLearning": probs[0],
                    "r/LearnMachineLearning": probs[1],
                },
            },
        }
    except Exception as e:
        logger.error(e)
        return {
            "status_code": 500,
            "body": {"message": "Something went wrong."},
        }
