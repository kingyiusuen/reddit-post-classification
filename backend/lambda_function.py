import logging
import pickle
import re
import string


logger = logging.getLogger()
logger.setLevel(logging.INFO)


headers = {
    "Access-Control-Allow-Origin": "master.d12233i7lji2r8.amplifyapp.com"
}


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
    if "text" not in event:
        logger.error("Key 'text' not found in event.")
        return {
            "status_code": 400,
            "headers": headers,
            "body": {"message": "Missing input."},
        }

    try:
        text = clean_text(event["text"])
        probs = model.predict_proba([text])[0]
        return {
            "status_code": 200,
            "headers": headers,
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
            "headers": headers,
            "body": {"message": "Something went wrong."},
        }
