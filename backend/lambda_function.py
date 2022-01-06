import logging
import pickle

logger = logging.getLogger()
logger.setLevel(logging.INFO)


with open("artifacts/model.pickle", "rb") as f:
    model = pickle.load(f)


def lambda_handler(event, context):
    if "text" not in event:
        logger.error("Key 'text' not found in event.")
        return {
            "status_code": 400,
            "message": "Missing input.",
        }

    try:
        probs = model.predict_proba([event["text"]])[0]
        return {
            "status_code": 200,
            "message": "Success",
            "payload": {
                "probs": probs,
            },
        }
    except Exception as e:
        logger.error(e)
        return {
            "status_code": 500,
            "message": "Something went wrong.",
        }
