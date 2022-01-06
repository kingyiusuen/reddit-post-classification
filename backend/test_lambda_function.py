from lambda_function import lambda_handler


def test_valid_input():
    event = {"body": {"text": "I love machine learning."}}
    response = lambda_handler(event, None)
    assert response["status_code"] == 200
    assert len(response["body"]["predictions"]) == 2


def test_invalid_input():
    event = {}
    response = lambda_handler(event, None)
    assert response["status_code"] == 400
