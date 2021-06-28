import tempfile

import pytest

from reddit_post_classification.data import Tokenizer


def test_add_token_and_len():
    tokenizer = Tokenizer()
    assert tokenizer._add_token("a") == 2
    assert tokenizer._add_token("a") == 2
    assert len(tokenizer) == 3


def test_train_tokenizer():
    tokenizer = Tokenizer(do_lowercase=True, min_frequency=2, capacity=5)
    tokenizer.train(["machine learning", "data", "learning data science"])
    for token in ["data", "learning"]:
        assert token in tokenizer.token_to_index
    for token in ["machine", "science"]:
        assert token not in tokenizer.token_to_index


def test_train_full_tokenizer():
    tokenizer = Tokenizer(do_lowercase=True, capacity=2)
    with pytest.raises(Exception):
        tokenizer.train(["analysis"])


def test_encode_and_decode():
    tokenizer = Tokenizer(min_frequency=1)
    tokenizer.train(["a", "b", "c"])
    assert tokenizer.encode(["a", "b", "d"]) == [2, 3, 1]
    assert tokenizer.decode([2, 3, 1, 0]) == ["a", "b"]
    with pytest.raises(Exception):
        tokenizer.decode([5])


def test_batch_encode_and_decode():
    tokenizer = Tokenizer(min_frequency=1)
    tokenizer.train(["a", "b", "c"])
    assert tokenizer.batch_encode([["a", "b", "d"], ["c", "a"]]) == [
        [2, 3, 1],
        [4, 2],
    ]
    assert tokenizer.batch_decode([[2, 3, 1, 0], [4, 2]]) == [
        ["a", "b"],
        ["c", "a"],
    ]
    with pytest.raises(Exception):
        tokenizer.decode([[2, 5], [1]])


@pytest.mark.parametrize(
    "max_length, padding, expected",
    [
        (None, "longest", [[2, 2, 2], [2, 2, 0], [2, 0, 0]]),
        (2, "longest", [[2, 2], [2, 2], [2, 0]]),
        (4, "max_length", [[2, 2, 2, 0], [2, 2, 0, 0], [2, 0, 0, 0]]),
        (None, "max_length", ValueError),
        (None, "will_cause_an_error", ValueError),
    ],
)
def test_pad(max_length, padding, expected):
    tokenizer = Tokenizer(min_frequency=1)
    tokenizer.train(["a", "b", "c"])
    token_ids = [[2, 2, 2], [2, 2], [2]]
    if type(expected) == type and issubclass(expected, Exception):
        with pytest.raises(expected):
            tokenizer.pad(token_ids, max_length, padding)
    else:
        assert tokenizer.pad(token_ids, max_length, padding) == expected


def test_save_and_load():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tokenizer1 = Tokenizer()
        tokenizer1.save(f"{tmp_dir}/tokenizer.json")
        tokenizer2 = Tokenizer.load(f"{tmp_dir}/tokenizer.json")
    return tokenizer1.__dict__ == tokenizer2.__dict__
