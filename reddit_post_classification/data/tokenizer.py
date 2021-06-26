import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, MutableMapping, Optional, Sequence, Union

from reddit_post_classification.utils import get_logger


log = get_logger(__name__)


class Tokenizer:
    """A tokenizer is responsible for converting text into tokens.

    Args:
        unk_token: The token to be used to replace out-of-vocabulary tokens.
        pad_token: The token to be used for padding.
        min_frequency: The minimum frequency a token should have in order to
            be included in the vocabulary.
        capacity: The maximum number of tokens to store, including unknown
            and padding tokens. If None, there is no limit to the number of
            tokens to store.
        token_to_index: The mapping between tokens and indexes.
    """

    def __init__(
        self,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        do_lowercase: bool = True,
        min_frequency: Optional[int] = 10,
        capacity: Optional[int] = None,
        token_to_index: Optional[Dict[str, int]] = None,
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.do_lowercase = do_lowercase
        self.min_frequency = min_frequency
        self.capacity = capacity

        if token_to_index is None:
            self.token_to_index: Dict[str, int] = {}
            self.index_to_token: Dict[int, str] = {}
            self.pad_index = self._add_token(self.pad_token)
            self.unk_index = self._add_token(self.unk_token)
        else:
            self.token_to_index = token_to_index
            self.index_to_token = {
                index: token for token, index in self.token_to_index.items()
            }
            self.pad_index = self.token_to_index[self.pad_token]
            self.unk_index = self.token_to_index[self.unk_token]

    def __len__(self) -> int:
        return len(self.token_to_index)

    def _add_token(self, token: str) -> int:
        """Add one token to the vocabulary.

        This method should not be called outside the class, as there is no
        check for `min_frequency` and `capacity`.

        Args:
            token: The token to be added.

        Returns:
            The index of the input token.
        """
        if token in self.token_to_index:
            return self.token_to_index[token]
        index = len(self)
        self.token_to_index[token] = index
        self.index_to_token[index] = token
        return index

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [
            self.token_to_index.get(token, self.unk_index) for token in tokens
        ]

    def decode(self, token_ids: Sequence[int]) -> List[str]:
        return [
            self.index_to_token.get(index, self.unk_token)
            for index in token_ids
        ]

    def batch_encode(
        self, sequences: Sequence[Sequence[str]]
    ) -> List[List[int]]:
        return [self.encode(tokens) for tokens in sequences]

    def batch_decode(
        self, sequences: Sequence[Sequence[int]]
    ) -> List[List[str]]:
        return [self.decode(token_ids) for token_ids in sequences]

    def pretokenize(self, post: MutableMapping) -> str:
        # Remove tags
        post["title"] = re.sub(r"\[[A-Z]+\]", "", post["title"])
        # Concatenate title and selftext
        text = post["title"] + " " + post["selftext"]
        # Remove URLs
        text = re.sub(r"http\S+", "", text)
        # Replace punctuation with space
        translator = str.maketrans(
            string.punctuation, " " * len(string.punctuation)
        )
        text = text.translate(translator)
        # Transform multiple spaces and \n to a single space
        text = re.sub(r"\s{1,}", " ", text)
        # Strip white spaces at the beginning and at the end
        text = text.strip()
        # Transform to lowercase
        if self.do_lowercase:
            text = text.lower()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        tokens = text.split()
        return tokens

    def pad(
        self,
        sequences: List[List[int]],
        max_length: Optional[int] = None,
        padding: str = "longest",
    ) -> List[List[int]]:
        """Pad sequences.

        Args:
            sequences: Sequences of token ids.
            max_length: Maximum length of the sequence. Sequences longer than
                this will be truncated.
            padding: Padding method. If "max_length", pad to max_length. If
                "longest", pad to the longest sequence. If padding is "longest"
                and max_length is not None, the length of the sequences will be
                min(longest_length, max_length).

        Returns:
            Padded sequences.
        """
        if padding == "max_length":
            if max_length is None:
                raise ValueError(
                    "max_length cannot be None when padding method is "
                    "'max_length'."
                )
            else:
                truncation_length = max_length
        elif padding == "longest":
            longest_length = max(len(sequence) for sequence in sequences)
            if max_length is None:
                truncation_length = longest_length
            else:
                truncation_length = min(max_length, longest_length)
        else:
            raise ValueError("Padding method not recognized.")

        padded_sequences = []
        for token_ids in sequences:
            padded_sequence = [self.pad_index] * truncation_length
            token_ids = token_ids[:truncation_length]
            padded_sequence[: len(token_ids)] = token_ids
            padded_sequences.append(padded_sequence)
        return padded_sequences

    def train(
        self,
        sequences: Union[Sequence[str]],
    ) -> "Tokenizer":
        """Train the tokenizer.

        Args:
            texts: The batch of input sequences.
        """
        n = self.capacity - len(self) if self.capacity else None
        if n is not None and n <= 0:
            return self

        log.info("Training tokenizer...")
        counter: Counter = Counter()
        for text in sequences:
            tokens = self.tokenize(text)
            counter.update(tokens)

        for token, count in counter.most_common(n):
            if self.min_frequency and count < self.min_frequency:
                break
            self._add_token(token)
        log.info(f"Training finished. Current vocab size: {len(self)}.")
        return self

    def save(self, filepath: Union[str, Path]) -> None:
        """Output configurations so that the tokenizer can be reproduced.

        Args:
            filepath: Path to output file.
        """
        filepath = Path(filepath)
        with open(filepath, "w") as fp:
            content = {
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "do_lowercase": self.do_lowercase,
                "min_frequency": self.min_frequency,
                "capacity": self.capacity,
                "token_to_index": self.token_to_index,
            }
            json.dump(content, fp)
        log.info(f"Tokenizer saved to {str(filepath)}.")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Tokenizer":
        """Create a `Tokenizer` from a configuration file outputted by `save`.

        Args:
            filepath: Path to the file to read from.

        Returns:
            A `Tokenizer` object.
        """
        with open(filepath) as f:
            kwargs = json.load(f)
        return cls(**kwargs)
