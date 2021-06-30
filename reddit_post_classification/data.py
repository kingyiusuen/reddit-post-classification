import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, MutableMapping, Optional, Sequence, Union

import pandas as pd
import torch
from omegaconf import ListConfig
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from reddit_post_classification.utils import get_logger


log = get_logger(__name__)


class RedditDataset(Dataset):
    def __init__(self, fname: Union[str, Path]) -> None:
        """A Dataset object.

        Args:
            fname (str, Path): Path to the data file.
        """
        self.data = pd.read_csv(fname)

    def __getitem__(self, idx: int):
        """Returns a sample from the dataset.

        Args:
            idx (int): The index of the sample in the dataset.
        """
        return self.data.iloc[idx]["text"], self.data.iloc[idx]["label"]

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.data)


class RedditDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        labels: Union[List[str], ListConfig],
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_seq_len: int = 512,
    ) -> None:
        """A class that handles data processing.

        Args:
            data_dir (str): Directory to the data.
            labels (List[str], ListConfig): A list of subreddit names.
            batch_size (int): The number samples to load per batch.
            num_workers (int): The number of subprocesses to use for data
                loading.
            pin_memory (bool): Whether to copy Tensors into CUDA pinned memory
                before returning them.
            max_seq_len (int): Maximum number of tokens per sequence.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.labels = labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_seq_len = max_seq_len

        processed_data_dir = self.data_dir / "processed"
        self.data_filenames = {
            "train": processed_data_dir / "train.csv",
            "val": processed_data_dir / "val.csv",
            "test": processed_data_dir / "test.csv",
        }
        self.tokenizer_filename = processed_data_dir / "tokenizer.json"

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training/testing."""
        self.tokenizer = Tokenizer.load(self.tokenizer_filename)

        if stage == "fit" or stage is None:
            log.info("Loading train and val data...")
            self.train_dataset = RedditDataset(self.data_filenames["train"])
            self.val_dataset = RedditDataset(self.data_filenames["val"])

        if stage == "test" or stage is None:
            log.info("Loading test data...")
            self.test_dataset = RedditDataset(self.data_filenames["test"])

    def collate_fn(self, batch) -> Dict[str, Tensor]:
        """Collate a batch of samples."""
        texts, labels = zip(*batch)
        token_ids = self.tokenizer.batch_encode(texts)
        token_ids = self.tokenizer.pad(token_ids, max_length=self.max_seq_len)
        return {
            "token_ids": torch.tensor(token_ids),
            "labels": torch.tensor(labels),
        }

    def train_dataloader(self) -> DataLoader:
        """Returns a dataloader for the train dataset."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns a dataloader for the validation dataset."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns a dataloader for the test dataset."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )


class Tokenizer:
    def __init__(
        self,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        do_lowercase: bool = True,
        min_frequency: Optional[int] = 10,
        capacity: Optional[int] = None,
        token_to_index: Optional[Dict[str, int]] = None,
    ) -> None:
        """A tokenizer is responsible for converting text into tokens.

        Args:
            unk_token (str): The token to be used to replace out-of-vocabulary
                tokens.
            pad_token (str): The token to be used for padding.
            do_lowercase (bool): Whether to convert letters to lowercase.
            min_frequency (Optional[int]): The minimum frequency a token should
                have in order to be included in the vocabulary.
            capacity (Optional[int]): The maximum number of tokens to store,
                including unknown and padding tokens. If None, there is no
                limit to the number of tokens to store.
            token_to_index (Optional[Dict[str, int]]): The mapping between
                tokens and indexes.
        """
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
        """Returns the vocab size."""
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
        """Convert a sequence of tokens into a sequence of indices."""
        return [
            self.token_to_index.get(token, self.unk_index) for token in tokens
        ]

    def decode(self, token_ids: Sequence[int]) -> List[str]:
        """Convert a sequence of indices into a sequence of tokens."""
        tokens = []
        for index in token_ids:
            if index not in self.index_to_token:
                raise RuntimeError(f"Found a new index {index}.")
            if index in [self.pad_index, self.unk_index]:
                continue
            tokens.append(self.index_to_token[index])
        return tokens

    def batch_encode(
        self, sequences: Sequence[Sequence[str]]
    ) -> List[List[int]]:
        """Convert sequences of tokens into sequences of indices."""
        return [self.encode(tokens) for tokens in sequences]

    def batch_decode(
        self, sequences: Sequence[Sequence[int]]
    ) -> List[List[str]]:
        """Convert sequences of indices into sequences of tokens."""
        return [self.decode(token_ids) for token_ids in sequences]

    def pretokenize(self, post: MutableMapping) -> str:
        """Text preprocessing before tokenization."""
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

        Raises:
            ValueError: Padding methods not recognized.
            ValueError: `max_length` is None when the padding method is
                "max_length".

        Returns:
            Padded sequences.
        """
        if padding == "max_length":
            if max_length is None:
                raise ValueError(
                    "max_length cannot be None when the padding method is "
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

    def train(self, sequences: Union[Sequence[str]]) -> "Tokenizer":
        """Train the tokenizer.

        Args:
            sequences: The batch of input sequences.

        Returns:
            A `Tokenizer` object.
        """
        n = self.capacity - len(self) if self.capacity else None
        if n is not None and n <= 0:
            raise RuntimeError("The tokenizer capacity is already full.")

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
