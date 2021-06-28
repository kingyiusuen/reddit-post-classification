from pathlib import Path
from typing import Dict, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from reddit_post_classification.data.reddit_dataset import RedditDataset
from reddit_post_classification.data.tokenizer import Tokenizer
from reddit_post_classification.utils import get_logger


log = get_logger(__name__)


class RedditDataModule(LightningDataModule):
    """A class that handles data processing.

    Args:
        data_dir: Directory to the data.
        labels: A list of subreddit names.
        batch_size: The number samples to load per batch.
        num_workers: The number of subprocesses to use for data loading.
        pin_memory: Whether to copy Tensors into CUDA pinned memory before
            returning them.
        max_seq_len: Maximum number of tokens per sequence.
    """

    def __init__(
        self,
        data_dir: str,
        labels: List[str],
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_seq_len: int = 512,
    ) -> None:
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
