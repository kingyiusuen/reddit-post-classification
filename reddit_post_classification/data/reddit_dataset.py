from pathlib import Path
from typing import Union

import pandas as pd
from torch.utils.data import Dataset


class RedditDataset(Dataset):
    """A Dataset object.

    Args:
        fname: Path to the data file.
    """

    def __init__(self, fname: Union[str, Path]) -> None:
        self.data = pd.read_csv(fname)

    def __getitem__(self, idx: int):
        """Returns a sample from the dataset."""
        return self.data.iloc[idx]["text"], self.data.iloc[idx]["label"]

    def __len__(self) -> int:
        """Returns the number of samples."""
        return len(self.data)
