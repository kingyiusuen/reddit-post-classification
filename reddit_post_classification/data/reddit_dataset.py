import pandas as pd
from torch.utils.data import Dataset


class RedditDataset(Dataset):
    def __init__(self, fname):
        self.data = pd.read_csv(fname)

    def __getitem__(self, idx: int):
        return self.data.iloc[idx]["text"], self.data.iloc[idx]["label"]

    def __len__(self) -> int:
        return len(self.data)
