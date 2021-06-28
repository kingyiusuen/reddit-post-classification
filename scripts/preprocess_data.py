from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from reddit_post_classification.data import Tokenizer
from reddit_post_classification.utils import get_logger


log = get_logger(__name__)


def main(cfg: DictConfig) -> None:
    data_dir = Path(cfg.data_dir)
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        raw_data_dir / "reddit_posts.csv",
        usecols=["title", "selftext", "subreddit_name"],
    )

    # Convert labels to ids
    df["label"] = df["subreddit_name"].map(
        {label: i for i, label in enumerate(cfg.subreddit_names)}
    )

    # Clean text
    df.fillna("", inplace=True)
    tokenizer = Tokenizer(do_lowercase=True)
    df["text"] = df.apply(tokenizer.pretokenize, axis=1)
    df = df[~(df["text"] == "")]
    df.drop(columns=["title", "selftext", "subreddit_name"], inplace=True)

    # Split datasets
    tmp_df, val_df = train_test_split(
        df, test_size=cfg.val_size, stratify=df["label"]
    )
    train_df, test_df = train_test_split(
        tmp_df, test_size=cfg.test_size, stratify=tmp_df["label"]
    )

    # Train a tokenizer using the text in the training set
    # Save the mapping from tokens to ids
    tokenizer.train(train_df["text"])
    tokenizer.save(processed_data_dir / "tokenizer.json")

    # Save dataframes to csv files
    train_df.to_csv(processed_data_dir / "train.csv", index=False)
    val_df.to_csv(processed_data_dir / "val.csv", index=False)
    test_df.to_csv(processed_data_dir / "test.csv", index=False)
    log.info(f"Processed datasets saved to {str(processed_data_dir)}.")


@hydra.main(config_path="../configs", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
