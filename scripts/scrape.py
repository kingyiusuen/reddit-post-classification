import hydra
from omegaconf import DictConfig

from reddit_post_classification.scraper import Scraper


def main(cfg: DictConfig) -> None:
    scraper: Scraper = hydra.utils.instantiate(cfg.scraper)
    scraper.scrape()


@hydra.main(config_path="../configs", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
