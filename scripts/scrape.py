import hydra
from omegaconf import DictConfig

from reddit_post_classification.scrape import Scraper


def main(config: DictConfig) -> None:
    scraper: Scraper = hydra.utils.instantiate(config.scraper)
    scraper.scrape()


@hydra.main(config_path="../configs", config_name="config")
def hydra_entry(config: DictConfig) -> None:
    main(config)


if __name__ == "__main__":
    hydra_entry()
