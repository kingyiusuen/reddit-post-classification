from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import praw

from reddit_post_classification.utils.python_logger import get_logger


log = get_logger(__name__)


class Scraper:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        subreddit_names: Union[str, List[str]],
        sort: str = "new",
        limit: int = 1000,
        selftext_only: bool = True,
        fname: str = "data/raw/reddit_posts.csv",
        since: Optional[int] = None,
    ):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        if isinstance(subreddit_names, str):
            subreddit_names = [subreddit_names]
        self.subreddit_names = subreddit_names
        self.sort = sort
        self.limit = limit
        self.selftext_only = selftext_only
        self.fname = Path(fname)
        self.fname.parent.mkdir(parents=True, exist_ok=True)
        self.since = since

    def scrape(self):
        log.info("Scraping begins")
        data = []
        for subreddit_name in self.subreddit_names:
            data += self._get_posts(subreddit_name)
        headers = ["id", "created_utc", "title", "selftext", "subreddit_name"]
        df = pd.DataFrame(data, columns=headers)
        df.to_csv(self.fname, index=False)

    def _get_posts(self, subreddit_name):
        posts = []
        subreddit = self.reddit.subreddit(subreddit_name)
        for post in getattr(subreddit, self.sort)(limit=self.limit):
            if self.since and post.created_utc <= self.since:
                continue
            if self.selftext_only and not post.is_self:
                continue
            if not post.title:
                continue
            entry = {
                "id": post.id,
                "created_utc": post.created_utc,
                "title": post.title,
                "selftext": post.selftext,
                "subreddit_name": subreddit_name,
            }
            posts.append(entry)
        log.info(f"{len(posts)} posts scraped from r/{subreddit_name}")
        return posts
