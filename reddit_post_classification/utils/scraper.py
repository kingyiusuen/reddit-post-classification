from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import praw

from reddit_post_classification.utils.python_logger import get_logger


log = get_logger(__name__)


class Scraper:
    """Scrape posts from subreddits and save them in a csv file.

    Args:
        client_id: The OAuth client id associated with your registered Reddit
            application.
        client_secret: The OAuth client secret associated with your registered
            Reddit application.
        user_agent: A unique description of your application.
        subreddit_names: Names of the subreddits to be scraped. Can be either
            a string or a list of strings.
        sort: How to iterate through the posts (e.g., 'hot', 'top',
            'controversial'). See the documentation of PRAW for all possible
            choices,
        limit: Maximum number of posts to scrape per subreddit. Actual number
            may be smaller than this, depending on the `sort` method and
            whether `selftext_only` and `since` are used.
        selftext_only: Scrape posts with selftext only.
        fname: Path of the output file.
        since: Scrape posts with a UTC time greater than (more recent than)
            this value.
    """

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
    ) -> None:
        if any(x is None for x in [client_id, client_secret, user_agent]):
            raise ValueError(
                "client_id, client_secret and user_agent are required."
            )
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

    def scrape(self) -> None:
        """Main method to scrape posts from subreddits."""
        log.info("Scraping begins")
        data = []
        for subreddit_name in self.subreddit_names:
            data += self._get_posts(subreddit_name)
        headers = ["id", "created_utc", "title", "selftext", "subreddit_name"]
        df = pd.DataFrame(data, columns=headers)
        df.to_csv(self.fname, index=False)

    def _get_posts(self, subreddit_name: str) -> List[Dict[str, Any]]:
        """Scrape posts from a subreddit.

        Args:
            subreddit_name: Name of the subreddit to be scraped.
        """
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
