from pydantic import BaseModel


class Post(BaseModel):
    title: str
    selftext: str
