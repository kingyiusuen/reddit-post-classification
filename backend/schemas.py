from typing import Optional

from pydantic import BaseModel, validator


class Post(BaseModel):
    """Schema for user inputs from the frontend."""

    title: str
    selftext: Optional[str]

    @validator("title")
    def title_must_not_be_empty(cls, value):
        """Validate the title."""
        if not len(value):
            raise ValueError("Title cannot be empty.")
        return value
