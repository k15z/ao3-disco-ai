from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, field_validator


class Tags(BaseModel):
    fandom: list[str]
    category: list[str]
    rating: list[str]
    character: list[str]
    relationship: list[str]
    freeform: list[str]


class Statistics(BaseModel):
    words: int
    hits: int
    kudos: int = 0
    comments: int = 0
    bookmarks: int = 0
    chapters: str
    language: str
    published: date
    status: Optional[date] = None

    @field_validator("words", "hits", "kudos", "comments", "bookmarks", mode="before")
    @classmethod
    def parse_int(cls, v):
        if v == "":
            return 0
        if isinstance(v, str):
            return int(v.replace(",", ""))
        return v


class Work(BaseModel):
    work_id: int

    title: str
    authors: list[str]
    summary: str

    tags: Tags
    statistics: Statistics


DenseMeta = List[str]
SparseMeta = Dict[str, int]
