import uuid
from typing import List, Union

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A cited span of a source file, in character offsets."""

    file_path: str
    first_character_index: int
    last_character_index: int


class Chunk(MinimalSource):
    """A piece of a source file with its text content."""

    text: str


class ChunkList(BaseModel):
    """All chunks of the corpus, as saved in data/processed/chunks.json."""

    chunks: List[Chunk]


class UnansweredQuestion(BaseModel):
    """A question without ground truth attached."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions, answered or not."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Retrieved sources for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieved sources plus the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Output of a search run over a whole dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of an answer run: search results with answers."""

    search_results: List[MinimalAnswer]
    k: int
