"""Local HTTP API (bonus): search and answer over HTTP with FastAPI."""

import uuid

from fastapi import FastAPI, Query

from src.common import DEFAULT_K, PROCESSED_DIR
from src.generation import Answerer
from src.models import MinimalAnswer, MinimalSearchResults
from src.retrieval import Searcher


def create_app(processed_dir: str = PROCESSED_DIR) -> FastAPI:
    """Build the FastAPI application.

    The index and the model are loaded once, when the server starts.

    Args:
        processed_dir: Directory written by the index commands.

    Returns:
        The application, ready for uvicorn.
    """
    app = FastAPI(title="RAG against the machine")
    searcher = Searcher(processed_dir=processed_dir)
    answerer = Answerer(processed_dir=processed_dir)

    @app.post("/search", response_model=MinimalSearchResults)
    async def search(
        query: str, k: int = Query(DEFAULT_K, ge=0, le=100)
    ) -> MinimalSearchResults:
        """Return the top-k sources for a query."""
        return MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=searcher.search(query, k),
        )

    @app.post("/answer", response_model=MinimalAnswer)
    async def answer(
        query: str, k: int = Query(DEFAULT_K, ge=0, le=100)
    ) -> MinimalAnswer:
        """Retrieve sources for a query and generate an answer."""
        result = await search(query, k)
        text = answerer.answer(query, result.retrieved_sources)
        return MinimalAnswer(**result.model_dump(), answer=text)

    return app
