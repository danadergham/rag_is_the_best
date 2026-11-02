"""Command-line interface: ``uv run python -m src <command> [options]``."""

import functools
import sys
from typing import Any, Callable

import fire

from src.common import (
    DEFAULT_K,
    DEFAULT_MAX_CHUNK_SIZE,
    PROCESSED_DIR,
    RAW_DIR,
    RagError,
)

SEARCH_RESULTS_DIR = "data/output/search_results"
ANSWERS_DIR = "data/output/search_results_and_answer"


def guarded(command: Callable[..., None]) -> Callable[..., None]:
    """Turn any failure of a command into a clean error message.

    Args:
        command: A CLI command.

    Returns:
        The same command, which never lets an exception escape.
    """
    @functools.wraps(command)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            command(*args, **kwargs)
        except KeyboardInterrupt:
            sys.exit("Interrupted.")
        except RagError as error:
            sys.exit(f"Error: {error}")
        except Exception as error:
            sys.exit(f"Error: {type(error).__name__}: {error}")

    return wrapper


def as_int(value: Any, name: str, minimum: int | None = None) -> int:
    """Check that a command-line value is an integer.

    Args:
        value: What Fire parsed from the command line.
        name: Option name, used in the error message.
        minimum: Smallest accepted value, if any.

    Returns:
        The value as an int.

    Raises:
        RagError: If the value is not an integer or is below the minimum.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise RagError(f"{name} must be an integer, got {value!r}.")
    if minimum is not None and value < minimum:
        raise RagError(f"{name} must be at least {minimum}, got {value}.")
    return value


class Cli:
    """RAG against the machine: ask questions about the vLLM codebase."""

    @guarded
    def index(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        raw_dir: str = RAW_DIR,
        out_dir: str = PROCESSED_DIR,
    ) -> None:
        """Chunk the corpus and build the BM25 index.

        Args:
            max_chunk_size: Maximum characters per chunk (at most 2000).
            raw_dir: Corpus directory.
            out_dir: Where the index is written.
        """
        from src.indexing import build_index

        size = as_int(max_chunk_size, "--max_chunk_size")
        count = build_index(raw_dir, out_dir, size)
        print(f"Ingestion complete! Indexed {count} chunks under {out_dir}/")

    @guarded
    def index_semantic(
        self, rebuild: bool = False, processed_dir: str = PROCESSED_DIR
    ) -> None:
        """Build or update the FAISS embedding index (incremental).

        Args:
            rebuild: Start from scratch instead of updating.
            processed_dir: Directory written by the `index` command.
        """
        from src.semantic import index_semantic

        counts = index_semantic(processed_dir, bool(rebuild))
        print(f"Semantic index updated: {counts}")

    @guarded
    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        method: str = "lexical",
        processed_dir: str = PROCESSED_DIR,
    ) -> None:
        """Print the top-k sources for one query.

        Args:
            query: The question.
            k: Number of sources.
            method: lexical, semantic or hybrid.
            processed_dir: Directory written by the index commands.
        """
        from src.retrieval import Searcher, format_source

        query = str(query)
        sources = Searcher(method, processed_dir).search(
            query, as_int(k, "-k", 0)
        )
        if not sources:
            print("No relevant source found.")
        for source in sources:
            print(format_source(source))

    @guarded
    def search_dataset(
        self,
        dataset_path: str,
        k: int = DEFAULT_K,
        save_directory: str = SEARCH_RESULTS_DIR,
        method: str = "lexical",
        processed_dir: str = PROCESSED_DIR,
    ) -> None:
        """Search every question of a dataset and save the results.

        Args:
            dataset_path: JSON dataset of questions.
            k: Number of sources per question.
            save_directory: Where the results JSON is written.
            method: lexical, semantic or hybrid.
            processed_dir: Directory written by the index commands.
        """
        from src.retrieval import search_dataset

        path = search_dataset(
            dataset_path, as_int(k, "-k", 0), save_directory, method,
            processed_dir,
        )
        print(f"Saved student_search_results to {path}")

    @guarded
    def answer(
        self,
        query: str,
        k: int = DEFAULT_K,
        method: str = "lexical",
        processed_dir: str = PROCESSED_DIR,
    ) -> None:
        """Retrieve sources for one query and answer it.

        Args:
            query: The question.
            k: Number of sources given to the model.
            method: lexical, semantic or hybrid.
            processed_dir: Directory written by the index commands.
        """
        from src.generation import NO_CONTEXT_ANSWER, Answerer
        from src.retrieval import Searcher, format_source

        query = str(query)
        sources = Searcher(method, processed_dir).search(
            query, as_int(k, "-k", 0)
        )
        if sources:
            text = Answerer(processed_dir=processed_dir).answer(query, sources)
        else:
            text = NO_CONTEXT_ANSWER
        print(text)
        if sources:
            unanswered = text.lower().startswith("i don't know")
            print("\nClosest sources (not relevant enough to answer):"
                  if unanswered else "\nSources:")
            for source in sources:
                print(f"  {format_source(source)}")

    @guarded
    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = ANSWERS_DIR,
        processed_dir: str = PROCESSED_DIR,
    ) -> None:
        """Answer every question of a search-results file.

        Args:
            student_search_results_path: JSON written by search_dataset.
            save_directory: Where the answers JSON is written.
            processed_dir: Directory holding the LLM cache.
        """
        from src.generation import answer_dataset

        path = answer_dataset(
            student_search_results_path, save_directory,
            processed_dir=processed_dir,
        )
        print(f"Saved student_search_results_and_answer to {path}")

    @guarded
    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Print recall@k of search results against a ground-truth dataset.

        Args:
            student_search_results_path: JSON written by search_dataset.
            dataset_path: Ground-truth AnsweredQuestions JSON.
        """
        from src.evaluation import evaluate

        print(evaluate(student_search_results_path, dataset_path))

    @guarded
    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        processed_dir: str = PROCESSED_DIR,
    ) -> None:
        """Start the local HTTP API (see /docs for the interactive page).

        Args:
            host: Address to listen on.
            port: Port to listen on.
            processed_dir: Directory written by the index commands.
        """
        import uvicorn

        from src.api import create_app

        uvicorn.run(
            create_app(processed_dir), host=host, port=as_int(port, "--port")
        )


if __name__ == "__main__":
    fire.Fire(Cli)
