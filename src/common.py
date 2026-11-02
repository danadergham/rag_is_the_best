"""Shared constants, the project error type and small JSON helpers."""

from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
DEFAULT_MAX_CHUNK_SIZE = 2000
DEFAULT_K = 10

CHUNKS_FILE = "chunks.json"
BM25_FILE = "bm25.pkl"
FAISS_DIR = "faiss_index"
RECORD_DB_FILE = "record_manager.sqlite"
EMBEDDING_CACHE_DIR = "embedding_cache"
LLM_CACHE_FILE = "llm_cache.sqlite"

ANSWER_MODEL = "Qwen/Qwen3-0.6B"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ModelT = TypeVar("ModelT", bound=BaseModel)


class RagError(Exception):
    """An expected problem (bad input, missing index...), clearly worded."""


def load_model(path: str, model_class: Type[ModelT]) -> ModelT:
    """Read a JSON file and validate it with a pydantic model.

    Args:
        path: Path of the JSON file.
        model_class: Pydantic model the content must match.

    Returns:
        The validated model instance.

    Raises:
        RagError: If the file is missing, unreadable or does not match.
    """
    try:
        with open(path, encoding="utf-8") as file:
            return model_class.model_validate_json(file.read())
    except FileNotFoundError:
        raise RagError(f"File not found: {path}")
    except (OSError, UnicodeDecodeError) as error:
        raise RagError(f"Cannot read {path}: {error}")
    except ValidationError as error:
        raise RagError(f"{path} is not a valid {model_class.__name__}: "
                       f"{error.error_count()} validation error(s)")


def save_model(model: BaseModel, directory: str, file_name: str) -> Path:
    """Write a pydantic model as indented JSON, creating the directory.

    Args:
        model: The model to serialise.
        directory: Output directory (created when missing).
        file_name: Name of the JSON file inside the directory.

    Returns:
        The path of the written file.
    """
    target = Path(directory) / file_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    return target
