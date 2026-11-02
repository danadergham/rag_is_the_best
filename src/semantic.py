"""Semantic index (bonus): MiniLM embeddings stored in FAISS.

Bonus features living here:
* B1 - sentence embeddings (all-MiniLM-L6-v2, CPU) in a FAISS vector store;
* B3 - incremental indexing with LangChain's index() + SQLRecordManager:
  only chunks that are new or changed get embedded;
* B4 - CacheBackedEmbeddings: vectors (and query vectors) are cached on disk.
"""

import shutil
from pathlib import Path
from typing import Dict, Optional

import faiss
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.indexes import SQLRecordManager
from langchain_classic.storage import LocalFileStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.indexing import index
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

from src.chunking import chunk_to_document
from src.common import (
    EMBEDDING_CACHE_DIR,
    EMBEDDING_MODEL,
    FAISS_DIR,
    PROCESSED_DIR,
    RECORD_DB_FILE,
    RagError,
)
from src.indexing import load_chunks

EMBEDDING_BATCH_SIZE = 256


def get_embeddings(
    processed_dir: str = PROCESSED_DIR,
) -> CacheBackedEmbeddings:
    """Create the embedding model wrapped in an on-disk cache.

    Args:
        processed_dir: Directory where the cache folder lives.

    Returns:
        An embeddings object that only computes vectors it has not seen.
    """
    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = LocalFileStore(str(Path(processed_dir) / EMBEDDING_CACHE_DIR))
    return CacheBackedEmbeddings.from_bytes_store(
        model,
        store,
        namespace=EMBEDDING_MODEL.split("/")[-1],
        query_embedding_cache=True,
        key_encoder="sha256",
    )


def new_vector_store(embeddings: CacheBackedEmbeddings) -> FAISS:
    """Create an empty FAISS store using cosine similarity.

    Args:
        embeddings: The embedding model (vectors are L2-normalised, so the
            inner product equals the cosine similarity).

    Returns:
        An empty vector store.
    """
    size = len(embeddings.embed_query("dimension probe"))
    return FAISS(
        embedding_function=embeddings,
        index=faiss.IndexFlatIP(size),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def load_vector_store(
    processed_dir: str = PROCESSED_DIR,
    embeddings: Optional[CacheBackedEmbeddings] = None,
) -> FAISS:
    """Load the FAISS store saved by index_semantic.

    The folder was written by us, so unpickling its docstore is safe.

    Args:
        processed_dir: Directory holding the faiss_index folder.
        embeddings: Embedding model to attach; loaded when not given.

    Returns:
        The vector store, ready to be queried.

    Raises:
        RagError: If the semantic index has not been built.
    """
    folder = Path(processed_dir) / FAISS_DIR
    if not folder.is_dir():
        raise RagError("Semantic index not found. "
                       "Run the `index_semantic` command first.")
    return FAISS.load_local(
        str(folder),
        embeddings or get_embeddings(processed_dir),
        allow_dangerous_deserialization=True,
    )


def index_semantic(
    processed_dir: str = PROCESSED_DIR, rebuild: bool = False
) -> Dict[str, object]:
    """Embed the chunks and store them in FAISS, incrementally.

    LangChain's index() hashes every chunk and asks the record manager
    which ones it already stored: unchanged chunks are skipped (not
    embedded again), changed ones are replaced, vanished ones deleted.

    Args:
        processed_dir: Directory holding chunks.json; the vector store,
            record database and embedding cache are written next to it.
        rebuild: Throw the existing semantic index away and start over.

    Returns:
        LangChain's counters: num_added, num_updated, num_skipped and
        num_deleted.

    Raises:
        RagError: If the lexical index is missing.
    """
    chunks = load_chunks(processed_dir)
    folder = Path(processed_dir) / FAISS_DIR
    database = Path(processed_dir) / RECORD_DB_FILE
    if rebuild or not folder.is_dir():
        # A record database without its vectors would wrongly skip chunks.
        shutil.rmtree(folder, ignore_errors=True)
        database.unlink(missing_ok=True)
    embeddings = get_embeddings(processed_dir)
    store = (load_vector_store(processed_dir, embeddings) if folder.is_dir()
             else new_vector_store(embeddings))
    manager = SQLRecordManager(
        "faiss/chunks", db_url=f"sqlite:///{database}"
    )
    manager.create_schema()
    documents = (
        chunk_to_document(chunk, chunk.text)
        for chunk in tqdm(chunks, desc="Embedding")
    )
    result = index(
        documents,
        manager,
        store,
        cleanup="full",
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    store.save_local(str(folder))
    return dict(result)
