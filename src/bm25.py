"""The lexical (BM25) retriever: building it and loading it back."""

import pickle
from pathlib import Path
from typing import List, cast

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from src.chunking import chunk_to_document
from src.common import BM25_FILE, PROCESSED_DIR, RagError
from src.models import Chunk
from src.tokenizer import tokenize


class GuardedBM25Retriever(BM25Retriever):
    """BM25Retriever that returns nothing when no query word is indexed.

    Plain BM25 ranks a gibberish query by all-zero scores, which gives an
    arbitrary list. Returning no result is more honest.
    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Return the top-k documents, or [] when the query matches nothing."""
        words = self.preprocess_func(query)
        if not any(word in self.vectorizer.idf for word in words):
            return []
        return super()._get_relevant_documents(query, run_manager=run_manager)


def search_text(chunk: Chunk) -> str:
    """Text that BM25 indexes: the file path first, then the content.

    Questions often name the file or module they are about, so putting the
    path in front lets those words match.

    Args:
        chunk: The chunk to index.

    Returns:
        The path, a newline, and the chunk text.
    """
    return f"{chunk.file_path}\n{chunk.text}"


def build_bm25(chunks: List[Chunk]) -> GuardedBM25Retriever:
    """Build a LangChain BM25 retriever over the chunks.

    Args:
        chunks: All chunks of the corpus.

    Returns:
        A retriever whose documents carry the chunk location as metadata.
    """
    documents = [
        chunk_to_document(chunk, search_text(chunk)) for chunk in chunks
    ]
    # from_documents builds `cls(...)`, but is typed to return the base class.
    return cast(GuardedBM25Retriever, GuardedBM25Retriever.from_documents(
        documents, preprocess_func=tokenize
    ))


def load_bm25(processed_dir: str = PROCESSED_DIR) -> GuardedBM25Retriever:
    """Load the BM25 retriever saved by the index command.

    Args:
        processed_dir: Directory holding bm25.pkl.

    Returns:
        The ready-to-query retriever.

    Raises:
        RagError: If the index has not been built yet.
    """
    path = Path(processed_dir) / BM25_FILE
    try:
        with open(path, "rb") as file:
            retriever: GuardedBM25Retriever = pickle.load(file)
    except FileNotFoundError:
        raise RagError(f"Index not found in {processed_dir}. "
                       "Run the `index` command first.")
    return retriever
