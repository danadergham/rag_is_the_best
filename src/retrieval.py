"""Search the index: one query, or a whole dataset of questions."""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from tqdm import tqdm

from src.bm25 import GuardedBM25Retriever, load_bm25
from src.common import PROCESSED_DIR, RagError, load_model, save_model
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

METHODS = ("lexical", "semantic", "hybrid")
# The hybrid method fuses longer candidate lists than the k that is returned.
HYBRID_CANDIDATES = 3
# BM25 is the stronger signal on this corpus (see README), so it weighs more.
HYBRID_WEIGHTS = [0.7, 0.3]


class Searcher:
    """Finds the sources most relevant to a question.

    Three methods share the same interface:
    * lexical: BM25 only (the default, needs just the `index` command);
    * semantic: FAISS embeddings only (needs `index_semantic` too);
    * hybrid: both rankings merged with reciprocal-rank fusion.
    """

    def __init__(
        self, method: str = "lexical", processed_dir: str = PROCESSED_DIR
    ) -> None:
        """Load the index needed by the method.

        Args:
            method: One of "lexical", "semantic" or "hybrid".
            processed_dir: Directory written by the index commands.

        Raises:
            RagError: If the method is unknown or its index is missing.
        """
        if method not in METHODS:
            raise RagError(f"Unknown method '{method}'. "
                           f"Choose one of: {', '.join(METHODS)}.")
        self.bm25: Optional[GuardedBM25Retriever] = (
            load_bm25(processed_dir) if method != "semantic" else None
        )
        self.vectors: Optional[VectorStoreRetriever] = None
        if method != "lexical":
            # Imported here: loading torch is slow and lexical doesn't need it.
            from src.semantic import load_vector_store
            self.vectors = load_vector_store(processed_dir).as_retriever()
        parts: List[Runnable[str, List[Document]]] = [
            part for part in (self.bm25, self.vectors) if part is not None
        ]
        if len(parts) == 1:
            self.retriever = parts[0]
            self.candidates = 1
        else:
            from langchain_classic.retrievers import EnsembleRetriever
            self.retriever = EnsembleRetriever(
                retrievers=parts, weights=HYBRID_WEIGHTS, id_key="chunk_id"
            )
            self.candidates = HYBRID_CANDIDATES

    def _set_k(self, k: int) -> None:
        """Tell each underlying retriever how many documents to fetch."""
        if self.bm25 is not None:
            self.bm25.k = k * self.candidates
        if self.vectors is not None:
            self.vectors.search_kwargs["k"] = k * self.candidates

    def retrieve(self, query: str, k: int) -> List[Document]:
        """Return the k best chunks for a query, best first.

        Args:
            query: The question or keywords.
            k: Number of chunks wanted (0 gives an empty list).

        Returns:
            Documents whose metadata holds the file path and offsets.
        """
        if k == 0 or not query.strip():
            return []
        self._set_k(k)
        return self.retriever.invoke(query)[:k]

    def search(self, query: str, k: int) -> List[MinimalSource]:
        """Return the k best source locations for a query.

        Args:
            query: The question or keywords.
            k: Number of sources wanted.

        Returns:
            Sources (path and character range), best first.
        """
        return [
            MinimalSource(
                file_path=doc.metadata["file_path"],
                first_character_index=doc.metadata["first_character_index"],
                last_character_index=doc.metadata["last_character_index"],
            )
            for doc in self.retrieve(query, k)
        ]


def format_source(source: MinimalSource) -> str:
    """Show a source like the subject does: ``path [first:last]``."""
    return (f"{source.file_path} "
            f"[{source.first_character_index}:{source.last_character_index}]")


def search_dataset(
    dataset_path: str,
    k: int,
    save_directory: str,
    method: str = "lexical",
    processed_dir: str = PROCESSED_DIR,
) -> Path:
    """Search every question of a dataset and save the results.

    Args:
        dataset_path: JSON dataset (answered or unanswered questions).
        k: Number of sources per question.
        save_directory: Directory where the results file is written; it
            keeps the dataset's file name.
        method: "lexical", "semantic" or "hybrid".
        processed_dir: Directory written by the index commands.

    Returns:
        The path of the written StudentSearchResults JSON.

    Raises:
        RagError: For a missing/invalid dataset or a missing index.
    """
    dataset = load_model(dataset_path, RagDataset)
    searcher = Searcher(method, processed_dir)
    results = [
        MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=searcher.search(question.question, k),
        )
        for question in tqdm(dataset.rag_questions, desc="Searching")
    ]
    output = StudentSearchResults(search_results=results, k=k)
    return save_model(output, save_directory, Path(dataset_path).name)
