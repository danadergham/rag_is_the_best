"""Split source files into chunks with exact character offsets."""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.models import Chunk

CHUNK_OVERLAP = 200
LANGUAGES = {".py": Language.PYTHON, ".md": Language.MARKDOWN}


def split_documents(
    documents: List[Document], max_chunk_size: int
) -> List[Chunk]:
    """Split loaded files into chunks that point back into them.

    Python and Markdown files are cut on code or heading boundaries; other
    files on paragraphs, lines, then words.

    Args:
        documents: Loaded files; ``metadata["source"]`` is the file path
            stored in each chunk, so it must match the corpus exactly.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        Chunks such that file_text[first:last] == chunk.text and
        len(chunk.text) <= max_chunk_size.
    """
    # The overlap must stay below the chunk size, or splitting can't advance.
    overlap = min(CHUNK_OVERLAP, max_chunk_size // 2)
    plain = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
    )
    splitters = {
        extension: RecursiveCharacterTextSplitter.from_language(
            language,
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
        )
        for extension, language in LANGUAGES.items()
    }
    chunks: List[Chunk] = []
    for document in tqdm(documents, desc="Chunking"):
        path = document.metadata["source"]
        splitter = splitters.get(Path(path).suffix.lower(), plain)
        for piece in splitter.split_documents([document]):
            start = piece.metadata["start_index"]
            chunks.append(
                Chunk(
                    file_path=path,
                    first_character_index=start,
                    last_character_index=start + len(piece.page_content),
                    text=piece.page_content,
                )
            )
    return chunks


def chunk_to_document(chunk: Chunk, content: str) -> Document:
    """Turn a chunk into a LangChain document that remembers its location.

    Args:
        chunk: The chunk to convert.
        content: The text the retriever should index for this chunk.

    Returns:
        A document whose metadata holds the path and the character range.
    """
    first = chunk.first_character_index
    return Document(
        page_content=content,
        metadata={
            "file_path": chunk.file_path,
            "first_character_index": first,
            "last_character_index": chunk.last_character_index,
            "chunk_id": f"{chunk.file_path}:{first}",
        },
    )
