"""Code-aware tokenizer used by BM25 for both documents and queries."""

import re
from typing import List

STOP_WORDS = frozenset(
    "the a an of to in is are was be and or for on with as by it this that "
    "what which how does do can from at into its their when where why vllm"
    .split()
)
# A "word" is an identifier-like run: letters, digits and underscores.
WORD = re.compile(r"[A-Za-z0-9_]+")
# Pieces of an identifier: HTTPServer -> HTTP, Server; max_len -> max, len.
PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def tokenize(text: str) -> List[str]:
    """Split text into lowercase tokens, breaking identifiers into parts.

    The full identifier is kept next to its parts, so a question that quotes
    ``fused_batched_moe`` and one that says "batched moe" both match.

    Args:
        text: Source code, documentation or a natural-language question.

    Returns:
        The tokens, stop words removed.
    """
    tokens: List[str] = []
    for word in WORD.findall(text):
        parts = [part.lower() for part in PART.findall(word)]
        if len(parts) > 1:
            tokens.append(word.lower())
        tokens.extend(parts)
    return [token for token in tokens if token not in STOP_WORDS]
