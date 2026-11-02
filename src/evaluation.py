"""Recall@k of search results against a ground-truth dataset."""

from typing import Dict, List, Sequence

from src.common import RagError, load_model
from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

IOU_THRESHOLD = 0.05
RECALL_KS = (1, 3, 5, 10)


def iou(first: MinimalSource, second: MinimalSource) -> float:
    """Intersection over union of two character ranges in the same file.

    Args:
        first: One source span.
        second: The other source span.

    Returns:
        0.0 when the files differ or the ranges do not overlap.
    """
    if first.file_path != second.file_path:
        return 0.0
    start = max(first.first_character_index, second.first_character_index)
    end = min(first.last_character_index, second.last_character_index)
    overlap = end - start
    if overlap <= 0:
        return 0.0
    union = ((first.last_character_index - first.first_character_index)
             + (second.last_character_index - second.first_character_index)
             - overlap)
    return overlap / union


def question_recall(
    retrieved: Sequence[MinimalSource],
    expected: Sequence[MinimalSource],
    k: int,
) -> float:
    """Share of the expected sources found in the top-k retrieved ones.

    Args:
        retrieved: Retrieved sources, best first.
        expected: Ground-truth sources.
        k: How many retrieved sources to look at.

    Returns:
        A value between 0 and 1.
    """
    if not expected:
        return 0.0
    top = retrieved[:k]
    found = sum(
        any(iou(source, truth) > IOU_THRESHOLD for source in top)
        for truth in expected
    )
    return found / len(expected)


def compute_recall(
    results: StudentSearchResults,
    dataset: RagDataset,
    ks: Sequence[int] = RECALL_KS,
) -> Dict[int, float]:
    """Average recall@k over all answered questions of a dataset.

    Args:
        results: Search results produced by search_dataset.
        dataset: Ground-truth dataset (with sources).
        ks: The values of k to report.

    Returns:
        A mapping k -> mean recall. Questions missing from the results
        count as 0.
    """
    by_id = {r.question_id: r for r in results.search_results}
    answered: List[AnsweredQuestion] = [
        q for q in dataset.rag_questions if isinstance(q, AnsweredQuestion)
    ]
    if not answered:
        raise RagError("The dataset has no ground-truth sources to score.")
    recalls: Dict[int, float] = {}
    for k in ks:
        total = 0.0
        for question in answered:
            found = by_id.get(question.question_id)
            if found is not None:
                total += question_recall(
                    found.retrieved_sources, question.sources, k
                )
        recalls[k] = total / len(answered)
    return recalls


def evaluate(student_search_results_path: str, dataset_path: str) -> str:
    """Load both files and format the recall report.

    Args:
        student_search_results_path: JSON written by search_dataset.
        dataset_path: Ground-truth AnsweredQuestions JSON.

    Returns:
        A printable report, one Recall@k per line.
    """
    results = load_model(student_search_results_path, StudentSearchResults)
    dataset = load_model(dataset_path, RagDataset)
    recalls = compute_recall(results, dataset)
    answered = sum(
        isinstance(q, AnsweredQuestion) for q in dataset.rag_questions
    )
    lines = ["Evaluation Results", "=" * 40,
             f"Questions evaluated: {answered}"]
    for k, value in recalls.items():
        lines.append(f"Recall@{k}: {value:.3f} ({value * 100:.1f}%)")
    return "\n".join(lines)
