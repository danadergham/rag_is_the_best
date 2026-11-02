"""Answer generation: retrieved snippets + a question -> Qwen3-0.6B."""

import re
from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from tqdm import tqdm

from src.common import (
    ANSWER_MODEL,
    LLM_CACHE_FILE,
    PROCESSED_DIR,
    load_model,
    save_model,
)
from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

MAX_NEW_TOKENS = 200
# Tokens of retrieved text given to the model (it accepts far more, but a
# small CPU model gets slow and vague with a very long context).
MAX_CONTEXT_TOKENS = 2000
NO_CONTEXT_ANSWER = (
    "No source was retrieved, so there is no context to answer from."
)

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You answer questions about the vLLM codebase using ONLY the context "
        "provided. First check that the context is about the same thing the "
        "question asks. Answer only if it is. If the question is unrelated "
        "to the context, or the context does not contain the answer, reply "
        "exactly: I don't know. Never guess or use outside knowledge. When "
        "you do answer, be concise and precise, and mention the relevant "
        "files, classes, functions or options.",
    ),
    # /no_think switches Qwen3's "thinking" off, which keeps answers short
    # on a CPU.
    ("human", "Context:\n{context}\n\nQuestion: {question} /no_think"),
])


@lru_cache(maxsize=64)
def read_file(file_path: str) -> str:
    """Read a source file the same way the chunker did (cached).

    Args:
        file_path: Path stored in a MinimalSource.

    Returns:
        The file content.
    """
    with open(file_path, encoding="utf-8", errors="replace") as file:
        return file.read()


def read_snippet(source: MinimalSource) -> str:
    """Return the text a source points at.

    Args:
        source: File path and character range.

    Returns:
        The text between the two offsets.
    """
    text = read_file(source.file_path)
    return text[source.first_character_index:source.last_character_index]


def clean_answer(text: str) -> str:
    """Remove the leftover thinking block from the output."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


class Answerer:
    """Generates grounded answers with a local Hugging Face model."""

    def __init__(
        self,
        model_id: str = ANSWER_MODEL,
        processed_dir: str = PROCESSED_DIR,
    ) -> None:
        """Load the model and switch on the LLM cache.

        Args:
            model_id: Hugging Face model, Qwen/Qwen3-0.6B by default.
            processed_dir: Where the LLM cache database is stored (bonus B4:
                asking the same question twice never runs the model twice).
        """
        cache_path = Path(processed_dir) / LLM_CACHE_FILE
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=str(cache_path)))
        llm = HuggingFacePipeline.from_model_id(
            model_id,
            task="text-generation",
            device=-1,
            pipeline_kwargs={
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
                "return_full_text": False,
            },
        )
        # ChatHuggingFace applies the model's own chat template.
        chat = ChatHuggingFace(llm=llm)
        self.tokenizer = chat.tokenizer
        self.chain = (
            PROMPT | chat | StrOutputParser() | RunnableLambda(clean_answer)
        )

    def build_context(self, sources: List[MinimalSource]) -> str:
        """Join the sources' text, best first, within the token budget.

        Args:
            sources: Retrieved sources, best first.

        Returns:
            Numbered snippets, each preceded by its file path.
        """
        parts: List[str] = []
        budget = MAX_CONTEXT_TOKENS
        for number, source in enumerate(sources, start=1):
            token_ids = self.tokenizer(
                read_snippet(source), add_special_tokens=False
            )["input_ids"]
            if not token_ids:
                continue
            snippet = self.tokenizer.decode(token_ids[:budget])
            parts.append(f"[{number}] {source.file_path}\n{snippet}")
            budget -= len(token_ids)
            if budget <= 0:
                break
        return "\n\n".join(parts)

    def answer(self, question: str, sources: List[MinimalSource]) -> str:
        """Answer a question from its retrieved sources.

        Args:
            question: The user's question.
            sources: Retrieved sources, best first.

        Returns:
            The generated answer, or a notice when there is no context.
        """
        context = self.build_context(sources)
        if not context:
            return NO_CONTEXT_ANSWER
        return str(self.chain.invoke(
            {"context": context, "question": question}
        ))


def answer_dataset(
    student_search_results_path: str,
    save_directory: str,
    model_id: str = ANSWER_MODEL,
    processed_dir: str = PROCESSED_DIR,
) -> Path:
    """Generate an answer for every question of a search-results file.

    Args:
        student_search_results_path: JSON written by search_dataset.
        save_directory: Directory where the answers file is written; it
            keeps the input file's name.
        model_id: Hugging Face model used to answer.
        processed_dir: Directory holding the LLM cache.

    Returns:
        The path of the written StudentSearchResultsAndAnswer JSON.

    Raises:
        RagError: For a missing or invalid input file.
    """
    results = load_model(student_search_results_path, StudentSearchResults)
    total = len(results.search_results)
    print(f"Loaded {total} questions")
    answerer = Answerer(model_id, processed_dir)
    answers: List[MinimalAnswer] = []
    for item in tqdm(results.search_results, desc="Answering"):
        try:
            text = answerer.answer(item.question, item.retrieved_sources)
        except Exception as error:  # keep going: one bad question != no file
            text = f"Error: could not generate an answer ({error})."
        answers.append(MinimalAnswer(**item.model_dump(), answer=text))
    print(f"Processed {len(answers)} of {total} questions")
    output = StudentSearchResultsAndAnswer(
        search_results=answers, k=results.k
    )
    return save_model(
        output, save_directory, Path(student_search_results_path).name
    )
