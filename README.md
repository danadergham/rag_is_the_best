*This project has been created as part of the 42 curriculum by ddergham.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about the
[vLLM](https://github.com/vllm-project/vllm) codebase (v0.10.1, shipped in
`data/raw/vllm-0.10.1/`).

Instead of hoping a small model "knows" vLLM, the system:

1. **indexes** the corpus (splits every file into chunks, builds a search index),
2. **retrieves** the chunks most relevant to a question,
3. **generates** a grounded answer with `Qwen/Qwen3-0.6B` from those chunks only,
4. **measures** retrieval quality with recall@k.

Wherever the subject lets us choose *how*, we use **LangChain** instead of
writing our own: text splitters, `BM25Retriever`, `HuggingFacePipeline`,
`DirectoryLoader`, `ChatPromptTemplate`, FAISS, `EnsembleRetriever`, the indexing API, caches.
The five bonuses are implemented (see [Bonus part](#bonus-part)).

## Instructions

Requirements: Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
make install        # uv sync: installs everything from pyproject.toml / uv.lock
make lint           # flake8 + mypy with the subject's flags (make lint-strict: mypy --strict)
make clean          # remove __pycache__ and .mypy_cache
make run            # builds the index; any command: make run ARG='search "query" -k 5'
make debug          # runs the CLI under pdb
```

The first `answer` / `index_semantic` call downloads `Qwen/Qwen3-0.6B`
(~1.2 GB) and `all-MiniLM-L6-v2` from Hugging Face into its cache
(`~/.cache/huggingface`, or `$HF_HOME`). Make sure that disk has room.

### Commands

All commands: `uv run python -m src <command> [options]` (Python Fire).

| Command | What it does |
|---|---|
| `index --max_chunk_size 2000` | Chunk `data/raw/` and build the BM25 index in `data/processed/` |
| `search "<query>" -k 10` | Print the top-k sources (`path [first:last]`) |
| `search_dataset --dataset_path P -k 10 --save_directory D` | Search a whole dataset, write a `StudentSearchResults` JSON |
| `answer "<query>" -k 10` | Retrieve, then answer with Qwen3-0.6B |
| `answer_dataset --student_search_results_path P --save_directory D` | Write a `StudentSearchResultsAndAnswer` JSON |
| `evaluate --student_search_results_path P --dataset_path G` | Our own Recall@1/3/5/10 |
| `index_semantic [--rebuild=True]` | *(bonus)* build/update the FAISS embedding index |
| `serve [--host H --port N]` | *(bonus)* local HTTP API |

`search`, `search_dataset` and `answer` accept `--method lexical|semantic|hybrid`
(default `lexical`). Every path is a CLI argument (the defaults are only
conveniences). Bad input (empty query, `k=0`, missing file, malformed JSON...)
prints `Error: ...` on stderr and exits with code 1: no traceback.

### Example usage

```text
$ uv run python -m src index --max_chunk_size 2000
Chunking: 100%|██████████| 1965/1965 [00:00<00:00, 9837.95it/s]
Ingestion complete! Indexed 14624 chunks under data/processed/

$ uv run python -m src search "How to configure the OpenAI server?" -k 3
data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py [3140:5080]
data/raw/vllm-0.10.1/docs/getting_started/quickstart.md [8079:9261]
data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md [0:1951]

$ uv run python -m src answer "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" -k 5
The HTTP endpoint used to dynamically load a LoRA adapter in vLLM is `/v1/load_lora_adapter`.

Sources:
  data/raw/vllm-0.10.1/docs/features/lora.md [4695:6098]
  ...

# The full pipeline used by the evaluator
uv run python -m src index --max_chunk_size 2000
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    -k 10 --save_directory data/output/search_results/UnansweredQuestions
./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json   # Recall@1/3/5/10
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

## System architecture

```
data/raw/ ──► chunking.py ──► indexing.py ──► data/processed/chunks.json   (text + path + offsets)
 (.py .md .txt)  (LangChain      │            data/processed/bm25.pkl      (BM25Retriever, cached)
                  splitters)     └─(bonus)──► semantic.py ─► faiss_index/  (MiniLM vectors)

question ──► retrieval.py (Searcher: lexical | semantic | hybrid) ──► top-k MinimalSource
                                                                          │ path + [first:last]
                         generation.py  ◄── reads the text at those offsets
                  ChatPromptTemplate | Qwen3-0.6B | parser ──► answer
```

| Module | Role |
|---|---|
| `models.py` | pydantic models of the subject (+ `Chunk`, `ChunkList`) |
| `chunking.py` | one splitter per file type, exact offsets |
| `tokenizer.py` | code-aware tokenizer used by BM25 |
| `bm25.py` | LangChain `BM25Retriever` (build / load) |
| `indexing.py` | corpus walk, `chunks.json`, persistence |
| `retrieval.py` | `Searcher` + `search_dataset` |
| `generation.py` | `Answerer` (token-budgeted context, Qwen3, LLM cache) + `answer_dataset` |
| `evaluation.py` | recall@k with IoU > 0.05 |
| `semantic.py` | bonus: embeddings, FAISS, incremental indexing, embedding cache |
| `api.py` | bonus: FastAPI app |
| `__main__.py` | Fire CLI; every command is wrapped so no traceback can escape |

## Chunking strategy

Only `.py`, `.md` and `.txt` files are indexed (1,965 of the corpus' 2,894
files: every question of the reference datasets cites one of these types; the
other files, e.g. 403 JSON or CUDA sources, would only add BM25 noise).

Three LangChain `RecursiveCharacterTextSplitter` configurations, chosen by
file extension:

* **Python code**: `from_language(Language.PYTHON)` cuts on class / function
  boundaries first, then blank lines, lines, words.
* **Markdown**: `from_language(Language.MARKDOWN)` cuts on headings, code
  fences and paragraphs first.
* **Plain text**: the default splitter (paragraphs, lines, words).

All use `chunk_size = --max_chunk_size` (default **2000**), `chunk_overlap = 200`
and `add_start_index=True`, so `chunk.metadata["start_index"]` gives the exact
character offset: `first_character_index = start`,
`last_character_index = start + len(chunk)`. Files are loaded with LangChain's
`DirectoryLoader` (`.py`, `.md`, `.txt`, hidden folders included, sorted by
path); a file that is not valid UTF-8 is skipped. File paths are stored exactly
as `DirectoryLoader` reports them, so use the default relative `--raw_dir`.
`--max_chunk_size` above 2000 is capped (with a warning).

**Effect of the chunk size** (official moulinette, `-k 10`):

| max_chunk_size | chunks | docs R@5 (public / private) | code R@5 (public / private) |
|---|---|---|---|
| 500 | 64,483 | 0.82 / 0.80 | 0.79 / 0.72 |
| 1000 | 30,102 | 0.85 / 0.83 | 0.80 / 0.75 |
| 2000 | 14,624 | 0.85 / 0.84 | 0.79 / 0.75 |

Chunks that are **too small** cut an explanation away from its keywords and
flood the top-k with fragments of the same file (R@5 falls). Chunks that are
**too large** dilute the scores (a short answer inside a long chunk) and
exceed what the model can read; above 2000 characters the moulinette rejects
them anyway. Differences between 1000 and 2000 are within noise, so we keep the
subject's default.

## Retrieval method

**BM25** (Okapi), through LangChain's `BM25Retriever` (`rank-bm25`). For a query
word `q` and a chunk `d`, the score is
`idf(q) · f(q,d)·(k1+1) / (f(q,d) + k1·(1 − b + b·|d|/avgdl))` summed over the
query words: a word counts more when it is **rare in the corpus** (idf), its
effect **saturates** with repetition (`k1 = 1.5`) and long chunks are
**normalised** (`b = 0.75`). Unlike TF-IDF, which grows linearly with the
term count and ignores length, BM25 saturates and normalises, which suits
chunks of very different lengths.

What made the difference was the **tokenizer** (`tokenizer.py`). LangChain's
default `text.split()` keeps `self.max_len(x):` as one token and cannot match a
question. Ours lowercases, splits `snake_case` / `CamelCase` into parts (and
keeps the whole identifier too), and drops a few stop words. We also put the
**file path in front of each chunk's indexed text**, because questions often name
the module (`nemotron_nas`, `fused_moe`...). On the private datasets, Recall@5 went from
0.63 → 0.84 (docs) and 0.12 → 0.75 (code) compared to whitespace splitting.

A query with no indexed word (gibberish) returns an empty list instead of an
arbitrary one (`GuardedBM25Retriever`). Results are ranked best first; the first
5 of the 10 returned are what Recall@5 looks at.

## Performance analysis

Measured with the official `exam_retrieval.sh` + moulinette on the **private**
datasets (CPU machine, 24 cores):

| Requirement | Result | Limit |
|---|---|---|
| Indexing | **8 s** | ≤ 300 s |
| 200 questions retrieved (2 × `search_dataset`, incl. start-up) | **14 s** | ≤ 90 s |
| Docs Recall@5 | **0.84** | ≥ 0.80 |
| Code Recall@5 | **0.75** | ≥ 0.50 |

Full recall curve (private / public): docs R@1/3/5/10 = 0.65/0.81/0.84/0.86 and
0.66/0.82/0.85/0.90; code R@1/3/5/10 = 0.50/0.70/0.75/0.82 and
0.47/0.75/0.79/0.87. Edge cases (`exam_edge_cases.sh`): 4/4 without traceback.

Tuning notes: we tuned on the public datasets and only checked the private ones
afterwards. BM25 parameters (`k1`, `b`), stemming, chunk overlap (0–400) and
de-duplicating overlapping hits all moved recall by less than ±3 points
(noise on 100 questions), so we kept the simplest setting.

Answers: Qwen3-0.6B needs roughly 15-25 s per question on CPU (the context is
capped at 2,000 tokens), so `answer_dataset` on 100 questions takes about
30-40 minutes; repeated questions are instant thanks to the cache.

## Design decisions

* **LangChain everywhere possible**: less code to defend, behaviour known.
* **BM25 as default method**: fast to build (8 s), no model needed, and the
  best recall in our measurements.
* **Offsets from `add_start_index`** plus a verification guard, instead of
  searching the text back in the file.
* **`answer` reads the text from disk** at the retrieved offsets, so
  `answer_dataset` only needs the search-results file (not the index).
* **Qwen3 "thinking" off** (`/no_think` at the end of the question;
  `ChatHuggingFace` applies the model's own chat template) and **greedy
  decoding**: short, deterministic, grounded answers. The prompt tells the model
  to use only the context and to reply exactly "I don't know" when the question
  is unrelated to the context or the answer is not in it.
* **Token budget**: the context passed to the model is limited to 2,000 tokens
  (best sources first, last one truncated). A longer context made the 0.6B model
  slower and vaguer.
* **Errors**: one `RagError` type for expected problems; every CLI command is
  wrapped, so even an unexpected exception ends as one `Error:` line.
* **The semantic index is a separate command** (`index_semantic`, ~4 min): the
  mandatory `index` has a 5 min limit and must stay fast.
* **The BM25 index is pickled**: loading it is faster than re-tokenizing 14k
  chunks at every cold start (it is our own local file, never downloaded).

## Challenges faced

* **Default BM25 tokenization gave 15 % code recall.** Fixed with the identifier
  splitting tokenizer and the path prefix (→ 75 %).
* **LangChain 1.x moved legacy classes** (`EnsembleRetriever`, `SQLRecordManager`,
  `CacheBackedEmbeddings`) to `langchain-classic`; we depend on it explicitly.
* **Qwen3 "thinking" mode** produced long reasoning blocks and was slow; solved
  with Qwen3's `/no_think` switch (the empty `<think></think>` it leaves is
  removed by `clean_answer`).
* **Gibberish queries** returned arbitrary chunks with BM25 (all scores zero);
  solved with the guarded retriever.
* **Hybrid was not better than lexical at k=5** with equal weights; weighting
  BM25 0.7 / semantic 0.3 brings it level and improves Recall@10 on docs.
* **Disk space**: the model downloads need ~2 GB of Hugging Face cache; on a
  small home partition set `HF_HOME` to a larger disk.

## Bonus part

All five bonuses run on CPU and are demonstrated with the commands below.

1. **Semantic embeddings**: `uv run python -m src index_semantic` embeds the
   chunks with `all-MiniLM-L6-v2` (`HuggingFaceEmbeddings`) into a FAISS store
   (`data/processed/faiss_index`). Use it alone with `--method semantic`
   (private R@5: docs 0.65, code 0.34: semantic alone is weaker than BM25 on
   identifier-heavy code questions; the model also only reads the first 256
   tokens of each chunk).
2. **Hybrid retrieval**: `--method hybrid` merges BM25 and FAISS rankings with
   LangChain's `EnsembleRetriever` (reciprocal-rank fusion, weights 0.7/0.3).
   Private R@5/R@10: docs 0.84/0.89, code 0.67/0.80.
3. **Incremental indexing**: `index_semantic` uses LangChain's `index()` with an
   `SQLRecordManager` (`cleanup="full"`). Each chunk is hashed; only new or
   changed chunks are embedded, vanished ones are deleted. Re-running without
   change: `{'num_added': 0, 'num_skipped': 14624, ...}`. After editing one doc
   and running `index` then `index_semantic`: `num_added: 1, num_deleted: 1,
   num_skipped: 14623`.
4. **Caching**: (a) the BM25 index is pickled (fast cold start); (b)
   `CacheBackedEmbeddings` + `LocalFileStore` cache document and query
   embeddings on disk (rebuilding the semantic index: 255 s → 42 s); (c)
   `SQLiteCache` (`set_llm_cache`) caches Qwen's answers: asking the same
   question twice does not run the model a second time.
5. **Local HTTP API**: `uv run python -m src serve --port 8000` starts a FastAPI
   app: `POST /search` and `POST /answer` with the parameters `query` and `k`;
   interactive docs at `/docs`.
   ```bash
   curl -X POST "localhost:8000/search?query=How%20to%20load%20a%20LoRA%20adapter%3F&k=3"
   ```

## Resources

* LangChain docs: [text splitters](https://python.langchain.com/docs/concepts/text_splitters/),
  [BM25Retriever](https://python.langchain.com/api_reference/community/retrievers/langchain_community.retrievers.bm25.BM25Retriever.html),
  [indexing API](https://python.langchain.com/docs/how_to/indexing/),
  [caching embeddings](https://python.langchain.com/docs/how_to/caching_embeddings/)
* [rank-bm25](https://github.com/dorianbrown/rank_bm25); Robertson & Zaragoza,
  *The Probabilistic Relevance Framework: BM25 and Beyond*
* [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B),
  [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
* [FAISS](https://github.com/facebookresearch/faiss), [FastAPI](https://fastapi.tiangolo.com/),
  [Python Fire](https://github.com/google/python-fire), [pydantic](https://docs.pydantic.dev/)

**How AI was used.** Claude Code (an AI assistant) helped with: planning the milestones.