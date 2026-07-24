# IIT Madras DL Gen-AI Project [2026]

## Milestone 3: Context Augmentation with RAG Pipelines

**Short Summary**

*RAG* stands for Retrieval-Augmented Generation. It is an artificial intelligence technique that improves a large language model (LLM) by looking up facts from an external knowledge base before writing an answer.

* Understood the limitations of general LLMs
* Learnt the RAG pipeline
* Loaded a simple pre-built vector database.
* Retrieved external context based on the question prompt.
* Fed the retrieved context + prompt + choices into the model to improve reasoning.

**Description**

Answers five-option multiple-choice questions for the Kaggle "Smart MCQ Solver Challenge" by retrieving relevant Wikipedia passages with a dense embedding index and prompting a local instruction-tuned LLM (Phi-3-mini) to rank the three most likely answers. This is a retrieval + prompting pipeline, not a fine-tuned classifier — no model weights are trained.

Notebook: `rag-23f3001480.ipynb`

## Pipeline overview

1. **Load competition data** — `train.csv` / `test.csv` from the Smart MCQ Solver Challenge.
2. **Load a Wikipedia corpus** — the auxiliary Kaggle dataset `mbanaei/stem-wiki-cohere-no-emb` is pulled via `kagglehub`
3. **Build a retrieval index** — a random 5,000-passage sample of the Wikipedia dump is embedded with `all-MiniLM-L6-v2` (Sentence-Transformers) and indexed with FAISS (`IndexFlatIP`, cosine similarity via normalized inner product). The index and passage metadata are persisted to disk (`wiki.index`, `wiki_meta.parquet`).
4. **Retrieve context** — for a given question, the top-3 most similar passages are fetched from the FAISS index.
5. **Prompt an LLM to rank answers** — `microsoft/Phi-3-mini-4k-instruct` is loaded and given the question, its five options, and the retrieved passages, then asked to return the three most likely correct options ranked best-first.
6. **Evaluate** — accuracy, macro F1, and MAP@3 are computed on a 200-row labeled sample drawn from `train.csv` (Note that `test.csv` has no ground truth, and in RAG, the model did NOT learn over train dataser, it learned form the knowledge_base).
7. **Generate submission** — the same ranking procedure runs over the full test set and results are written to `submission.csv`.

## Data and models used

| Component | Source |
|---|---|
| Competition data | `/kaggle/input/competitions/smart-mcq-solver-challenge/{train,test,sample_submission}.csv` |
| Wikipedia corpus | `mbanaei/stem-wiki-cohere-no-emb` (via kagglehub) — columns: `id`, `title`, `text`, `url`, `wiki_id`, `views`, `paragraph_id`, `langs` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Generator model | `microsoft/Phi-3-mini-4k-instruct` (loaded in `bfloat16` on GPU, `float32` on CPU) |

## Retrieval details

- Corpus is **subsampled to 5,000 passages** (`random_state=42`) out of the full Wikipedia dump, noted in-notebook as a speed tradeoff for Kaggle's session time limits — not the full corpus.
- Embeddings are L2-normalized so FAISS inner product search (`IndexFlatIP`) is equivalent to cosine similarity.
- `retrieve(query, k=3)` returns the raw passage text for the top-*k* matches to a query string.

## Prompting details

For each question, the prompt supplied to Phi-3-mini includes:
- The retrieved passages as context
- The question text
- All five options (A–E)
- An instruction to respond with only the three most likely option letters, ranked best-first, comma-separated (e.g. `"B, D, A"`)

Generation is deterministic (`do_sample=False`, `max_new_tokens=10`). The response is parsed by scanning generated characters for valid option letters (A–E), preserving first-seen order and deduplicating; if fewer than 3 valid letters are found, remaining option letters are appended to pad the ranking to 3.

## Evaluation

Run on a random 200-row sample of `train.csv` (`random_state=42`), since `test.csv` has no labels:

| Metric | Value |
|---|---|
| Accuracy (top-1) | 0.6100 |
| Macro F1 | 0.5996 |
| MAP@3 | 0.7625 |

MAP@3 is computed by awarding `1/rank` when the true answer appears in the top-3 predictions (1.0 first place, 0.5 second, 0.333 third, 0 otherwise), averaged over all evaluated rows — the same scoring logic the competition uses.

## Output

`submission.csv` with columns `ID` and `Prediction`, where `Prediction` is a space-separated string of the top-3 ranked option letters per test question (e.g. `"A E D"`). Row count and column-name assertions are run before writing the file.

## Requirements

```
pandas
numpy
faiss-gpu (or faiss-cpu)
sentence-transformers
transformers
accelerate
torch
scikit-learn
datasets
kagglehub
```

Requires GPU access for practical runtime (Phi-3-mini inference over the full test set is described as the slow part of the pipeline).

---
By: [Naini Diwan](https://naini-diwan.github.io/Hello-Naini/)

Roll No.: 23f3001480
