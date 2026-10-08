# DocAsk: classify documents and ask questions about them

DocAsk is a small generative-AI document assistant in Python. It does three things a
document platform needs:

1. **Classifies** each document (NDA, invoice, employment agreement, legal memo, meeting minutes)
   by fine-tuning a **Hugging Face transformer (DistilBERT) in PyTorch**, compared against a TF-IDF baseline.
2. **Answers questions** with **retrieval-augmented generation (RAG)**: it finds the most relevant
   passages with **semantic embeddings (MiniLM)** and answers only from them, citing the source document.
   With an Anthropic API key it uses Claude; without one it uses a deterministic extractive answerer.
3. **Evaluates everything** (classification, retrieval, end-to-end answers) and runs the checks in
   **CI as quality gates**, so a change that makes the system worse fails the build.

## How it works

```
documents ──► classifier (TF-IDF baseline | fine-tuned DistilBERT) ──► type + confidence
    │                                                                   (confidence < 0.5 → needs_review)
    └─► chunk ──► embed (MiniLM | TF-IDF) ──► index
                                              ▲
question ──► embed ──► top-k chunks ──────────┘──► answer (Claude, grounded | extractive) + cited doc id
```

| File | What it does |
|---|---|
| `docask/data.py` | Generates 300 documents (5 types × 3 writing styles) with known facts and 600 questions with known answers |
| `docask/classify.py` | TF-IDF + logistic regression baseline; DistilBERT fine-tuning with a plain PyTorch training loop |
| `docask/retrieve.py` | Chunking with overlap; TF-IDF, dense (mean-pooled MiniLM embeddings, cosine), and hybrid retrievers |
| `docask/answer.py` | Grounded LLM answers with citations (Claude), or an extractive fallback |
| `docask/evaluate.py` | Accuracy, macro-F1, recall@k, MRR, answer accuracy, citation accuracy → `eval/results.json` |
| `docask/api.py` | FastAPI: `POST /documents`, `POST /classify`, `POST /ask`, `GET /health` |
| `tests/` | Unit tests + quality gates (run in GitHub Actions) |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                  # tests + quality gates (no downloads)
python -m docask.evaluate                  # full evaluation; downloads DistilBERT + MiniLM the first time
uvicorn docask.api:app --reload            # API at http://127.0.0.1:8000/docs
DOCASK_MODE=hf uvicorn docask.api:app      # same API using the Hugging Face models
export ANTHROPIC_API_KEY=...               # optional: grounded answers from Claude instead of extractive
docker build -t docask . && docker run -p 8000:8000 docask
```

## Evaluation design

- **The test set uses a writing style the models never saw in training.** Each document type has
  three templates; training uses two, testing uses the third. A random split would let a model
  pass by memorizing templates.
- **Always compare to a simple baseline.** A transformer has to beat TF-IDF + logistic regression
  to justify its cost.
- **Retrieval and answers are scored separately**, so when an answer is wrong you can tell whether
  retrieval failed (wrong document) or generation failed (right document, wrong sentence).

## Results

All numbers from `python -m docask.evaluate` (seed 7). Classification is tested on a writing
style the models never saw in training (100 documents).

| Classifier | Accuracy | Macro-F1 | Train time (Mac CPU) |
|---|---|---|---|
| TF-IDF + logistic regression (baseline) | 0.970 | 0.970 | 0.1 s |
| **DistilBERT, fine-tuned 3 epochs** | **1.000** | **1.000** | 75 s |
| Tiny BERT trained from scratch (`--offline`) | 0.200 | 0.067 | 13 s |

| Retriever (200 questions) | Recall@1 | Recall@3 | MRR@10 |
|---|---|---|---|
| TF-IDF (keyword) | 0.990 | 1.000 | 0.995 |
| all-MiniLM-L6-v2 (semantic) | 0.960 | 1.000 | 0.979 |
| **Hybrid (0.5 × keyword + 0.5 × semantic)** | **0.995** | **1.000** | **0.998** |

End-to-end answers (extractive answerer, hybrid retrieval, 200 questions): **0.955 accuracy, 0.995 correct citation**.

### What the results show

- **Pretraining is what makes a transformer work on small data.** The same architecture trained
  from scratch reaches 100% on its training documents but 20% (chance) on the unseen style: with
  200 examples it memorizes templates. Fine-tuning pretrained DistilBERT reaches 100%, and fixes
  the baseline's errors (NDAs written without the usual legal vocabulary, which TF-IDF labeled as
  employment agreements), because it brings general language understanding.
- **Semantic search is not automatically better.** Here the questions hinge on exact company names
  ("Harbor Analytics Inc"), and keyword matching finds exact names more reliably than embeddings,
  which place similar names ("Harbor Analytics Group") close together.
- **Hybrid search beats both.** Rescaling each retriever's scores to 0-1 and averaging them
  (`HybridRetriever`) reaches recall@1 0.995: keyword matching pins down the exact name, and the
  semantic score breaks ties on meaning. It cut retrieval misses in half versus keyword alone
  (2 to 1 out of 200) and lifted end-to-end citation accuracy from 0.990 to 0.995.
  Alpha = 0.5 was not tuned; tuning it properly would need a separate validation set.
- **Where the extractive answerer fails:** headers like "To: Litigation Team. From: ..." get split
  into separate sentences, so "Who wrote the memo?" sometimes returns the wrong line. A grounded LLM
  answer fixes this kind of error, which is what the `ANTHROPIC_API_KEY` path is for.
- **Caveat:** 100% is on clean synthetic documents with a 100-document test set. Real documents are
  longer, messier, and often scanned.

## Limits and next steps

- The documents are synthetic. Real documents are longer, messier, and often scanned, so OCR and
  layout-aware chunking would matter.
- The index is rebuilt in memory on every insert. A real system would use a vector database.
- Next: tune the hybrid weight on a validation set, harder test data (paraphrased questions where
  keywords don't match), a confidence threshold tuned on a validation set, and monitoring of answer
  quality over time.
