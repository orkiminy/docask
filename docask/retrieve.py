"""Retrieval for RAG: chunk documents, then find the chunks most relevant to a question.

Two retrievers with the same interface so they can be compared in evaluation:
- TfidfRetriever : sparse keyword matching (no downloads)
- DenseRetriever : semantic embeddings from a Hugging Face sentence-embedding model
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class Chunk:
    doc_id: str
    text: str


@dataclass
class Hit:
    doc_id: str
    text: str
    score: float


def chunk_text(doc_id: str, text: str, max_words: int = 80, overlap: int = 20) -> list[Chunk]:
    """Split long text into overlapping word windows so each chunk fits the model and keeps context."""
    words = text.split()
    if len(words) <= max_words:
        return [Chunk(doc_id, text)]
    step = max_words - overlap
    return [Chunk(doc_id, " ".join(words[i:i + max_words])) for i in range(0, len(words) - overlap, step)]


class _Retriever:
    def __init__(self):
        self.chunks: list[Chunk] = []

    def index(self, docs: list[tuple[str, str]]) -> "_Retriever":
        """docs: list of (doc_id, text)."""
        self.chunks = [c for doc_id, text in docs for c in chunk_text(doc_id, text)]
        self._build()
        return self

    def search(self, query: str, k: int = 3) -> list[Hit]:
        scores = self._scores(query)
        hits, seen = [], set()
        for i in np.argsort(-scores):
            c = self.chunks[i]
            if c.doc_id in seen:  # one hit per document
                continue
            seen.add(c.doc_id)
            hits.append(Hit(c.doc_id, c.text, float(scores[i])))
            if len(hits) == k:
                break
        return hits


class TfidfRetriever(_Retriever):
    name = "tfidf"

    def _build(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([c.text for c in self.chunks])

    def _scores(self, query: str) -> np.ndarray:
        return (self.matrix @ self.vec.transform([query]).T).toarray().ravel()


class DenseRetriever(_Retriever):
    """Mean-pooled embeddings from a Hugging Face model (default: all-MiniLM-L6-v2), cosine similarity."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.model_name = model_name
        self.name = model_name.split("/")[-1]
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = self.tokenizer(texts[i:i + 64], padding=True, truncation=True, max_length=256, return_tensors="pt")
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (hidden * mask).sum(1) / mask.sum(1)  # mean pooling over real tokens
                out.append(torch.nn.functional.normalize(emb, dim=-1).numpy())
        return np.vstack(out)

    def _build(self):
        self.matrix = self.embed([c.text for c in self.chunks])

    def _scores(self, query: str) -> np.ndarray:
        return self.matrix @ self.embed([query])[0]


def _minmax(x: np.ndarray) -> np.ndarray:
    """Rescale scores to 0-1 so keyword and semantic scores are comparable."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class HybridRetriever(_Retriever):
    """Hybrid search: alpha * keyword score + (1 - alpha) * semantic score.

    Keyword search is precise on exact names and IDs; semantic search handles paraphrases.
    Each score list is min-max normalized first, because TF-IDF and cosine scores live on
    different scales.
    """

    def __init__(self, alpha: float = 0.5, dense: "DenseRetriever | None" = None):
        super().__init__()
        self.alpha = alpha
        self.sparse = TfidfRetriever()
        self.dense = dense or DenseRetriever()   # reuse an already-loaded model if given
        self.name = f"hybrid (alpha={alpha})"

    def _build(self):
        self.sparse.chunks = self.dense.chunks = self.chunks
        self.sparse._build()
        self.dense._build()

    def _scores(self, query: str) -> np.ndarray:
        return (self.alpha * _minmax(self.sparse._scores(query))
                + (1 - self.alpha) * _minmax(self.dense._scores(query)))
