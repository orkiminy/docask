"""DocAsk API: add documents (auto-classified), classify text, and ask questions over your documents.

    uvicorn docask.api:app --reload        # then open http://127.0.0.1:8000/docs

Set DOCASK_MODE=hf to use the Hugging Face models (DistilBERT classifier + MiniLM retrieval);
the default "fast" mode uses the TF-IDF models so it starts instantly with no downloads.
"""
from __future__ import annotations

import os
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .answer import answer
from .classify import BaselineClassifier, TransformerClassifier
from .data import make_corpus
from .retrieve import DenseRetriever, TfidfRetriever


class DocIn(BaseModel):
    text: str = Field(min_length=1)
    doc_id: str | None = None


class TextIn(BaseModel):
    text: str = Field(min_length=1)


class AskIn(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(3, ge=1, le=10)


def create_app(mode: str | None = None, seed_corpus: bool = True) -> FastAPI:
    mode = mode or os.environ.get("DOCASK_MODE", "fast")
    app = FastAPI(title="DocAsk", description="Classify documents and answer questions about them (RAG).")
    corpus, _ = make_corpus()
    clf = TransformerClassifier() if mode == "hf" else BaselineClassifier()
    clf.fit([d.text for d in corpus], [d.label for d in corpus])
    retriever = DenseRetriever() if mode == "hf" else TfidfRetriever()
    store: dict[str, dict] = {}
    if seed_corpus:
        store.update({d.doc_id: {"text": d.text, "label": d.label} for d in corpus})
    if store:
        retriever.index([(k, v["text"]) for k, v in store.items()])

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": mode, "documents": len(store)}

    @app.post("/classify")
    def classify(body: TextIn):
        label, conf = clf.predict_with_confidence([body.text])[0]
        return {"label": label, "confidence": round(conf, 3), "needs_review": conf < 0.5}

    @app.post("/documents")
    def add_document(body: DocIn):
        doc_id = body.doc_id or f"doc-{uuid.uuid4().hex[:8]}"
        label, conf = clf.predict_with_confidence([body.text])[0]
        store[doc_id] = {"text": body.text, "label": label}
        retriever.index([(k, v["text"]) for k, v in store.items()])  # simple full re-index; fine at this scale
        return {"doc_id": doc_id, "label": label, "confidence": round(conf, 3)}

    @app.post("/ask")
    def ask(body: AskIn):
        if not store:
            raise HTTPException(400, "no documents indexed yet")
        hits = retriever.search(body.question, k=body.k)
        out = answer(body.question, hits)
        out["sources"] = [{"doc_id": h.doc_id, "score": round(h.score, 3), "type": store[h.doc_id]["label"]} for h in hits]
        return out

    return app


app = create_app()
