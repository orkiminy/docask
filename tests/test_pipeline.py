"""Fast tests that need no downloads. The quality thresholds act as regression gates in CI:
if a change makes classification or retrieval worse, the build fails."""
import importlib.util

import pytest
from fastapi.testclient import TestClient

from docask.answer import extractive_answer
from docask.api import create_app
from docask.classify import BaselineClassifier
from docask.data import LABELS, make_corpus, train_test_split
from docask.evaluate import eval_classifier, eval_retriever
from docask.retrieve import Hit, TfidfRetriever, chunk_text

HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


def test_corpus_is_balanced_and_split_by_unseen_style(corpus):
    docs, questions = corpus
    assert len(docs) == 300 and len(questions) == 600
    assert {d.label for d in docs} == set(LABELS)
    train, test = train_test_split(docs)
    assert {d.style for d in train}.isdisjoint({d.style for d in test})
    for q in questions[:50]:  # every expected answer really is in its source document
        doc = next(d for d in docs if d.doc_id == q.doc_id)
        assert q.answer in doc.text


def test_chunking_overlaps_long_text():
    chunks = chunk_text("d", " ".join(f"w{i}" for i in range(200)), max_words=80, overlap=20)
    assert len(chunks) == 3
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]


def test_baseline_classifier_quality_gate(corpus):
    train, test = train_test_split(corpus[0])
    r = eval_classifier(BaselineClassifier(), train, test)
    assert r["accuracy"] >= 0.90, r


def test_tfidf_retrieval_quality_gate(corpus):
    docs, questions = corpus
    r = eval_retriever(TfidfRetriever(), docs, questions[::5])
    assert r["recall@3"] >= 0.95, r


def test_extractive_answer_returns_grounded_sentence():
    hits = [Hit("inv-1", "INVOICE 12. Bill from A to B. Total due: $5,000. Payment due by June 1, 2026.", 1.0)]
    out = extractive_answer("How much is the total due?", hits)
    assert "$5,000" in out["answer"] and out["source"] == "inv-1"


def test_api_classify_add_and_ask():
    client = TestClient(create_app(mode="fast", seed_corpus=False))
    new = client.post("/documents", json={"doc_id": "acme-inv", "text":
                      "INVOICE INV-7777. Bill from Acme Robotics to Zenith Foods. Total due: $9,450. Payment due by March 3, 2026."}).json()
    assert new["label"] == "invoice"
    ans = client.post("/ask", json={"question": "How much does Zenith Foods owe Acme Robotics?"}).json()
    assert "$9,450" in ans["answer"] and ans["sources"][0]["doc_id"] == "acme-inv"
    assert client.post("/classify", json={"text": "Meeting minutes. Chaired by Ana. Next meeting Friday."}).json()["label"] == "meeting_minutes"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_offline_transformer_training_loop_works(corpus):
    """Smoke test of the PyTorch fine-tuning code path (tiny model, no downloads).

    It checks the model can learn its training data. It does NOT check generalization:
    a tiny model trained from scratch memorizes the training templates (see README),
    which is exactly why the real run uses a pretrained model.
    """
    from docask.classify import TransformerClassifier
    train, _ = train_test_split(corpus[0])
    clf = TransformerClassifier(pretrained=False, epochs=6, lr=3e-3)
    clf.fit([d.text for d in train], [d.label for d in train])
    assert clf.losses[-1] < clf.losses[0] / 5
    preds = clf.predict([d.text for d in train])
    assert sum(p == d.label for p, d in zip(preds, train)) / len(train) >= 0.9
