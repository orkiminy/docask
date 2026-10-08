"""Evaluation harness: classification, retrieval, and answer quality, written to eval/results.json.

    python -m docask.evaluate              # full run with Hugging Face models (needs internet once)
    python -m docask.evaluate --offline    # no downloads: baseline + tiny offline transformer + TF-IDF retrieval

Run it after every change and compare results.json: a drop is a regression.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from sklearn.metrics import accuracy_score, f1_score

from .answer import answer
from .classify import BaselineClassifier, TransformerClassifier
from .data import make_corpus, train_test_split
from .retrieve import DenseRetriever, HybridRetriever, TfidfRetriever


def eval_classifier(clf, train, test) -> dict:
    t0 = time.time()
    clf.fit([d.text for d in train], [d.label for d in train])
    preds = clf.predict([d.text for d in test])
    gold = [d.label for d in test]
    errors = [{"doc_id": d.doc_id, "gold": g, "pred": str(p)} for d, g, p in zip(test, gold, preds) if g != p]
    return {"model": clf.name, "accuracy": round(accuracy_score(gold, preds), 3),
            "macro_f1": round(f1_score(gold, preds, average="macro"), 3),
            "train_seconds": round(time.time() - t0, 1), "errors": errors[:10]}


def eval_retriever(retriever, docs, questions, k: int = 3) -> dict:
    retriever.index([(d.doc_id, d.text) for d in docs])
    hit1 = hitk = rr = 0.0
    for q in questions:
        ids = [h.doc_id for h in retriever.search(q.question, k=10)]
        if ids and ids[0] == q.doc_id:
            hit1 += 1
        if q.doc_id in ids[:k]:
            hitk += 1
        if q.doc_id in ids:
            rr += 1 / (ids.index(q.doc_id) + 1)
    n = len(questions)
    return {"retriever": retriever.name, "recall@1": round(hit1 / n, 3), f"recall@{k}": round(hitk / n, 3),
            "mrr@10": round(rr / n, 3)}


def eval_answers(retriever, questions, k: int = 3) -> dict:
    """End-to-end RAG: retrieve, answer, then check the expected value appears in the answer."""
    correct = cited = 0
    for q in questions:
        out = answer(q.question, retriever.search(q.question, k=k))
        correct += q.answer.lower() in out["answer"].lower()
        cited += out["source"] == q.doc_id
        method = out["method"]
    n = len(questions)
    return {"method": method, "answer_accuracy": round(correct / n, 3), "correct_citation": round(cited / n, 3), "n": n}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="no Hugging Face downloads")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--questions", type=int, default=200, help="questions used for retrieval/answer eval")
    args = ap.parse_args(argv)

    docs, questions = make_corpus()
    train, test = train_test_split(docs)
    questions = questions[:: max(1, len(questions) // args.questions)][: args.questions]

    results = {"data": {"documents": len(docs), "train": len(train), "test_unseen_style": len(test),
                        "questions": len(questions)}}
    results["classification"] = [
        eval_classifier(BaselineClassifier(), train, test),
        eval_classifier(TransformerClassifier(pretrained=not args.offline,
                                              epochs=args.epochs if not args.offline else 8,
                                              lr=5e-5 if not args.offline else 3e-3), train, test),
    ]
    retrievers = [TfidfRetriever()]
    if not args.offline:
        retrievers.append(DenseRetriever())
        retrievers.append(HybridRetriever(alpha=0.5, dense=retrievers[1]))
    results["retrieval"] = [eval_retriever(r, docs, questions) for r in retrievers]
    best = max(retrievers, key=lambda r: next(x["recall@1"] for x in results["retrieval"] if x["retriever"] == r.name))
    results["answers"] = eval_answers(best, questions[:50] if os.environ.get("ANTHROPIC_API_KEY") else questions)
    results["answers"]["retriever"] = best.name

    os.makedirs("eval", exist_ok=True)
    with open("eval/results.json", "w") as f:
        json.dump(results, f, indent=2)
    _print(results)
    return results


def _print(r: dict):
    print(f"\nData: {r['data']}\n\nClassification (test set = writing style never seen in training)")
    for c in r["classification"]:
        print(f"  {c['model']:<26} accuracy {c['accuracy']:.3f}  macro-F1 {c['macro_f1']:.3f}  ({c['train_seconds']}s)")
    print("\nRetrieval")
    for x in r["retrieval"]:
        print(f"  {x['retriever']:<26} recall@1 {x['recall@1']:.3f}  recall@3 {x['recall@3']:.3f}  MRR@10 {x['mrr@10']:.3f}")
    a = r["answers"]
    print(f"\nAnswers ({a['method']}, {a['retriever']} retrieval, {a['n']} questions): "
          f"accuracy {a['answer_accuracy']:.3f}, correct citation {a['correct_citation']:.3f}\n\nSaved eval/results.json")


if __name__ == "__main__":
    main()
