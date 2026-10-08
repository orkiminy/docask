"""Answer generation: grounded LLM answers (Claude) when an API key is set, otherwise an extractive fallback."""
from __future__ import annotations

import os
import re

from .retrieve import Hit

STOP = set("a an the of to in on for and or is are was be by with at from what who when how which does do did "
           "will would about between much many s".split())
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9$,]+", text.lower()) if t not in STOP}


# What kind of value each question type asks for, so we prefer sentences that contain one.
ANSWER_TYPES = [
    (r"\bhow much\b|\bsalary\b|\bowe\b|\bcost\b|\btotal\b", r"\$[\d,]+"),
    (r"\bwhen\b|\bdeadline\b|\bdate\b", r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}"),
    (r"\bhow long\b", r"\b\d+ years?\b"),
    (r"\bwho\b", r"\b[A-Z][a-z]+ [A-Z][a-z]+\b"),
    (r"\bwhich state\b|\bgoverned\b|\blaw\b", r"\b(?:New York|Delaware|Illinois)\b"),
    (r"\brisk\b", r"\b(?:low|moderate|high)\b"),
    (r"\brole\b|\bposition\b", r"\b(?:Engineer|Manager|Paralegal|Analyst|Designer)\b"),
]


def _expected_type(question: str) -> str | None:
    q = question.lower()
    for trigger, value_pattern in ANSWER_TYPES:
        if re.search(trigger, q):
            return value_pattern
    return None


def extractive_answer(question: str, hits: list[Hit]) -> dict:
    """Pick the best sentence from the retrieved chunks, with no LLM.

    Score = a big bonus if the sentence contains the kind of value the question asks for
    (a dollar amount for "how much", a date for "when", ...) + words shared with the
    question. The top-ranked chunk is strongly preferred (retrieval is usually right). Deterministic and cheap,
    which makes it a useful baseline to measure an LLM against.
    """
    q = _tokens(question)
    want = _expected_type(question)
    best = (-1.0, "", "")
    for rank, h in enumerate(hits):
        for sent in re.split(r"(?<=[.!?])\s+", h.text):
            names = set(re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", question))
            has_value = bool(want and re.search(want, sent)) and not (want.endswith("[a-z]+\\b") and
                                                                      set(re.findall(want, sent)) <= names)
            score = len(q & _tokens(sent)) + (10.0 if has_value else 0.0) - 20.0 * rank
            if score > best[0]:
                best = (score, sent.strip(), h.doc_id)
    return {"answer": best[1], "source": best[2], "method": "extractive"}


def llm_answer(question: str, hits: list[Hit], model: str | None = None) -> dict:
    """Ask Claude to answer ONLY from the retrieved passages and to cite the document id."""
    import anthropic

    context = "\n\n".join(f"[{h.doc_id}] {h.text}" for h in hits)
    prompt = ("Answer the question using only the documents below. Reply with a short answer, then the "
              "document id in brackets. If the documents do not contain the answer, say \"I don't know\".\n\n"
              f"Documents:\n{context}\n\nQuestion: {question}")
    msg = anthropic.Anthropic().messages.create(
        model=model or os.environ.get("DOCASK_LLM_MODEL", DEFAULT_MODEL), max_tokens=200,
        messages=[{"role": "user", "content": prompt}])
    text = msg.content[0].text.strip()
    cited = re.search(r"\[([\w-]+)\]", text)
    return {"answer": text, "source": cited.group(1) if cited else (hits[0].doc_id if hits else ""), "method": "llm"}


def answer(question: str, hits: list[Hit]) -> dict:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return llm_answer(question, hits)
        except ImportError:
            pass  # anthropic package not installed: fall back
    return extractive_answer(question, hits)
