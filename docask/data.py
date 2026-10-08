"""Synthetic corpus of professional documents (the kind a legal/document platform stores).

Every document is generated from a template with random but unique details, so we
always know (a) its true type, for classification, and (b) which document answers
each question, for retrieval and answer evaluation.

Templates are split: each document type has several writing styles, and the
TEST set uses a style the model never saw in training. That checks whether a
classifier learned the document type or just memorized one template.
"""
import random
from dataclasses import dataclass, field

LABELS = ["nda", "invoice", "employment_agreement", "legal_memo", "meeting_minutes"]

FIRST = ["Maya", "Daniel", "Priya", "Lucas", "Amara", "Noah", "Elena", "Omar", "Grace", "Hiro",
         "Sofia", "Ethan", "Leila", "Marco", "Nadia", "Theo", "Ines", "Kofi", "Yara", "Ben"]
LAST = ["Levin", "Okafor", "Shah", "Moreau", "Tanaka", "Russo", "Kim", "Haddad", "Novak", "Silva",
        "Brennan", "Costa", "Weiss", "Adeyemi", "Larsen", "Park", "Fischer", "Rossi", "Cohen", "Diaz"]
CO_A = ["Harbor", "Summit", "Cedar", "Atlas", "Beacon", "Northwind", "Granite", "Juniper", "Meridian", "Lumen",
        "Riverside", "Copper", "Sterling", "Bluebird", "Ironwood", "Silverline", "Oakmont", "Brightwater"]
CO_B = ["Analytics", "Logistics", "Capital", "Health", "Robotics", "Foods", "Media", "Energy", "Labs", "Partners"]
CO_C = ["Inc", "Group", "LLC", "Holdings", "Co"]
CITIES = ["New York", "Chicago", "Boston", "Austin", "Denver", "Seattle", "London", "Toronto"]
TOPICS = ["data retention policy", "vendor onboarding", "Q3 budget", "office relocation", "security audit",
          "client renewal", "hiring plan", "product launch", "pricing review", "litigation hold"]
ROLES = ["Software Engineer", "Account Manager", "Paralegal", "Data Analyst", "Product Designer", "Office Manager"]


@dataclass
class Document:
    doc_id: str
    label: str
    style: int
    text: str
    facts: dict = field(default_factory=dict)  # values a question can ask about


@dataclass
class Question:
    question: str
    doc_id: str   # the document that contains the answer
    answer: str   # string that must appear in a correct answer


def _person(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def _company(rng, used):
    while True:
        name = f"{rng.choice(CO_A)} {rng.choice(CO_B)} {rng.choice(CO_C)}"
        if name not in used:
            used.add(name)
            return name


def _date(rng):
    return f"{rng.choice(['January', 'March', 'April', 'June', 'August', 'October'])} {rng.randint(1, 28)}, 2026"


def _money(rng, lo, hi):
    return f"${rng.randint(lo, hi):,}"


# ---- templates: (text, facts, questions) builders per label and style -------------

def _nda(rng, style, co, other):
    f = {"party": co, "counterparty": other, "years": str(rng.choice([2, 3, 5])), "state": rng.choice(["New York", "Delaware", "Illinois"])}
    texts = [
        f"MUTUAL NON-DISCLOSURE AGREEMENT. This Agreement is entered into by {co} and {other}. Each party may disclose "
        f"Confidential Information to the other. The receiving party shall not disclose Confidential Information to any third "
        f"party. Obligations survive for {f['years']} years after termination. This Agreement is governed by the laws of {f['state']}.",
        f"Confidentiality Agreement between {co} (the Disclosing Party) and {other} (the Recipient). Recipient agrees to keep "
        f"all proprietary materials secret and to use them solely to evaluate a potential business relationship. The duty of "
        f"confidentiality lasts {f['years']} years. Governing law: {f['state']}.",
        f"{co} and {other} agree that any trade secrets, technical data, or business plans shared between them remain private. "
        f"Neither side will reveal such information without written consent, for a period of {f['years']} years. "
        f"Disputes will be resolved under {f['state']} law.",
    ]
    qs = [Question(f"How long do the confidentiality obligations between {co} and {other} last?", "", f"{f['years']} years"),
          Question(f"Which state's law governs the agreement between {co} and {other}?", "", f["state"])]
    return texts[style], f, qs


def _invoice(rng, style, co, other):
    f = {"vendor": co, "client": other, "total": _money(rng, 1200, 48000), "due": _date(rng), "number": f"INV-{rng.randint(1000, 9999)}"}
    texts = [
        f"INVOICE {f['number']}. Bill from {co} to {other}. Professional services rendered. Total due: {f['total']}. "
        f"Payment due by {f['due']}. Please remit payment by wire transfer.",
        f"{co} - Statement of charges for {other}. Reference {f['number']}. Amount payable: {f['total']}. "
        f"Payment is expected no later than {f['due']}. Late payments incur a 1.5% monthly fee.",
        f"Billing notice {f['number']}: {other} owes {co} {f['total']} for consulting work completed this quarter. "
        f"Kindly settle the balance on or before {f['due']}.",
    ]
    qs = [Question(f"How much does {other} owe {co}?", "", f["total"]),
          Question(f"When is payment due on the invoice from {co} to {other}?", "", f["due"])]
    return texts[style], f, qs


def _employment(rng, style, co, other):
    person = _person(rng)
    f = {"employer": co, "employee": person, "role": rng.choice(ROLES), "salary": _money(rng, 65000, 180000), "start": _date(rng)}
    texts = [
        f"EMPLOYMENT AGREEMENT. {co} hereby employs {person} as {f['role']}. Base salary: {f['salary']} per year. "
        f"Start date: {f['start']}. Employment is at-will and may be terminated by either party with notice.",
        f"Offer Letter. Dear {person}, {co} is pleased to offer you the position of {f['role']} with an annual salary of "
        f"{f['salary']}. Your first day will be {f['start']}. This offer is contingent on a background check.",
        f"{person} will join {co} as {f['role']} beginning {f['start']}, earning {f['salary']} annually plus standard "
        f"benefits. The employee agrees to the company's confidentiality and conduct policies.",
    ]
    qs = [Question(f"What is {person}'s salary at {co}?", "", f["salary"]),
          Question(f"What role will {person} have at {co}?", "", f["role"])]
    return texts[style], f, qs


def _memo(rng, style, co, other):
    author = _person(rng)
    f = {"client": co, "author": author, "risk": rng.choice(["low", "moderate", "high"]), "deadline": _date(rng)}
    texts = [
        f"MEMORANDUM. To: Litigation Team. From: {author}. Re: {co} v. {other}. We assess the litigation risk as {f['risk']}. "
        f"The response to the complaint must be filed by {f['deadline']}. Recommend preserving all relevant documents.",
        f"Legal analysis prepared by {author} regarding the dispute between {co} and {other}. Based on precedent, exposure "
        f"is {f['risk']}. Key filing deadline: {f['deadline']}. Next steps include witness interviews.",
        f"Privileged and confidential. {author} reviewed the claims {other} raised against {co}. Overall risk: {f['risk']}. "
        f"Counsel must respond before {f['deadline']}.",
    ]
    qs = [Question(f"Who wrote the memo about {co} and {other}?", "", author),
          Question(f"What is the assessed risk in the dispute between {co} and {other}?", "", f["risk"])]
    return texts[style], f, qs


def _minutes(rng, style, co, other):
    chair = _person(rng)
    f = {"company": co, "chair": chair, "topic": rng.choice(TOPICS), "next": _date(rng), "city": rng.choice(CITIES)}
    texts = [
        f"MEETING MINUTES - {co}. Chaired by {chair} in {f['city']}. Agenda item: {f['topic']}. The team agreed on next "
        f"steps and action owners. Next meeting scheduled for {f['next']}.",
        f"Notes from the {co} working session in {f['city']}. Facilitator: {chair}. Discussion focused on {f['topic']}. "
        f"Follow-up session set for {f['next']}.",
        f"Summary of discussion: {chair} led the {co} team meeting ({f['city']} office) on {f['topic']}. Decisions were "
        f"recorded and the group will reconvene on {f['next']}.",
    ]
    qs = [Question(f"Who chaired the {co} meeting about {f['topic']}?", "", chair),
          Question(f"When is the next {co} meeting about {f['topic']}?", "", f["next"])]
    return texts[style], f, qs


BUILDERS = {"nda": _nda, "invoice": _invoice, "employment_agreement": _employment,
            "legal_memo": _memo, "meeting_minutes": _minutes}
TRAIN_STYLES, TEST_STYLES = (0, 1), (2,)


def make_corpus(per_style: int = 20, seed: int = 7) -> tuple[list[Document], list[Question]]:
    """Return (documents, questions). Each document gets 2 questions with known answers."""
    rng = random.Random(seed)
    used: set[str] = set()
    docs, questions = [], []
    for label, build in BUILDERS.items():
        for style in (0, 1, 2):
            for _ in range(per_style):
                co, other = _company(rng, used), _company(rng, used)
                text, facts, qs = build(rng, style, co, other)
                doc_id = f"{label}-{len(docs):04d}"
                docs.append(Document(doc_id, label, style, text, facts))
                for q in qs:
                    q.doc_id = doc_id
                    questions.append(q)
    return docs, questions


def train_test_split(docs: list[Document]):
    """Split by writing style so the test set is unseen phrasing, not just unseen documents."""
    train = [d for d in docs if d.style in TRAIN_STYLES]
    test = [d for d in docs if d.style in TEST_STYLES]
    return train, test
