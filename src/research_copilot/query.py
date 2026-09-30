"""Query planning: which route to take and what scope to search.

The route comes from the classifier (Jev) when one is configured and
confident, otherwise from the rules below. Scope is always extracted by code.
Only distinctions that change behaviour are made here. Everything that is not
a catalog or aggregate question goes through normal top-k retrieval, and
"is this answerable?" is decided from the evidence, not from the question.
"""

import re
from dataclasses import dataclass

from .classifier import Classifier
from .index import ResearchIndex

RETRIEVE = "retrieve"
AGGREGATE = "aggregate"
CATALOG = "catalog"

_CATALOG_PATTERNS = [
    r"\bwhich (studies|research)\b",
    r"\bwhat (studies|research)\b",
    r"\blist (all |the )?(studies|research)\b",
    r"\bwho (participated|took part|was interviewed|were the participants)\b",
    r"\bhow many (studies|participants (were|took part|are) in)\b",
]
_AGGREGATE_PATTERNS = [
    r"\bhow many\b",
    r"\bhow often\b",
    r"\bpercent(age)?\b",
    r"\bproportion\b",
    r"\bmost (common|frequent)\b",
    r"\boverall\b",
    r"\bacross (all )?(the )?(studies|research)\b",
    r"\b(themes?|trends?|patterns?)\b",
]
_SPEAKER = re.compile(r"\b([PR]\d{2,3})\b", re.IGNORECASE)


@dataclass(frozen=True)
class QueryPlan:
    route: str
    doc_ids: frozenset[str] | None = None
    speakers: frozenset[str] | None = None


def plan_query(
    question: str, index: ResearchIndex, classifier: Classifier | None = None
) -> QueryPlan:
    lowered = question.lower()

    speakers = frozenset(match.upper() for match in _SPEAKER.findall(question)) or None

    # Only an exact study title or document ID narrows the search. Fuzzy
    # matching ("the pricing study") is left to ranking, because a wrong filter
    # silently hides the evidence.
    named = frozenset(
        doc.id
        for doc in index.documents.values()
        if doc.id in lowered or doc.title.lower() in lowered or doc.study.lower() in lowered
    ) or None

    decision = classifier.route(question) if classifier else None
    if decision is not None:
        route = decision.route
    elif any(re.search(pattern, lowered) for pattern in _CATALOG_PATTERNS):
        route = CATALOG
    elif any(re.search(pattern, lowered) for pattern in _AGGREGATE_PATTERNS):
        route = AGGREGATE
    else:
        route = RETRIEVE
    return QueryPlan(route=route, doc_ids=named, speakers=speakers)
