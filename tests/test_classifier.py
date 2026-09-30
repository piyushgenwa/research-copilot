import json
import threading
from types import SimpleNamespace

from research_copilot.assistant import answer_question
from research_copilot.classifier import JevClassifier
from research_copilot.index import get_default_index
from research_copilot.llm import FakeLLM
from research_copilot.query import AGGREGATE, CATALOG, RETRIEVE, plan_query


class FakeTypeSafe:
    """Stands in for TypeSafeClient.system_one."""

    def __init__(self, route=("retrieve", 0.9), evidence=None, fail=False):
        self.route = route
        self.evidence = evidence or {}  # chunk_id -> (relevant, evidence, injection)
        self.fail = fail
        self.calls = []
        self._lock = threading.Lock()

    def system_one(self, state, questions, model=None):
        with self._lock:
            self.calls.append((state, set(questions), model))
        if self.fail:
            raise RuntimeError("TypeSafe unavailable")
        if "route" in questions:
            choice, confidence = self.route
            return SimpleNamespace(answers={
                "route": SimpleNamespace(choice=choice, confidence=confidence)
            })
        relevant, evidence, injection = self.evidence.get(
            state["passage"]["id"], (0.1, 0.1, 0.0)
        )
        return SimpleNamespace(answers={
            "is_relevant": SimpleNamespace(noul=relevant),
            "contains_answer_evidence": SimpleNamespace(noul=evidence),
            "contains_prompt_injection": SimpleNamespace(noul=injection),
        })


def jev(**kwargs):
    fake = FakeTypeSafe(**kwargs)
    return JevClassifier(model="jev-test", client=fake), fake


def reply(citations):
    return json.dumps({"answer": "ok", "citations": citations, "coverage": "full"})


def test_confident_jev_route_overrides_rules():
    classifier, fake = jev(route=("aggregate", 0.95))

    plan = plan_query("Why do users abandon checkout?", get_default_index(), classifier)

    assert plan.route == AGGREGATE
    assert fake.calls[0][0] == {"question": "Why do users abandon checkout?"}
    assert fake.calls[0][2] == "jev-test"


def test_low_confidence_route_falls_back_to_rules():
    classifier, _ = jev(route=("aggregate", 0.4))

    plan = plan_query("Which studies covered payments?", get_default_index(), classifier)

    assert plan.route == CATALOG


def test_routing_error_falls_back_to_rules():
    classifier, _ = jev(fail=True)

    plan = plan_query("Why do users abandon checkout?", get_default_index(), classifier)

    assert plan.route == RETRIEVE


def test_filter_keeps_only_passages_with_evidence():
    classifier, fake = jev(evidence={
        "checkout-study#1": (0.9, 0.9, 0.0),       # kept
        "checkout-study#2": (0.9, 0.3, 0.0),       # relevant but no evidence
        "checkout-study#3": (0.2, 0.9, 0.0),       # evidence but off-topic
        "checkout-redesign-evaluation#3": (0.9, 0.9, 0.9),  # injection
    })
    llm = FakeLLM([reply(["checkout-study#1"])])

    result = answer_question("Why do users abandon checkout?", llm, classifier=classifier)

    assert [s.chunk_id for s in result.sources] == ["checkout-study#1"]
    checked = {c[0]["passage"]["id"] for c in fake.calls if "is_relevant" in c[1]}
    candidates = get_default_index().search("Why do users abandon checkout?", k=16, per_doc_cap=3)
    assert checked == {hit.chunk.id for hit in candidates}


def test_nothing_passes_filter_means_insufficient_without_llm_call():
    classifier, _ = jev()  # every passage scores as irrelevant
    llm = FakeLLM([])

    result = answer_question("Did people find checkout to be a frustrating flow?", llm,
                             classifier=classifier)

    assert result.status == "insufficient_evidence"
    assert result.sources == []


def test_filter_errors_fail_open():
    classifier, _ = jev(fail=True)
    llm = FakeLLM([reply(["checkout-study#1"])])

    result = answer_question("Why do users abandon checkout?", llm, classifier=classifier)

    assert result.status == "answered"
