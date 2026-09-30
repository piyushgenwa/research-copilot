import json

import pytest

from research_copilot.assistant import InvalidLLMOutput, answer_question, parse_llm_output
from research_copilot.llm import LLM, FakeLLM

CHECKOUT_QUESTION = "Why do users abandon checkout?"


def reply(answer="Participants abandon checkout when re-entering card details.",
          citations=("checkout-study#1",), coverage="full"):
    return json.dumps({"answer": answer, "citations": list(citations), "coverage": coverage})


class RecordingLLM(LLM):
    def __init__(self, response=None):
        self.prompts = []
        self.response = response or reply()

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_assistant_passes_research_context_to_llm():
    llm = RecordingLLM()

    result = answer_question(CHECKOUT_QUESTION, llm)

    assert result.status == "answered"
    assert 'id="checkout-study#1"' in llm.prompts[0]
    assert "re-enter" in llm.prompts[0].lower()


def test_citations_map_to_source_documents():
    llm = FakeLLM([reply(citations=["checkout-study#1", "checkout-redesign-evaluation#3"])])

    result = answer_question(CHECKOUT_QUESTION, llm)

    assert [s.doc_id for s in result.sources] == ["checkout-study", "checkout-redesign-evaluation"]
    assert result.sources[0].speaker == "P17"
    assert "re-enter my card details" in result.sources[0].quote


def test_assistant_handles_no_search_results():
    llm = RecordingLLM()

    result = answer_question("quantum teleportation", llm)

    assert result.status == "insufficient_evidence"
    assert result.sources == []
    assert llm.prompts == []


def test_llm_reports_insufficient_evidence():
    llm = FakeLLM([reply(answer="The excerpts don't cover desktop checkout.",
                         citations=[], coverage="none")])

    result = answer_question("Is desktop checkout frustrating?", llm)

    assert result.status == "insufficient_evidence"
    assert "desktop" in result.answer
    assert result.sources == []


def test_partial_coverage_is_reported():
    llm = FakeLLM([reply(coverage="partial")])

    result = answer_question(CHECKOUT_QUESTION, llm)

    assert result.status == "partial"
    assert result.sources


def test_hallucinated_citations_are_dropped():
    llm = FakeLLM([reply(citations=["checkout-study#1", "made-up-study#9"])])

    result = answer_question(CHECKOUT_QUESTION, llm)

    assert [s.chunk_id for s in result.sources] == ["checkout-study#1"]


def test_retries_once_after_malformed_output():
    llm = FakeLLM(["Sure! Here's what I found: checkout is bad", reply()])

    result = answer_question(CHECKOUT_QUESTION, llm)

    assert result.status == "answered"
    assert result.sources[0].chunk_id == "checkout-study#1"


@pytest.mark.parametrize(
    "bad_outputs",
    [
        ["not json", '{"answer": "truncated'],
        [reply(citations=["made-up#1"]), reply(citations=[])],
        ['{"answer": "x", "citations": "checkout-study#1", "coverage": "full"}'] * 2,
        [reply(coverage="maybe")] * 2,
        [],  # LLM raises on every call
    ],
)
def test_falls_back_to_evidence_when_output_stays_invalid(bad_outputs):
    result = answer_question(CHECKOUT_QUESTION, FakeLLM(bad_outputs))

    assert result.status == "evidence_only"
    assert result.sources
    assert all(source.doc_id for source in result.sources)


def test_code_fenced_json_is_accepted():
    parsed = parse_llm_output("```json\n" + reply() + "\n```", {"checkout-study#1"})

    assert parsed.coverage == "full"
    assert parsed.citations == ["checkout-study#1"]


def test_answer_without_valid_citations_is_rejected():
    with pytest.raises(InvalidLLMOutput):
        parse_llm_output(reply(citations=["other#1"]), {"checkout-study#1"})


def test_aggregate_questions_state_their_limitation():
    llm = RecordingLLM(reply(citations=["checkout-study#1"]))

    result = answer_question("How many participants were frustrated by checkout?", llm)

    assert any("not an exhaustive scan" in note for note in result.limitations)
    assert "count distinct participants" in llm.prompts[0]


def test_catalog_questions_do_not_call_the_llm():
    llm = RecordingLLM()

    result = answer_question("Who participated in the Loyalty Program Research?", llm)

    assert llm.prompts == []
    assert result.status == "answered"
    assert "P54" in result.answer
    assert [s.doc_id for s in result.sources] == ["loyalty-program-study"]


def test_participant_scope_limits_evidence():
    llm = RecordingLLM(reply(citations=["search-filters-diary-study#2"]))

    answer_question("What did P63 say about filters?", llm)

    assert 'speaker="P63"' in llm.prompts[0]
    assert 'speaker="P64"' not in llm.prompts[0]
