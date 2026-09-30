from research_copilot.index import get_default_index
from research_copilot.search import search


def test_search_finds_relevant_document():
    results = search("re-enter card details")
    assert results
    assert results[0].id == "checkout-study"


def test_search_returns_empty_for_unknown_topic():
    results = search("quantum teleportation")
    assert results == []


def test_question_framing_words_do_not_match():
    assert search("what do people think about the weather") == []


def test_word_forms_match_via_stemming():
    hits = get_default_index().search("frustrating")
    assert "checkout-study#2" in {hit.chunk.id for hit in hits}  # "frustration"


def test_filters_restrict_results():
    hits = get_default_index().search("filter", speakers={"P63"})
    assert hits
    assert {hit.chunk.speaker for hit in hits} == {"P63"}


def test_per_document_cap_spreads_results():
    hits = get_default_index().search("checkout payment card", k=10, per_doc_cap=2)
    doc_counts = {}
    for hit in hits:
        doc_counts[hit.chunk.doc_id] = doc_counts.get(hit.chunk.doc_id, 0) + 1
    assert max(doc_counts.values()) <= 2
