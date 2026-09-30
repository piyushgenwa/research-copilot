import pytest

from research_copilot.index import get_default_index
from research_copilot.query import AGGREGATE, CATALOG, RETRIEVE, plan_query


@pytest.mark.parametrize(
    "question, route",
    [
        ("Did people find checkout frustrating?", RETRIEVE),
        ("Why do users abandon checkout?", RETRIEVE),
        ("How many participants mentioned repeating information?", AGGREGATE),
        ("What are the most common complaints overall?", AGGREGATE),
        ("What themes come up across studies?", AGGREGATE),
        ("Which studies covered payments?", CATALOG),
        ("Who participated in the pricing study?", CATALOG),
    ],
)
def test_routes(question, route):
    assert plan_query(question, get_default_index()).route == route


def test_participant_ids_become_speaker_filter():
    plan = plan_query("What did p63 and R101 say?", get_default_index())
    assert plan.speakers == {"P63", "R101"}


def test_exact_study_name_becomes_document_filter():
    plan = plan_query("In Delivery Tracking Research, what went wrong?", get_default_index())
    assert plan.doc_ids == {"delivery-tracking-study"}


def test_vague_study_reference_does_not_filter():
    plan = plan_query("What did the pricing study find?", get_default_index())
    assert plan.doc_ids is None
