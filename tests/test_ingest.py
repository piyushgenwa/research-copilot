from research_copilot.ingest import parse_document

SAMPLE = """# Sample Study

Study: Sample Study Name
Date: 2026-01-01
Participants: P01, P02

Participant P01:
"First quote."

Participant P02 (Day 3):
"Diary entry
over two lines."

Respondent R7:
"Survey answer."

Researcher observation:
An observation.

Key finding:
A note.
"""


def test_parses_header_fields():
    document, _ = parse_document("sample", SAMPLE)

    assert document.title == "Sample Study"
    assert document.study == "Sample Study Name"
    assert document.date == "2026-01-01"
    assert document.participants == ("P01", "P02")


def test_splits_into_citable_chunks():
    _, chunks = parse_document("sample", SAMPLE)

    assert [(c.id, c.kind, c.speaker) for c in chunks] == [
        ("sample#1", "quote", "P01"),
        ("sample#2", "quote", "P02"),
        ("sample#3", "quote", "R7"),
        ("sample#4", "observation", ""),
        ("sample#5", "note", ""),
    ]
    assert chunks[1].text == '"Diary entry over two lines."'
    assert chunks[1].label == "Participant P02 (Day 3)"


def test_missing_header_fields_default_sensibly():
    document, chunks = parse_document("bare", "# Bare\n\nResearcher observation:\nText.")

    assert document.study == "Bare"
    assert document.participants == ()
    assert len(chunks) == 1


TRANSCRIPT = """# Interview

Moderator:
What did you check before paying?

Participant P01:
"The total."

Participant P02:
"Nothing."

Researcher observation:
Most checked the total.

Participant P03:
"Unprompted comment."
"""


def test_moderator_questions_become_context_not_chunks():
    _, chunks = parse_document("t", TRANSCRIPT)

    assert [(c.id, c.speaker, c.context) for c in chunks] == [
        ("t#1", "P01", "What did you check before paying?"),
        ("t#2", "P02", "What did you check before paying?"),
        ("t#3", "", ""),
        ("t#4", "P03", ""),  # an observation ends the moderator's question
    ]
