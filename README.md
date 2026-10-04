# Research Copilot — Interview Starter

## Context

This repository is a deliberately small prototype of a conversational AI system for UX researchers.

Researchers have a collection of interview transcripts, study notes, and participant feedback. The goal is to let a researcher ask questions about the corpus and receive answers grounded in the underlying evidence.

## Running

Requires Python 3.11+.

```bash
python -m pytest
```

To run the API:

```bash
uvicorn research_copilot.api:app --reload
```

Answers are generated with Gemini. Set a key and model (without a key, `/ask` returns the retrieved evidence without a synthesized answer):

```bash
pip install -e .
export GEMINI_API_KEY=...
export GEMINI_MODEL=...        # required, no default
# optional: LLM_PROVIDER=gemini|claude|stub (Claude needs '.[claude]' and ANTHROPIC_API_KEY)
```

Classification steps use TypeSafe's Jev model when a key is set (otherwise rules and keyword scores are used):

```bash
export TYPESAFE_API_KEY=...
export TYPESAFE_DEFAULT_MODEL=jev-latest   # optional
```

- **Routing**: a Choice between `catalog`, `aggregate` and `retrieve`. Below 0.6 confidence, or on error, the rule-based router decides.
- **Evidence filter**: each retrieved excerpt gets three Nouls (relevant, contains answer evidence, prompt injection) and is kept only if relevant ≥ 0.45, evidence > 0.55 and injection ≤ 0.70. If nothing passes, the answer is `insufficient_evidence` without calling the LLM. On error an excerpt is kept.

## Web app

`/` serves a single-page app:

- **Ask**: a chat-style view with a collapsible chat-history sidebar and a "New chat" button. History is stored in the browser (localStorage); the backend stays stateless and answers each question independently. Answers show a status, limitations and collapsible sources, and each source links to its place in the transcript.
- **Library**: every study as a card (method, date, participants, excerpts), filterable by text, participant ID or method. Opening a study shows the transcript with moderator questions, participant quotes and researcher observations, plus a link to the raw markdown.

Library endpoints: `GET /api/studies`, `GET /api/studies/{id}`, `GET /api/studies/{id}/markdown`.

## Data format

Each file in `data/` is one study: a `# Title`, optional `Study:`, `Date:`, `Method:`, `Participants:` fields, then blocks headed `Participant P12:`, `Respondent R3 (Score 9):`, `Researcher observation:` or `Moderator:`. Moderator questions are not indexed as evidence; they are attached as context to the answers that follow.

## How it works

Single question in, single answer out (`POST /ask`).

1. **Ingest** (`ingest.py`): each markdown file becomes a `Document` plus one `Chunk` per participant quote or researcher observation. Chunk IDs are `<doc_id>#<n>`.
2. **Index** (`index.py`): in-memory BM25 over chunks, built once at startup. `scripts/benchmark.py` measures ~1.5 ms per query at 5,000 documents.
3. **Plan** (`query.py`): rule-based route. `catalog` questions ("which studies…", "who participated…") are answered from metadata without the LLM. `aggregate` questions ("how many", "themes", "overall") retrieve more and carry a stated limitation. Participant IDs and exact study names narrow the search.
4. **Answer** (`assistant.py`): the LLM must return `{"answer", "citations", "coverage"}`. Citations not in the retrieved set are dropped. An answer with no valid citation is rejected. One retry, then fall back to returning the evidence.

Response shape:

```json
{"status": "answered | partial | insufficient_evidence | evidence_only",
 "answer": "...",
 "sources": [{"chunk_id", "doc_id", "title", "kind", "speaker", "quote"}],
 "limitations": ["..."]}
```

Known limits: keyword retrieval misses paraphrases ("again"/"twice" vs "repeat"), and counts come from top-k excerpts, not a full scan.

## Original starter behavior

The prototype exposes a basic search function and an LLM abstraction. The assistant flow is intentionally incomplete.

### Important constraint

When the assistant eventually returns an answer, source document IDs must be preserved so that answers can reference the underlying evidence.

Do not assume that every question can be answered from the corpus. The system should have a clear behavior when there is insufficient evidence.

## Interview task

Extend the prototype so a researcher can ask a question and receive:

1. A useful answer grounded in the available research.
2. References to the source documents used.
3. A sensible response when the available research is insufficient.

You may use a coding agent as a pair-programming partner. You remain responsible for understanding, reviewing, testing, and making the final engineering decisions.
