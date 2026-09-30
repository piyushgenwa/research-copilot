from fastapi.testclient import TestClient

from research_copilot import api


client = TestClient(api.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home_serves_the_question_page():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="composer"' in response.text


def test_ask_without_llm_returns_evidence(monkeypatch):
    monkeypatch.setattr(api, "llm", api.StubLLM())

    response = client.post("/ask", json={"question": "Why do users abandon checkout?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "evidence_only"
    assert body["sources"][0]["doc_id"]
    assert set(body) == {"status", "answer", "sources", "limitations"}


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_list_studies():
    studies = client.get("/api/studies").json()

    assert len(studies) >= 40
    first = studies[0]
    assert {"id", "title", "date", "method", "participants", "excerpt_count"} <= set(first)
    dates = [s["date"] for s in studies if s["date"]]
    assert dates == sorted(dates, reverse=True)


def test_get_study_returns_excerpts_with_citation_ids():
    study = client.get("/api/studies/checkout-study").json()

    assert study["title"] == "Mobile Checkout Usability"
    assert study["excerpts"][0]["chunk_id"] == "checkout-study#1"
    assert study["excerpts"][0]["speaker"] == "P17"


def test_transcript_excerpts_carry_moderator_questions():
    study = client.get("/api/studies/guest-checkout-interviews").json()

    first = study["excerpts"][0]
    assert first["speaker"] == "P77"
    assert first["context"].startswith("Tell me about the last time")


def test_study_markdown_download():
    response = client.get("/api/studies/checkout-study/markdown")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.text.startswith("# Mobile Checkout Usability")


def test_unknown_study_is_404():
    assert client.get("/api/studies/nope").status_code == 404
    assert client.get("/api/studies/..%2Fpyproject/markdown").status_code == 404
