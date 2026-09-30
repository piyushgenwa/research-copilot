from fastapi.testclient import TestClient

from research_copilot import api


client = TestClient(api.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
