from types import SimpleNamespace

import pytest

from research_copilot.llm import ClaudeLLM, GeminiLLM, LLMRefusal, StubLLM


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def fake_client(response):
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_claude_llm_returns_text_blocks():
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text='{"answer": '),
            SimpleNamespace(type="text", text='"x"}'),
        ],
    )
    client, messages = fake_client(response)

    assert ClaudeLLM(client=client).generate("prompt") == '{"answer": "x"}'
    assert messages.kwargs["model"] == "claude-opus-5"
    assert messages.kwargs["messages"] == [{"role": "user", "content": "prompt"}]


def test_claude_llm_raises_on_refusal():
    client, _ = fake_client(SimpleNamespace(stop_reason="refusal", content=[]))

    with pytest.raises(LLMRefusal):
        ClaudeLLM(client=client).generate("prompt")


class FakeGeminiModels:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_gemini_llm_requests_json_and_returns_text():
    models = FakeGeminiModels(SimpleNamespace(text='{"answer": "x"}'))
    llm = GeminiLLM(model="some-gemini-model", client=SimpleNamespace(models=models))

    assert llm.generate("prompt") == '{"answer": "x"}'
    assert models.kwargs["model"] == "some-gemini-model"
    assert models.kwargs["contents"] == "prompt"
    assert models.kwargs["config"].response_mime_type == "application/json"


def test_gemini_llm_raises_when_blocked():
    blocked = SimpleNamespace(
        text=None, prompt_feedback=SimpleNamespace(block_reason="SAFETY"), candidates=[]
    )
    llm = GeminiLLM(model="m", client=SimpleNamespace(models=FakeGeminiModels(blocked)))

    with pytest.raises(LLMRefusal, match="SAFETY"):
        llm.generate("prompt")


def test_provider_selection(monkeypatch):
    from research_copilot import api

    for var in ("LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert isinstance(api.build_llm(), StubLLM)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with pytest.raises(RuntimeError, match="GEMINI_MODEL"):
        api.build_llm()

    monkeypatch.setenv("GEMINI_MODEL", "some-gemini-model")
    assert isinstance(api.build_llm(), GeminiLLM)
