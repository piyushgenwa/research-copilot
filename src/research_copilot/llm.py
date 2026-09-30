class LLM:
    """Thin abstraction around a language model.

    The interview environment does not require real API credentials.
    Tests can replace this implementation with a deterministic fake.
    """

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Connect an LLM implementation here.")


class FakeLLM(LLM):
    """Deterministic LLM double for interviewer-controlled scenarios.

    Queue responses with `enqueue`; each `generate` call pops the next one.
    Used to inject malformed/incomplete outputs during the reliability stage
    regardless of the structured shape the candidate designs.
    """

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])

    def enqueue(self, response: str) -> None:
        self._responses.append(response)

    def generate(self, prompt: str) -> str:
        if not self._responses:
            raise RuntimeError("FakeLLM has no queued responses left.")
        return self._responses.pop(0)


class StubLLM(LLM):
    """Used when no provider is configured; the assistant falls back to evidence only."""

    def generate(self, prompt: str) -> str:
        return "LLM integration is not configured."


class LLMRefusal(RuntimeError):
    pass


class GeminiLLM(LLM):
    """Gemini via the google-genai SDK, returning JSON (the assistant's only use)."""

    def __init__(self, model: str, api_key: str | None = None, client=None):
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client
        self._model = model

    def generate(self, prompt: str) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        text = response.text
        if not text:
            # Blocked prompts and safety stops come back with no text.
            feedback = getattr(response, "prompt_feedback", None)
            reason = getattr(feedback, "block_reason", None) if feedback else None
            if reason is None and getattr(response, "candidates", None):
                reason = response.candidates[0].finish_reason
            raise LLMRefusal(f"Gemini returned no text (reason: {reason})")
        return text


class ClaudeLLM(LLM):
    """Claude via the Anthropic SDK. Credentials resolve from the environment."""

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 16000, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response = self._client.beta.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            # On a safety decline, the API re-runs the request on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise LLMRefusal("model declined to answer")
        return "".join(block.text for block in response.content if block.type == "text")
