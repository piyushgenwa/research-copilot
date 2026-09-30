import os
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .assistant import answer_question
from .classifier import DEFAULT_MODEL, Classifier, JevClassifier
from .index import get_default_index
from .llm import LLM, ClaudeLLM, GeminiLLM, StubLLM


def build_llm() -> LLM:
    provider = os.environ.get("LLM_PROVIDER") or (
        "gemini" if os.environ.get("GEMINI_API_KEY") else "stub"
    )
    if provider == "gemini":
        model = os.environ.get("GEMINI_MODEL")
        if not model:
            raise RuntimeError("GEMINI_MODEL must be set to use the Gemini provider.")
        return GeminiLLM(model=model, api_key=os.environ.get("GEMINI_API_KEY"))
    if provider == "claude":
        return ClaudeLLM(model=os.environ.get("CLAUDE_MODEL", "claude-opus-5"))
    return StubLLM()


def build_classifier() -> Classifier:
    if not os.environ.get("TYPESAFE_API_KEY"):
        return Classifier()
    return JevClassifier(model=os.environ.get("TYPESAFE_DEFAULT_MODEL", DEFAULT_MODEL))


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_default_index()  # build the index once, before the first request
    yield


app = FastAPI(title="Research Copilot", lifespan=lifespan)
llm = build_llm()
classifier = build_classifier()


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: QuestionRequest):
    return asdict(answer_question(request.question, llm, classifier=classifier))
