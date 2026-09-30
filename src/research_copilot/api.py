import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .assistant import answer_question
from .classifier import DEFAULT_MODEL, Classifier, JevClassifier
from .index import DATA_DIR, get_default_index
from .llm import LLM, ClaudeLLM, GeminiLLM, StubLLM
from .models import Document


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


INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: QuestionRequest):
    return asdict(answer_question(request.question, llm, classifier=classifier))


@app.get("/api/studies")
def list_studies():
    index = get_default_index()
    documents = sorted(index.documents.values(), key=lambda d: (d.date, d.id), reverse=True)
    return [_summary(document) for document in documents]


@app.get("/api/studies/{doc_id}")
def get_study(doc_id: str):
    document = _document_or_404(doc_id)
    chunks = get_default_index().chunks_by_doc.get(doc_id, [])
    return {
        **_summary(document),
        "excerpts": [
            {
                "chunk_id": chunk.id,
                "kind": chunk.kind,
                "speaker": chunk.speaker,
                "label": chunk.label,
                "context": chunk.context,
                "text": chunk.text,
            }
            for chunk in chunks
        ],
    }


@app.get("/api/studies/{doc_id}/markdown", response_class=PlainTextResponse)
def get_study_markdown(doc_id: str):
    # The ID is checked against the index before touching the filesystem.
    document = _document_or_404(doc_id)
    return PlainTextResponse(
        (DATA_DIR / f"{document.id}.md").read_text(encoding="utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'inline; filename="{document.id}.md"'},
    )


def _document_or_404(doc_id: str) -> Document:
    document = get_default_index().documents.get(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Study not found")
    return document


def _summary(document: Document) -> dict:
    return {
        "id": document.id,
        "title": document.title,
        "study": document.study,
        "date": document.date,
        "method": document.method,
        "participants": list(document.participants),
        "excerpt_count": len(get_default_index().chunks_by_doc.get(document.id, [])),
    }
