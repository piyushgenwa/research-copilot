from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    content: str
    study: str = ""
    date: str = ""
    method: str = ""
    participants: tuple[str, ...] = ()


@dataclass(frozen=True)
class Chunk:
    """One citable unit of evidence: a participant quote or a researcher note."""

    id: str  # "<doc_id>#<n>"
    doc_id: str
    title: str
    kind: str  # "quote" | "observation" | "note"
    speaker: str  # participant ID for quotes, "" otherwise
    label: str  # the block header as written, e.g. "Participant P63 (Day 2)"
    text: str
    context: str = ""  # the moderator question this answers, in transcripts


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Source:
    chunk_id: str
    doc_id: str
    title: str
    kind: str
    speaker: str
    quote: str
    context: str = ""


@dataclass(frozen=True)
class Answer:
    # "answered" | "partial" | "insufficient_evidence" | "evidence_only"
    status: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
