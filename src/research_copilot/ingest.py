"""Parse research markdown files into documents and citable chunks.

Expected shape (header fields are optional apart from the title):

    # Title
    Study: ...
    Date: ...
    Participants: P12, P17

    Participant P17:
    "quote"

    Researcher observation:
    text
"""

import re
from pathlib import Path

from .models import Chunk, Document

_BLOCK_HEADER = re.compile(r"^([A-Za-z][A-Za-z ]*?)(?:\s+([A-Z]\d+))?(?:\s*\(([^)]*)\))?:$")
_FIELD = re.compile(r"^([A-Za-z]+):\s+(.+)$")
_QUOTE_LABELS = {"participant", "respondent"}
_MODERATOR_LABELS = {"moderator", "interviewer", "researcher"}


def parse_document(doc_id: str, text: str) -> tuple[Document, list[Chunk]]:
    lines = text.splitlines()
    title = lines[0].removeprefix("# ").strip() if lines else doc_id
    body = lines[1:]
    content = "\n".join(body).strip()

    fields: dict[str, str] = {}
    blocks: list[tuple[str, str, str, list[str]]] = []  # (label, kind, speaker, lines)

    for raw in body:
        line = raw.strip()
        if not line:
            continue
        header = _BLOCK_HEADER.match(line)
        if header:
            role, speaker, _ = header.groups()
            role_key = role.strip().lower()
            if role_key in _QUOTE_LABELS:
                kind = "quote"
            elif role_key in _MODERATOR_LABELS:
                kind = "moderator"
            elif "observation" in role_key:
                kind = "observation"
            else:
                kind = "note"
            blocks.append((line[:-1], kind, speaker or "", []))
            continue
        field = _FIELD.match(line)
        if field and not blocks:
            fields[field.group(1).lower()] = field.group(2).strip()
            continue
        if blocks:
            blocks[-1][3].append(line)
        # Free text before the first block with no "Key: value" shape is ignored.

    participants = tuple(
        p.strip() for p in fields.get("participants", "").split(",") if p.strip()
    )
    document = Document(
        id=doc_id,
        title=title,
        content=content,
        study=fields.get("study", title),
        date=fields.get("date", ""),
        method=fields.get("method", ""),
        participants=participants,
    )

    chunks: list[Chunk] = []
    question = ""
    for label, kind, speaker, block_lines in blocks:
        if not block_lines:
            continue
        text = " ".join(block_lines)
        # Moderator turns are not evidence; they give context to the answers
        # that follow, until the next moderator turn or observation.
        if kind == "moderator":
            question = text
            continue
        if kind != "quote":
            question = ""
        chunks.append(
            Chunk(
                id=f"{doc_id}#{len(chunks) + 1}",
                doc_id=doc_id,
                title=title,
                kind=kind,
                speaker=speaker,
                label=label,
                text=text,
                context=question,
            )
        )
    return document, chunks


def load_corpus(data_dir: Path) -> tuple[list[Document], list[Chunk]]:
    documents: list[Document] = []
    chunks: list[Chunk] = []
    for path in sorted(data_dir.glob("*.md")):
        document, doc_chunks = parse_document(path.stem, path.read_text(encoding="utf-8"))
        documents.append(document)
        chunks.extend(doc_chunks)
    return documents, chunks
