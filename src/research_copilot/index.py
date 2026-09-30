"""In-memory BM25 index over research chunks.

Built once at startup. Scoring only touches the postings of the query terms,
so query cost grows with matching chunks rather than corpus size.
"""

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from .ingest import load_corpus
from .models import Chunk, Document, ScoredChunk
from .text import tokenize

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

K1 = 1.5
B = 0.75


class ResearchIndex:
    def __init__(self, documents: Iterable[Document], chunks: Iterable[Chunk]):
        self.documents: dict[str, Document] = {d.id: d for d in documents}
        self.chunks: list[Chunk] = list(chunks)
        self.chunks_by_id: dict[str, Chunk] = {c.id: c for c in self.chunks}
        self.chunks_by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in self.chunks:
            self.chunks_by_doc[chunk.doc_id].append(chunk)

        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []
        for position, chunk in enumerate(self.chunks):
            # Title and label are indexed with the text so a quote that never says
            # "checkout" still matches when it comes from a checkout study, and
            # participant IDs ("P17") are searchable.
            terms = tokenize(f"{chunk.title} {chunk.label} {chunk.context} {chunk.text}")
            self._lengths.append(len(terms))
            for term, count in Counter(terms).items():
                self._postings[term].append((position, count))

        total = len(self.chunks)
        self._avg_length = (sum(self._lengths) / total) if total else 0.0
        self._idf = {
            term: math.log(1 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    @classmethod
    def from_directory(cls, data_dir: Path) -> "ResearchIndex":
        return cls(*load_corpus(data_dir))

    def search(
        self,
        query: str,
        k: int = 8,
        doc_ids: set[str] | None = None,
        speakers: set[str] | None = None,
        per_doc_cap: int | None = None,
    ) -> list[ScoredChunk]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for position, tf in self._postings[term]:
                norm = K1 * (1 - B + B * self._lengths[position] / self._avg_length)
                scores[position] += idf * tf * (K1 + 1) / (tf + norm)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        results: list[ScoredChunk] = []
        per_doc: Counter[str] = Counter()
        for position, score in ranked:
            chunk = self.chunks[position]
            if doc_ids is not None and chunk.doc_id not in doc_ids:
                continue
            if speakers is not None and chunk.speaker not in speakers:
                continue
            if per_doc_cap is not None and per_doc[chunk.doc_id] >= per_doc_cap:
                continue
            per_doc[chunk.doc_id] += 1
            results.append(ScoredChunk(chunk=chunk, score=score))
            if len(results) >= k:
                break
        return results

    def search_documents(self, query: str, limit: int = 5) -> list[Document]:
        """Rank whole documents by their best-matching chunk."""
        best: dict[str, float] = {}
        for hit in self.search(query, k=len(self.chunks)):
            best.setdefault(hit.chunk.doc_id, hit.score)
        ranked = sorted(best, key=lambda doc_id: -best[doc_id])
        return [self.documents[doc_id] for doc_id in ranked[:limit]]


@lru_cache(maxsize=1)
def get_default_index() -> ResearchIndex:
    return ResearchIndex.from_directory(DATA_DIR)
