"""Index build and query latency on a synthetic corpus.

Synthetic documents are made by recombining real chunks, so term statistics
stay realistic. Usage: python scripts/benchmark.py [num_documents]
"""

import random
import sys
import time

from research_copilot.index import DATA_DIR, ResearchIndex
from research_copilot.ingest import load_corpus
from research_copilot.models import Chunk, Document

QUERIES = [
    "Why do users abandon checkout?",
    "Did people find checkout frustrating?",
    "How many participants were annoyed by repeating information?",
    "What did participants say about notifications?",
    "What do users think about dark mode?",
]


def synthetic_corpus(num_documents: int, seed: int = 7):
    documents, chunks = load_corpus(DATA_DIR)
    rng = random.Random(seed)
    out_docs, out_chunks = [], []
    for n in range(num_documents):
        template = rng.choice(documents)
        doc_id = f"{template.id}-{n}"
        out_docs.append(Document(id=doc_id, title=template.title, content=""))
        for i, source in enumerate(rng.sample(chunks, 6), start=1):
            out_chunks.append(Chunk(
                id=f"{doc_id}#{i}", doc_id=doc_id, title=template.title, kind=source.kind,
                speaker=source.speaker, label=source.label, text=source.text,
            ))
    return out_docs, out_chunks


def main(num_documents: int) -> None:
    documents, chunks = synthetic_corpus(num_documents)
    start = time.perf_counter()
    index = ResearchIndex(documents, chunks)
    build = time.perf_counter() - start
    print(f"{num_documents} documents, {len(chunks)} chunks: index built in {build * 1000:.0f} ms")

    for query in QUERIES:
        start = time.perf_counter()
        for _ in range(20):
            hits = index.search(query, k=8, per_doc_cap=3)
        elapsed = (time.perf_counter() - start) / 20
        print(f"  {elapsed * 1000:6.1f} ms  {len(hits)} hits  {query}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000)
