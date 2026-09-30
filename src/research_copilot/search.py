from .index import DATA_DIR, get_default_index
from .ingest import load_corpus
from .models import Document


def load_documents() -> list[Document]:
    documents, _ = load_corpus(DATA_DIR)
    return documents


def search(query: str, limit: int = 5) -> list[Document]:
    """Document-level search over the shared chunk index."""
    return get_default_index().search_documents(query, limit=limit)
