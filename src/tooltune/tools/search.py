"""Episode-local deterministic BM25 retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class LocalDocument:
    document_id: str
    text: str
    title: str = ""


class LocalSearch:
    """A closed-corpus search tool; it never accesses the network."""

    def __init__(self, documents: list[LocalDocument], *, max_chars: int = 3_000):
        if not documents:
            documents = [LocalDocument("empty", "No local evidence was supplied.")]
        self.documents = documents
        self.max_chars = max_chars
        corpus = [
            _tokens(f"{doc.title} {doc.text}") or ["__empty__"] for doc in documents
        ]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, *, top_k: int = 3) -> str:
        if not query.strip():
            return "SEARCH_ERROR: empty query"
        top_k = min(max(int(top_k), 1), 5, len(self.documents))
        query_tokens = _tokens(query) or ["__empty__"]
        scores = self._bm25.get_scores(query_tokens)
        order = sorted(range(len(self.documents)), key=lambda i: (-float(scores[i]), i))
        parts: list[str] = []
        used = 0
        for rank, index in enumerate(order[:top_k], start=1):
            doc = self.documents[index]
            heading = f"[{rank}] id={doc.document_id}"
            if doc.title:
                heading += f" title={doc.title}"
            chunk = f"{heading}\n{doc.text.strip()}"
            remaining = self.max_chars - used
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk[: max(0, remaining - 15)] + "...[truncated]"
            parts.append(chunk)
            used += len(chunk) + 2
        return "\n\n".join(parts)
