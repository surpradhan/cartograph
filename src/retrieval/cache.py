from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

if TYPE_CHECKING:
    import numpy as np  # noqa: F811
    from sentence_transformers import SentenceTransformer  # noqa: F811

# MiniLM output dimensionality
_EMBEDDING_DIM = 384


class SourceCache:
    """
    FAISS-backed cache for deduplicating web search results.

    Uses cosine similarity on sentence embeddings to detect near-duplicate
    snippets, preventing redundant sources from polluting the research context.

    Falls back to URL-based deduplication when faiss / sentence-transformers
    are not installed (e.g. during development without the full ML stack).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.92,
    ):
        self.threshold = threshold
        self.stored_urls: list[str] = []
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None  # type: ignore[name-defined]

        if _FAISS_AVAILABLE:
            self.model = SentenceTransformer(model_name)
            self.index = faiss.IndexFlatIP(_EMBEDDING_DIM)
        else:
            logger.warning(
                "faiss / sentence-transformers not available; "
                "falling back to URL-only deduplication"
            )

    def _embed(self, text: str):
        return self.model.encode([text], normalize_embeddings=True)

    def is_duplicate(self, text: str, url: str) -> bool:
        """Return True if a semantically similar source is already cached."""
        if not _FAISS_AVAILABLE or self.index is None:
            return url in self.stored_urls

        if self.index.ntotal == 0:
            return False
        embedding = self._embed(text)
        scores, _ = self.index.search(np.array(embedding), 1)
        return float(scores[0][0]) > self.threshold

    def add(self, text: str, url: str) -> None:
        """Add a source to the cache."""
        if _FAISS_AVAILABLE and self.index is not None and self.model is not None:
            embedding = self._embed(text)
            self.index.add(np.array(embedding))
        self.stored_urls.append(url)

    def __len__(self) -> int:
        if _FAISS_AVAILABLE and self.index is not None:
            return self.index.ntotal
        return len(self.stored_urls)
