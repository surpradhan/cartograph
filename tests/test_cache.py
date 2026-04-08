"""Tests for SourceCache — covers both FAISS and URL-fallback modes."""

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.cache import SourceCache


# ── URL-fallback mode (no FAISS) ──────────────────────────────────────────────

@patch("src.retrieval.cache._FAISS_AVAILABLE", False)
def test_url_fallback_not_duplicate_when_empty():
    cache = SourceCache()
    assert cache.is_duplicate("some text", "https://new.com") is False


@patch("src.retrieval.cache._FAISS_AVAILABLE", False)
def test_url_fallback_detects_duplicate_url():
    cache = SourceCache()
    cache.add("text", "https://seen.com")
    assert cache.is_duplicate("different text", "https://seen.com") is True


@patch("src.retrieval.cache._FAISS_AVAILABLE", False)
def test_url_fallback_different_url_not_duplicate():
    cache = SourceCache()
    cache.add("text", "https://seen.com")
    assert cache.is_duplicate("text", "https://other.com") is False


@patch("src.retrieval.cache._FAISS_AVAILABLE", False)
def test_url_fallback_len():
    cache = SourceCache()
    assert len(cache) == 0
    cache.add("a", "https://a.com")
    cache.add("b", "https://b.com")
    assert len(cache) == 2


# ── FAISS mode ────────────────────────────────────────────────────────────────

def _make_faiss_cache():
    """Return a SourceCache with mocked FAISS and SentenceTransformer."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((1, 384), dtype="float32")

    mock_index = MagicMock()
    mock_index.ntotal = 0

    cache = SourceCache.__new__(SourceCache)
    cache.threshold = 0.92
    cache.stored_urls = []
    cache.model = mock_model
    cache.index = mock_index
    return cache


@patch("src.retrieval.cache._FAISS_AVAILABLE", True)
def test_faiss_mode_empty_index_not_duplicate():
    cache = _make_faiss_cache()
    cache.index.ntotal = 0
    assert cache.is_duplicate("text", "https://x.com") is False


@patch("src.retrieval.cache._FAISS_AVAILABLE", True)
def test_faiss_mode_high_similarity_is_duplicate():
    import numpy as np

    cache = _make_faiss_cache()
    cache.index.ntotal = 1
    cache.index.search.return_value = (np.array([[0.95]]), np.array([[0]]))

    assert cache.is_duplicate("near-duplicate text", "https://x.com") is True


@patch("src.retrieval.cache._FAISS_AVAILABLE", True)
def test_faiss_mode_low_similarity_not_duplicate():
    import numpy as np

    cache = _make_faiss_cache()
    cache.index.ntotal = 1
    cache.index.search.return_value = (np.array([[0.5]]), np.array([[0]]))

    assert cache.is_duplicate("unrelated text", "https://x.com") is False


@patch("src.retrieval.cache._FAISS_AVAILABLE", True)
def test_faiss_mode_add_updates_index_and_urls():
    cache = _make_faiss_cache()
    cache.add("text", "https://added.com")

    cache.index.add.assert_called_once()
    assert "https://added.com" in cache.stored_urls
