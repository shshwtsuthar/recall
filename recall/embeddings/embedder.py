"""
Embedding module — converts text to vectors using sentence-transformers.

The embedder is a thin wrapper so the model can be swapped later
(e.g., to a larger model or an API-based one) without changing
any calling code.

Usage:
    from recall.embeddings.embedder import Embedder

    embedder = Embedder()  # loads model on first call
    vector = embedder.embed("Fix the auth bypass in utils.py")
    vectors = embedder.embed_batch(["text1", "text2", "text3"])
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("recall.embedder")

_model_cache: dict[str, Any] = {}


def _get_model(model_name: str):
    """Lazy-load the sentence-transformers model (cached)."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _get_model(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        """Return the embedding dimension for the loaded model."""
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a list of floats."""
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed multiple texts. Returns a list of vectors."""
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10,
        )
        return vectors.tolist()
