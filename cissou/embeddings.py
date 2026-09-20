"""Query/document embeddings via a small local multilingual model.

This runs through ``fastembed`` (ONNX Runtime) rather than
``sentence-transformers`` + PyTorch. The difference matters operationally: a
PyTorch + transformers install pulls in CUDA libraries and a much larger
runtime footprint, which risks exceeding a 512MB Render free-tier instance
before the model weights or Flask itself even load. fastembed's ONNX path
avoids that dependency tree entirely while still running fully locally
(no API calls, $0 marginal cost).

Model choice: ``paraphrase-multilingual-MiniLM-L12-v2`` is the smallest
option in fastembed's catalog (0.22GB) that actually covers French, English
*and* Arabic — the smaller English-only models (bge-small, all-MiniLM-L6,
arctic-xs) don't meet ESI's trilingual requirement.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384

_lock = threading.Lock()
_model = None


def _get_model():
    """Lazily load the model once per process, on first use.

    Loading happens off the request path in practice: the vector store's
    ``build_or_load`` calls this once at app startup (or is skipped entirely
    on a cache hit), never per-query for the corpus side, and only for a
    single short string on the query side.
    """
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from fastembed import TextEmbedding

                log.info("loading embedding model %s", MODEL_NAME)
                _model = TextEmbedding(model_name=MODEL_NAME , threads=1)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts, L2-normalised so dot product == cosine similarity."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")
    model = _get_model()
    vectors = np.array(list(model.embed(texts)), dtype="float32")
    return _normalize(vectors)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
