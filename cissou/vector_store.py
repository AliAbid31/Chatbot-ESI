"""Local FAISS vector store over the knowledge base's chunk embeddings.

The corpus is small (a few hundred chunks at most), so an exact flat index
(``IndexFlatIP``) is used rather than an approximate one (IVF/HNSW) — at this
scale there's no accuracy/speed tradeoff to make, and a flat index needs no
training step.

Embeddings are cached to disk as a raw ``.npy`` array (not a serialized FAISS
index — rebuilding ``IndexFlatIP.add()`` from an array is instant, so there's
nothing to gain from persisting the index object itself). The cache key is a
hash of every chunk's text plus the embedding model name, mirroring the
content-hash invalidation pattern used elsewhere in this project's sibling
codebase: change the knowledge base, and the next startup re-embeds
automatically; change nothing, and a Render restart loads straight from disk
with no embedding calls at all.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import EMBED_DIM, MODEL_NAME, embed_texts

log = logging.getLogger(__name__)

# Bump this if the caching or indexing logic changes in a way that should
# invalidate old caches even though the source text didn't change.
INDEX_VERSION = "v1"


@dataclass(frozen=True)
class VectorHit:
    index: int      # position of the chunk in the knowledge base's chunk list
    score: float    # cosine similarity, roughly 0..1


def _signature(texts: list[str]) -> str:
    h = hashlib.sha256()
    h.update(MODEL_NAME.encode())
    h.update(INDEX_VERSION.encode())
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


class VectorStore:
    """Thin wrapper around a FAISS ``IndexFlatIP``."""

    def __init__(self, vectors: np.ndarray) -> None:
        import faiss  # imported lazily: only needed once semantic search is used

        self.index = faiss.IndexFlatIP(EMBED_DIM)
        if len(vectors):
            self.index.add(vectors)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[VectorHit]:
        if self.index.ntotal == 0:
            return []
        scores, ids = self.index.search(
            query_vector.reshape(1, -1).astype("float32"),
            min(top_k, self.index.ntotal),
        )
        return [
            VectorHit(index=int(i), score=float(s))
            for i, s in zip(ids[0], scores[0])
            if i >= 0
        ]

    @classmethod
    def build_or_load(cls, texts: list[str], cache_dir: Path) -> "VectorStore":
        """Build the embedding index, or load it from disk if content is unchanged."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        sig_path = cache_dir / "signature.txt"
        vec_path = cache_dir / "vectors.npy"
        sig = _signature(texts)

        if sig_path.exists() and vec_path.exists() and sig_path.read_text().strip() == sig:
            log.info("vector store: cache hit (%d chunks), loading %s", len(texts), vec_path)
            vectors = np.load(vec_path)
        else:
            log.info("vector store: cache miss, embedding %d chunks locally", len(texts))
            vectors = embed_texts(texts)
            np.save(vec_path, vectors)
            sig_path.write_text(sig)

        return cls(vectors)
