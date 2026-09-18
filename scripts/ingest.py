#!/usr/bin/env python3
"""Pre-build the local embedding/vector index for hybrid retrieval.

    python scripts/ingest.py

This is optional but recommended before deploying: it runs the (only)
CPU-heavy step — embedding every chunk with the local model — once, ahead of
time, and writes the result to ``data/index/``. Commit that directory (or run
this as a Render build step) so the deployed instance loads a cache hit on
startup instead of running the embedding pass on Render's 0.1 free-tier CPU.

Safe to re-run any time the knowledge base changes: the cache is keyed by a
hash of the chunk text, so unchanged content is a no-op and changed content
re-embeds automatically.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

    from cissou.config import Settings
    from cissou.knowledge import load_knowledge_source
    from cissou.retrieval import HybridKnowledgeBase

    settings = Settings.from_env()
    document, doc_chunks = load_knowledge_source(settings)
    if doc_chunks:
        print(f"Loading multi-document knowledge base from {settings.knowledge_dir} "
              f"({len(doc_chunks)} chunks before indexing) ...")
    else:
        print(f"Loading single-file knowledge base from {settings.knowledge_path} ...")

    started = time.perf_counter()
    kb = HybridKnowledgeBase(
        document,
        chunks=doc_chunks or None,
        cache_dir=settings.vector_index_dir,
        top_k=settings.retrieval_top_k,
        bm25_top_k=settings.bm25_top_k,
        vector_top_k=settings.vector_top_k,
        enable_semantic=True,  # always attempt the real build here, regardless of ENABLE_SEMANTIC_RETRIEVAL
        max_context_chars=settings.max_context_chars,
        min_score=settings.min_chunk_score,
    )
    elapsed = time.perf_counter() - started

    if kb.vector_store is None:
        print("WARNING: semantic index failed to build — check network access "
              "to the embedding model source. Falling back to BM25-only.")
        return 1

    print(f"Done in {elapsed:.1f}s | {kb.stats['chunks']} chunks | "
          f"index cached at {settings.vector_index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
