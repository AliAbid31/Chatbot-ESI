#!/usr/bin/env python3
"""Ask CISSOU a question from the terminal, without running the server.

    python scripts/ask.py "Quelles sont les spécialités en 2CS ?"
    python scripts/ask.py --retrieval-only "admission average"
    python scripts/ask.py            # interactive REPL
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Query CISSOU from the CLI.")
    parser.add_argument("question", nargs="*", help="question (omit for a REPL)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="show retrieved passages without calling an LLM")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

    from cissou.config import Settings
    from cissou.knowledge import load_knowledge_source
    from cissou.retrieval import HybridKnowledgeBase
    from cissou.router import build_router
    from cissou.service import ChatService
    from cissou.sessions import SessionStore

    settings = Settings.from_env()
    document, doc_chunks = load_knowledge_source(settings)
    kb = HybridKnowledgeBase(
        document,
        chunks=doc_chunks or None,
        cache_dir=settings.vector_index_dir,
        top_k=settings.retrieval_top_k,
        bm25_top_k=settings.bm25_top_k,
        vector_top_k=settings.vector_top_k,
        enable_semantic=settings.enable_semantic_retrieval,
        max_context_chars=settings.max_context_chars,
        min_score=settings.min_chunk_score,
    )
    service = None if args.retrieval_only else ChatService(
        kb, build_router(settings), SessionStore())

    print(f"CISSOU · {kb.stats['chunks']} sections · "
          f"{'hybrid' if kb.vector_store is not None else 'BM25-only'} retrieval · "
          f"{'retrieval only' if args.retrieval_only else 'live'}\n")

    def handle(question: str, session_id: str | None) -> str | None:
        if args.retrieval_only:
            for hit in kb.search(question, top_k=args.top_k):
                print(f"  [{hit.score:6.2f}] {hit.chunk.heading}")
                print(f"           {hit.chunk.text[:200].strip()}...\n")
            return None
        result = service.answer(question, session_id)
        print(f"\n{result.reply}\n")
        print(f"— {result.provider} · {result.latency_ms}ms · "
              f"{len(result.sources)} sources")
        return result.session_id

    if args.question:
        handle(" ".join(args.question), None)
        return 0

    session_id = None
    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question in {"", "exit", "quit"}:
            return 0
        session_id = handle(question, session_id) or session_id


if __name__ == "__main__":
    raise SystemExit(main())
