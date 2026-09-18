"""Flask application factory."""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

from .api import RateLimiter, bp
from .config import BASE_DIR, Settings
from .knowledge import load_knowledge_source
from .retrieval import HybridKnowledgeBase
from .router import build_router
from .service import ChatService
from .sessions import SessionStore

log = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )


def create_app(settings: Settings | None = None) -> Flask:
    load_dotenv(BASE_DIR / ".env", override=False)
    configure_logging()
    settings = settings or Settings.from_env()

    app = Flask(
        __name__,
        static_folder=str(BASE_DIR / "static"),
        static_url_path="/static",
    )
    CORS(app, resources={r"/api/*": {"origins": settings.cors_origins.split(",")}})

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
    router = build_router(settings)
    sessions = SessionStore(
        ttl_seconds=settings.session_ttl_seconds,
        max_sessions=settings.max_sessions,
        max_turns=settings.max_history_turns,
    )

    app.config["SETTINGS"] = settings
    app.config["CHAT_SERVICE"] = ChatService(kb, router, sessions)
    app.config["RATE_LIMITER"] = RateLimiter(settings.rate_limit_per_minute)
    app.register_blueprint(bp)

    log.info(
        "CISSOU ready | %d chars, %d chunks | retrieval: %s | providers: %s",
        kb.stats["characters"], kb.stats["chunks"],
        "hybrid (BM25 + local embeddings)" if kb.vector_store is not None else "BM25-only (semantic index unavailable)",
        ", ".join(f"{p.provider.name}({len(p.keys)} keys)" for p in router.pools) or "none (offline mode)",
    )
    return app
