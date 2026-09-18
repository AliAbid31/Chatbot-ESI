"""Application settings, resolved once from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def collect_keys(prefix: str, *aliases: str) -> list[str]:
    """Collect ``PREFIX`` and ``PREFIX_2..PREFIX_20`` into a de-duplicated list.

    Order is preserved so the primary key stays first. Duplicates are dropped
    because rotating onto the same key twice wastes a retry on a quota that is
    already exhausted.
    """
    prefixes = (prefix, *aliases)
    raw = [os.getenv(name) for name in prefixes]
    raw.extend(
        os.getenv(f"{name}_{i}")
        for name in prefixes
        for i in range(2, 21)
    )
    seen: set[str] = set()
    keys: list[str] = []
    for key in raw:
        key = (key or "").strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


@dataclass(frozen=True)
class Settings:
    # --- Knowledge base ---
    knowledge_path: Path = BASE_DIR / "data" / "esi_knowledge.md"
    fallback_pdf_path: Path = BASE_DIR / "data" / "ESI101_data.pdf"
    # Multi-document knowledge base: data/documents/<category>/*.md. Takes
    # priority over knowledge_path/fallback_pdf_path when it exists and has
    # content — the single markdown file remains supported as a fallback for
    # anyone who hasn't migrated their content into category folders yet.
    knowledge_dir: Path = BASE_DIR / "data" / "documents"
    # Cached local embeddings + persisted vector data for the semantic half
    # of hybrid retrieval. Safe to delete: it's rebuilt automatically from
    # the knowledge base on next startup, just slower (an embedding pass
    # instead of a cache load).
    vector_index_dir: Path = BASE_DIR / "data" / "index"

    # --- Retrieval ---
    retrieval_top_k: int = 10
    bm25_top_k: int = 10
    vector_top_k: int = 10
    max_context_chars: int = 24_000
    min_chunk_score: float = 0.15
    # Off in tests/CI: without this, every test that builds an app would hit
    # fastembed's model-download retry loop (three attempts with growing
    # backoff) against a network CI likely doesn't have, turning a fast
    # offline test suite into one that hangs for minutes. Real deployments
    # want this on; set ENABLE_SEMANTIC_RETRIEVAL=false to force BM25-only.
    enable_semantic_retrieval: bool = True

    # --- Providers ---
    gemini_keys: list[str] = field(default_factory=list)
    gemini_model: str = "gemini-3.5-flash"
    groq_keys: list[str] = field(default_factory=list)
    groq_model: str = "qwen/qwen3.8-27b"
    # Groq's free tier allows 1000 output tokens per minute; asking for more
    # gets the request rejected outright.
    groq_max_output_tokens: int = 900
    provider_order: tuple[str, ...] = ("gemini", "groq")

    # --- Generation ---
    temperature: float = 0.3
    max_output_tokens: int = 1400
    # One provider call, capped below answer_deadline so a slow call plus the
    # fallback still returns inside the deadline.
    request_timeout: float = 30.0
    # Bound the work one request will do discovering unhealthy keys, so a single
    # student never waits through the whole pool after a restart.
    max_attempts_per_request: int = 4
    answer_deadline: float = 45.0

    # --- Sessions ---
    session_ttl_seconds: int = 3600
    max_sessions: int = 2_000
    max_history_turns: int = 8

    # --- HTTP ---
    max_message_chars: int = 2_000
    cors_origins: str = "*"
    rate_limit_per_minute: int = 20

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            knowledge_path=Path(os.getenv("KNOWLEDGE_PATH") or cls.knowledge_path),
            knowledge_dir=Path(os.getenv("KNOWLEDGE_DIR") or cls.knowledge_dir),
            vector_index_dir=Path(os.getenv("VECTOR_INDEX_DIR") or cls.vector_index_dir),
            retrieval_top_k=_env_int("RETRIEVAL_TOP_K", cls.retrieval_top_k),
            bm25_top_k=_env_int("BM25_TOP_K", cls.bm25_top_k),
            vector_top_k=_env_int("VECTOR_TOP_K", cls.vector_top_k),
            enable_semantic_retrieval=_env_bool("ENABLE_SEMANTIC_RETRIEVAL",
                                                cls.enable_semantic_retrieval),
            max_context_chars=_env_int("MAX_CONTEXT_CHARS", cls.max_context_chars),
            gemini_keys=collect_keys(
                "GOOGLE_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"
            ),
            gemini_model=os.getenv("GEMINI_MODEL") or cls.gemini_model,
            groq_keys=collect_keys("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL") or cls.groq_model,
            groq_max_output_tokens=_env_int("GROQ_MAX_OUTPUT_TOKENS",
                                            cls.groq_max_output_tokens),
            provider_order=tuple(
                p.strip()
                for p in (os.getenv("PROVIDER_ORDER") or "gemini,groq").split(",")
                if p.strip()
            ),
            temperature=_env_float("TEMPERATURE", cls.temperature),
            max_output_tokens=_env_int("MAX_OUTPUT_TOKENS", cls.max_output_tokens),
            request_timeout=_env_float("REQUEST_TIMEOUT", cls.request_timeout),
            max_attempts_per_request=_env_int("MAX_ATTEMPTS_PER_REQUEST",
                                              cls.max_attempts_per_request),
            answer_deadline=_env_float("ANSWER_DEADLINE", cls.answer_deadline),
            session_ttl_seconds=_env_int("SESSION_TTL_SECONDS", cls.session_ttl_seconds),
            max_history_turns=_env_int("MAX_HISTORY_TURNS", cls.max_history_turns),
            max_message_chars=_env_int("MAX_MESSAGE_CHARS", cls.max_message_chars),
            cors_origins=os.getenv("CORS_ORIGINS") or cls.cors_origins,
            rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", cls.rate_limit_per_minute),
        )

    @property
    def offline_mode(self) -> bool:
        """True when no provider credentials exist at all."""
        return not (self.gemini_keys or self.groq_keys)
