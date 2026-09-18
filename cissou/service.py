"""The chat pipeline: retrieve -> prompt -> generate -> remember."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from .prompts import build_system_prompt
from .retrieval import KnowledgeBase
from .router import LLMRouter
from .sessions import Session, SessionStore

log = logging.getLogger(__name__)

# Follow-ups ("and in 2CP?") carry too few words to retrieve on their own, so we
# prepend the previous user question to the retrieval query only.
FOLLOWUP_MAX_WORDS = 6


@dataclass
class ChatResult:
    reply: str
    session_id: str
    provider: str
    sources: list[str]
    latency_ms: int


class ChatService:
    def __init__(self, kb: KnowledgeBase, router: LLMRouter, sessions: SessionStore) -> None:
        self.kb = kb
        self.router = router
        self.sessions = sessions

    def _retrieval_query(self, message: str, history: list[dict]) -> str:
        if len(re.findall(r"\w+", message)) > FOLLOWUP_MAX_WORDS:
            return message
        previous = [t["content"] for t in history if t["role"] == "user"]
        return f"{previous[-1]} {message}" if previous else message

    def prepare(self, message: str, session: Session) -> tuple[str, list[dict], list[str]]:
        history = self.sessions.history(session)
        context, sources = self.kb.build_context(self._retrieval_query(message, history))
        return build_system_prompt(context), history, sources

    def answer(self, message: str, session_id: str | None = None) -> ChatResult:
        started = time.perf_counter()
        session = self.sessions.get(session_id)
        system, history, sources = self.prepare(message, session)
        reply, provider = self.router.generate(system, history, message)
        self.sessions.record(session, message, reply)
        elapsed = int((time.perf_counter() - started) * 1000)
        log.info("chat provider=%s sources=%d latency=%dms", provider, len(sources), elapsed)
        return ChatResult(reply.strip(), session.session_id, provider, sources, elapsed)
