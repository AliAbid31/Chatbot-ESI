"""Per-conversation history.

The previous version kept a single module-level ``ConversationBufferMemory``, so
every visitor appended to the *same* transcript — one student's questions leaked
into another's context and the buffer grew without bound. History is now scoped
to a session id, trimmed to the last N turns, and evicted by TTL and LRU.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    turns: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        self.last_seen = time.time()


class SessionStore:
    """Thread-safe, bounded, in-process conversation store.

    In-process is intentional for a single-dyno deployment; swap for Redis if the
    app is ever scaled to multiple workers.
    """

    def __init__(self, *, ttl_seconds: int = 3600, max_sessions: int = 2000,
                 max_turns: int = 8) -> None:
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def _purge_expired(self, now: float) -> None:
        stale = [sid for sid, s in self._sessions.items() if now - s.last_seen > self.ttl]
        for sid in stale:
            del self._sessions[sid]

    def get(self, session_id: str | None) -> Session:
        now = time.time()
        with self._lock:
            self._purge_expired(now)
            if session_id and session_id in self._sessions:
                session = self._sessions.pop(session_id)
                session.last_seen = now
                self._sessions[session_id] = session   # refresh LRU position
                return session

            # Unknown ids mint a fresh session rather than being trusted, so a
            # client cannot claim someone else's transcript by guessing an id.
            new_id = session_id if _is_valid_id(session_id) else uuid.uuid4().hex
            session = Session(session_id=new_id)
            self._sessions[new_id] = session
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)   # evict least recently used
            return session

    def record(self, session: Session, user_message: str, reply: str) -> None:
        with self._lock:
            session.add("user", user_message)
            session.add("assistant", reply)
            excess = len(session.turns) - self.max_turns * 2
            if excess > 0:
                del session.turns[:excess]

    def history(self, session: Session) -> list[dict]:
        with self._lock:
            return list(session.turns)

    def reset(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


def _is_valid_id(session_id: str | None) -> bool:
    """Accept only ids shaped like the ones we mint."""
    return bool(session_id) and len(session_id) == 32 and all(
        c in "0123456789abcdef" for c in session_id
    )
