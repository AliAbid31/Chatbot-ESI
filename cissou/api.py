"""HTTP surface: chat, streaming chat, health and diagnostics."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict, deque

from flask import Blueprint, Response, current_app, jsonify, request, send_from_directory

log = logging.getLogger(__name__)
bp = Blueprint("api", __name__)


class RateLimiter:
    """Fixed-window per-IP limiter. Enough for a small campus deployment."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[client]
            while bucket and now - bucket[0] > 60:
                bucket.popleft()
            if len(bucket) >= self.per_minute:
                return False
            bucket.append(now)
            return True


def _client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    return forwarded.split(",")[0].strip() or request.remote_addr or "unknown"


def _fallback_session_id() -> str:
    """A pseudo session id for clients that never send one.

    The website's chat widget currently discards the ``session_id`` a reply
    returns instead of resending it, so without this every message would
    start a brand-new session and multi-turn follow-ups ("and in 2CP?")
    would never have any history to draw on. IP + User-Agent is a weak proxy
    for "same visitor" — it collides for people behind the same NAT/campus
    gateway, and breaks if a network changes mid-conversation — but it's the
    only continuity signal available without a frontend change (ideally: the
    client stores and resends the ``session_id`` from the response).

    Hashed to 32 lowercase hex characters to match the shape SessionStore
    already expects for a valid, mintable id (see sessions._is_valid_id).
    """
    fingerprint = f"{_client_ip()}|{request.headers.get('User-Agent', '')}"
    return hashlib.md5(fingerprint.encode()).hexdigest()


def _read_message() -> tuple[str | None, str | None, tuple[dict, int] | None]:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, None, ({"error": "invalid_request", "message": "JSON body required."}, 400)

    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return None, None, ({"error": "empty_message",
                             "message": "Please include a non-empty 'message' field."}, 400)

    limit = current_app.config["SETTINGS"].max_message_chars
    if len(message) > limit:
        return None, None, ({"error": "message_too_long",
                             "message": f"Keep your question under {limit} characters."}, 413)

    session_id = data.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        session_id = _fallback_session_id()
    return message.strip(), session_id, None


@bp.get("/")
def index():
    return send_from_directory(current_app.static_folder, "index.html")


@bp.get("/health")
def health():
    service = current_app.config["CHAT_SERVICE"]
    return jsonify({
        "status": "ok",
        "assistant": "CISSOU",
        "knowledge": service.kb.stats,
        "llm_available": service.router.healthy,
        "active_sessions": len(service.sessions),
    })


@bp.get("/api/diagnostics")
def diagnostics():
    service = current_app.config["CHAT_SERVICE"]
    return jsonify({
        "knowledge": service.kb.stats,
        "router": service.router.status(),
        "active_sessions": len(service.sessions),
    })


@bp.post("/api/chat")
@bp.post("/chat")  # the live esi101 frontend calls this bare path, not /api/chat
def chat():
    if not current_app.config["RATE_LIMITER"].allow(_client_ip()):
        return jsonify({"error": "rate_limited",
                        "message": "Too many questions at once — give me a moment!"}), 429

    message, session_id, error = _read_message()
    if error:
        payload, status = error
        return jsonify(payload), status

    service = current_app.config["CHAT_SERVICE"]
    try:
        result = service.answer(message, session_id)
    except Exception:
        log.exception("chat failed")
        return jsonify({"error": "internal_error",
                        "message": "CISSOU hit an unexpected problem. Please try again."}), 500

    return jsonify({
        "reply": result.reply,
        "session_id": result.session_id,
        "provider": result.provider,
        "sources": result.sources,
        "latency_ms": result.latency_ms,
    })


@bp.post("/api/chat/stream")
def chat_stream():
    """Server-sent events, so the UI can render tokens as they arrive."""
    if not current_app.config["RATE_LIMITER"].allow(_client_ip()):
        return jsonify({"error": "rate_limited",
                        "message": "Too many questions at once — give me a moment!"}), 429

    message, session_id, error = _read_message()
    if error:
        payload, status = error
        return jsonify(payload), status

    service = current_app.config["CHAT_SERVICE"]
    session = service.sessions.get(session_id)
    structured_reply = service.structured_curriculum_answer(message, session)
    system, history, sources = service.prepare(message, session)

    def events():
        yield _sse({"type": "meta", "session_id": session.session_id, "sources": sources})
        parts: list[str] = []
        provider = "unknown"
        try:
            if structured_reply is not None:
                provider = "knowledge"
                parts.append(structured_reply)
                yield _sse({"type": "delta", "text": structured_reply})
            else:
                for fragment, name in service.router.stream(system, history, message):
                    provider = name
                    parts.append(fragment)
                    yield _sse({"type": "delta", "text": fragment})
        except Exception:
            log.exception("stream failed")
            yield _sse({"type": "error", "message": "The answer was interrupted. Please retry."})
            return
        reply = "".join(parts).strip()
        if reply:
            service.sessions.record(session, message, reply)
        yield _sse({"type": "done", "provider": provider})

    return Response(events(), content_type="text/event-stream; charset=utf-8",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@bp.post("/api/session/reset")
def reset_session():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if not isinstance(session_id, str):
        return jsonify({"error": "invalid_request", "message": "session_id required."}), 400
    cleared = current_app.config["CHAT_SERVICE"].sessions.reset(session_id)
    return jsonify({"cleared": cleared})


@bp.get("/api/search")
def search():
    """Inspect retrieval without spending an LLM call — useful for tuning.

    With hybrid retrieval enabled, also returns the BM25-only ranking, the
    vector-only ranking, and how RRF fused them — so a regression can be
    traced to a specific stage instead of just seeing the final top-k.
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "invalid_request", "message": "Query parameter 'q' required."}), 400
    kb = current_app.config["CHAT_SERVICE"].kb
    hits = kb.search(query)
    payload = {"query": query, "results": [
        {"heading": h.chunk.heading, "score": round(h.score, 3), "excerpt": h.chunk.text[:280]}
        for h in hits
    ]}
    if hasattr(kb, "debug_search"):
        payload["retrieval_breakdown"] = kb.debug_search(query)
    return jsonify(payload)
