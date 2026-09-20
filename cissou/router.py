"""Key pooling and cross-provider failover.

The original implementation mutated a module-level key index from inside a
request handler, so concurrent requests raced each other and a key that was
permanently revoked was retried forever. This replaces it with a locked pool
per provider that tracks each key's health:

* ``AuthError``  -> the key is dead (revoked/denied); disable it for the process.
* ``QuotaError`` -> the key is rate limited; cool it off and move to the next.
* ``TransientError`` -> retry the same key once, then move on.

When every key of every provider is unusable the router degrades to the offline
provider instead of returning a 500, so the service keeps answering from the
knowledge base.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

from .providers import (AuthError, EchoProvider, LLMProvider, QuotaError,
                        TransientError)

log = logging.getLogger(__name__)

# Quota errors cover two very different situations: a per-minute rate limit that
# clears in seconds, and a daily quota that does not. We prefer the provider's
# own Retry-After hint, and otherwise back off progressively so a brief burst
# does not sideline a key for a quarter of an hour.
QUOTA_BACKOFF_SECONDS = (30.0, 120.0, 600.0, 1800.0)
MAX_COOLDOWN_SECONDS = 3600.0
TRANSIENT_RETRIES = 1
TRANSIENT_COOLDOWN_SECONDS = 1.0


@dataclass
class KeyState:
    key: str
    label: str
    disabled: bool = False
    cooldown_until: float = 0.0
    successes: int = 0
    failures: int = 0
    quota_strikes: int = 0

    def available(self, now: float) -> bool:
        return not self.disabled and now >= self.cooldown_until

    def public(self) -> dict:
        return {
            "label": self.label,
            "status": "disabled" if self.disabled
            else "cooling" if time.monotonic() < self.cooldown_until
            else "ready",
            "successes": self.successes,
            "failures": self.failures,
            "cooldown_seconds": max(0, round(self.cooldown_until - time.monotonic())),
        }


@dataclass
class ProviderPool:
    provider: LLMProvider
    keys: list[KeyState] = field(default_factory=list)
    _cursor: int = 0

    def next_key(self, now: float) -> KeyState | None:
        """Round-robin over healthy keys so load spreads instead of hammering #1."""
        count = len(self.keys)
        for offset in range(count):
            state = self.keys[(self._cursor + offset) % count]
            if state.available(now):
                self._cursor = (self._cursor + offset + 1) % count
                return state
        return None

    def healthy(self) -> bool:
        now = time.monotonic()
        return any(k.available(now) for k in self.keys)


class LLMRouter:
    def __init__(self, pools: list[ProviderPool], *, fallback: LLMProvider | None = None,
                 max_attempts: int = 4, deadline: float = 45.0) -> None:
        self.pools = pools
        self.fallback = fallback or EchoProvider(model="offline")
        self.max_attempts = max_attempts
        self.deadline = deadline
        self._lock = threading.Lock()

    # -- key checkout -------------------------------------------------
    def _checkout(self, pool: ProviderPool) -> KeyState | None:
        with self._lock:
            return pool.next_key(time.monotonic())

    def _penalise(self, state: KeyState, exc: Exception) -> None:
        with self._lock:
            state.failures += 1
            if isinstance(exc, AuthError):
                state.disabled = True
                log.warning("%s: disabled (%s)", state.label, exc)
            elif isinstance(exc, QuotaError):
                hint = getattr(exc, "retry_after", None)
                if hint:
                    # The provider told us exactly how long to wait.
                    delay = min(float(hint), MAX_COOLDOWN_SECONDS)
                else:
                    index = min(state.quota_strikes, len(QUOTA_BACKOFF_SECONDS) - 1)
                    delay = QUOTA_BACKOFF_SECONDS[index]
                state.quota_strikes += 1
                state.cooldown_until = time.monotonic() + delay
                log.warning("%s: quota exceeded, cooling down %.0fs", state.label, delay)
            elif isinstance(exc, TransientError):
                state.cooldown_until = time.monotonic() + TRANSIENT_COOLDOWN_SECONDS

    def _reward(self, state: KeyState) -> None:
        with self._lock:
            state.successes += 1
            state.cooldown_until = 0.0
            state.quota_strikes = 0    # the key recovered; forget past strikes

    # -- public API ---------------------------------------------------
    def generate(self, system: str, history: list[dict], message: str) -> tuple[str, str]:
        """Return ``(reply, provider_name)``, trying providers in order.

        Bounded by ``max_attempts`` keys and a wall-clock deadline: unhealthy
        keys stay disabled once discovered, so the cost of finding them is
        spread over several requests instead of landing on the first student
        after a restart.
        """
        errors: list[str] = []
        expires = time.monotonic() + self.deadline
        attempts = 0
        for pool in self.pools:
            while (state := self._checkout(pool)) is not None:
                if attempts >= self.max_attempts or time.monotonic() >= expires:
                    log.warning("answer budget spent (%d attempts); using fallback. "
                                "last errors: %s", attempts, "; ".join(errors[-2:]) or "none")
                    return (self.fallback.generate("", system, history, message),
                            self.fallback.name)
                attempts += 1
                for attempt in range(TRANSIENT_RETRIES + 1):
                    try:
                        reply = pool.provider.generate(state.key, system, history, message)
                        self._reward(state)
                        return reply, pool.provider.name
                    except TransientError as exc:
                        if attempt == TRANSIENT_RETRIES:
                            self._penalise(state, exc)
                            errors.append(f"{state.label}: {exc}")
                            break
                        else:
                            time.sleep(0.6 * (attempt + 1))
                    except (AuthError, QuotaError) as exc:
                        self._penalise(state, exc)
                        errors.append(f"{state.label}: {exc}")
                        break
        log.error("all providers exhausted: %s", "; ".join(errors[-4:]) or "no keys configured")
        return self.fallback.generate("", system, history, message), self.fallback.name

    def stream(self, system: str, history: list[dict], message: str) -> Iterator[tuple[str, str]]:
        """Yield ``(fragment, provider_name)``.

        Failover only applies before the first fragment is emitted: once bytes
        have reached the client we cannot restart on another provider without
        duplicating text, so a mid-stream failure ends the stream.
        """
        expires = time.monotonic() + self.deadline
        attempts = 0
        for pool in self.pools:
            while (state := self._checkout(pool)) is not None:
                if attempts >= self.max_attempts or time.monotonic() >= expires:
                    yield self.fallback.generate("", system, history, message), self.fallback.name
                    return
                attempts += 1
                fragments: list[str] = []
                try:
                    for fragment in pool.provider.stream(state.key, system, history, message):
                        fragments.append(fragment)
                    if fragments:
                        self._reward(state)
                        for fragment in fragments:
                            yield fragment, pool.provider.name
                        return
                    self._penalise(state, TransientError("empty stream"))
                except (AuthError, QuotaError, TransientError) as exc:
                    self._penalise(state, exc)
                    # Do not leak a partial provider response. Another provider
                    # or the fallback can still return a complete answer.
                    if isinstance(exc, TransientError):
                        continue
        yield self.fallback.generate("", system, history, message), self.fallback.name

    @property
    def healthy(self) -> bool:
        return any(pool.healthy() for pool in self.pools)

    def status(self) -> dict:
        return {
            "healthy": self.healthy,
            "providers": [
                {
                    "name": pool.provider.name,
                    "model": pool.provider.model,
                    "keys": [k.public() for k in pool.keys],
                }
                for pool in self.pools
            ],
        }


def build_router(settings) -> LLMRouter:
    """Assemble provider pools from settings, in the configured order."""
    from .providers import GeminiProvider, GroqProvider

    factories = {
        "gemini": (GeminiProvider, settings.gemini_model, settings.gemini_keys,
                   settings.max_output_tokens),
        "groq": (GroqProvider, settings.groq_model, settings.groq_keys,
                 settings.groq_max_output_tokens),
    }
    pools: list[ProviderPool] = []
    for name in settings.provider_order:
        entry = factories.get(name)
        if not entry:
            log.warning("unknown provider in PROVIDER_ORDER: %s", name)
            continue
        cls, model, keys, max_tokens = entry
        if not keys:
            continue
        provider = cls(
            model=model,
            temperature=settings.temperature,
            max_output_tokens=max_tokens,
            timeout=settings.request_timeout,
        )
        pools.append(ProviderPool(
            provider=provider,
            keys=[KeyState(key=k, label=f"{name}#{i}") for i, k in enumerate(keys, 1)],
        ))
    return LLMRouter(
        pools,
        max_attempts=settings.max_attempts_per_request,
        deadline=settings.answer_deadline,
    )
