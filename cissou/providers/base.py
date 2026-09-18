"""Provider interface and the error taxonomy the router dispatches on."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class LLMError(RuntimeError):
    """Base class for provider failures."""


class AuthError(LLMError):
    """Key is invalid, revoked or denied. Retrying the same key is pointless."""


class QuotaError(LLMError):
    """Rate limit or quota exhausted. Another key may still work.

    ``retry_after`` carries the provider's own hint in seconds when it sends
    one. It matters: a per-minute token limit clears in seconds, while a daily
    quota does not, and the two are otherwise indistinguishable.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TransientError(LLMError):
    """Server-side or network hiccup. The same key is worth retrying."""


class LLMProvider(ABC):
    """A single named backend, driven with one API key at a time."""

    name: str = "provider"

    def __init__(self, model: str, *, temperature: float = 0.3,
                 max_output_tokens: int = 1400, timeout: float = 60.0) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    @abstractmethod
    def generate(self, api_key: str, system: str, history: list[dict], message: str) -> str:
        """Return the assistant reply, or raise an ``LLMError`` subclass."""

    def stream(self, api_key: str, system: str, history: list[dict],
               message: str) -> Iterator[str]:
        """Yield reply fragments. Defaults to a single-chunk non-streaming call."""
        yield self.generate(api_key, system, history, message)
