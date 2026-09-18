"""Google Gemini backend (google-genai SDK)."""
from __future__ import annotations

import re
from typing import Iterator

from .base import AuthError, LLMProvider, QuotaError, TransientError

_AUTH_MARKERS = ("permission_denied", "api key not valid", "unauthenticated",
                 "denied access", "api_key_invalid", "403", "401")
_QUOTA_MARKERS = ("quota", "resource_exhausted", "rate limit", "429")


def classify(exc: Exception) -> Exception:
    """Map an SDK exception onto the router's error taxonomy."""
    text = str(exc).lower()
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status == 429 or any(m in text for m in _QUOTA_MARKERS):
        return QuotaError(str(exc), retry_after=_retry_delay(text))
    if status in (401, 403) or any(m in text for m in _AUTH_MARKERS):
        return AuthError(str(exc))
    if status == 404 or "not found" in text:
        return AuthError(f"model unavailable: {exc}")
    return TransientError(str(exc))


def _retry_delay(text: str) -> float | None:
    """Gemini embeds a RetryInfo like 'retryDelay': '27s' in quota errors."""
    match = re.search(r"retrydelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", text)
    return float(match.group(1)) if match else None


class GeminiProvider(LLMProvider):
    name = "gemini"

    def _client(self, api_key: str):
        from google import genai  # lazy: keeps import cost off the request path
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            system_instruction=self._system,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return client, config

    @staticmethod
    def _contents(history: list[dict], message: str) -> list[dict]:
        contents = [
            {"role": "model" if turn["role"] == "assistant" else "user",
             "parts": [{"text": turn["content"]}]}
            for turn in history
        ]
        contents.append({"role": "user", "parts": [{"text": message}]})
        return contents

    def generate(self, api_key: str, system: str, history: list[dict], message: str) -> str:
        self._system = system
        try:
            client, config = self._client(api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=self._contents(history, message),
                config=config,
            )
            text = (response.text or "").strip()
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed LLMError
            raise classify(exc) from exc
        if not text:
            raise TransientError("empty response from Gemini")
        return text

    def stream(self, api_key: str, system: str, history: list[dict],
               message: str) -> Iterator[str]:
        self._system = system
        try:
            client, config = self._client(api_key)
            chunks = client.models.generate_content_stream(
                model=self.model,
                contents=self._contents(history, message),
                config=config,
            )
            for chunk in chunks:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:  # noqa: BLE001
            raise classify(exc) from exc
