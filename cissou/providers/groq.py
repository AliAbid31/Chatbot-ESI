"""Groq backend via its OpenAI-compatible REST endpoint.

Used as the failover provider: Gemini free-tier keys are quota-limited and get
revoked when leaked, and a second vendor keeps CISSOU answering regardless.
"""
from __future__ import annotations

import json
from typing import Iterator

import requests

from .base import AuthError, LLMProvider, QuotaError, TransientError

ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


def _retry_after(response: requests.Response) -> float | None:
    """Groq returns the seconds until the limit clears; trust it when present."""
    raw = response.headers.get("retry-after") or response.headers.get("Retry-After")
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


class GroqProvider(LLMProvider):
    name = "groq"

    def _payload(self, system: str, history: list[dict], message: str, stream: bool) -> dict:
        messages = [{"role": "system", "content": system}]
        messages += [{"role": t["role"], "content": t["content"]} for t in history]
        messages.append({"role": "user", "content": message})
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            # Honour the configured budget exactly. Groq rejects a request whose
            # requested max_tokens exceeds the account's output-tokens-per-minute
            # limit (1000 on the free tier), so inflating this floors throughput
            # rather than raising it.
            "max_tokens": self.max_output_tokens,
            "stream": stream,
        }

    @staticmethod
    def _check(response: requests.Response) -> None:
        if response.status_code in (401, 403):
            if "network settings" in response.text.lower():
                raise TransientError(f"groq network access failed: {response.text[:200]}")
            raise AuthError(f"groq auth failed: {response.text[:200]}")
        if response.status_code == 429:
            raise QuotaError(f"groq quota exceeded: {response.text[:200]}",
                             retry_after=_retry_after(response))
        if response.status_code >= 400:
            raise TransientError(f"groq {response.status_code}: {response.text[:200]}")

    def generate(self, api_key: str, system: str, history: list[dict], message: str) -> str:
        try:
            response = requests.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}"},
                json=self._payload(system, history, message, stream=False),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise TransientError(str(exc)) from exc
        self._check(response)
        try:
            text = response.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError) as exc:
            raise TransientError(f"malformed groq response: {exc}") from exc
        if not text:
            raise TransientError("empty response from Groq")
        return text

    def stream(self, api_key: str, system: str, history: list[dict],
               message: str) -> Iterator[str]:
        try:
            response = requests.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}"},
                json=self._payload(system, history, message, stream=True),
                timeout=self.timeout,
                stream=True,
            )
        except requests.RequestException as exc:
            raise TransientError(str(exc)) from exc
        self._check(response)
        for raw in response.iter_lines(decode_unicode=False):
            if not raw:
                continue
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise TransientError(f"groq returned invalid UTF-8: {exc}") from exc
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content")
            except (KeyError, IndexError, ValueError):
                continue
            if delta:
                yield delta
