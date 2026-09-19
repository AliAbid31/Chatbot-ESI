"""Offline provider used by tests and by local runs without any API key.

It performs no network I/O and answers straight from the retrieved context, so
the whole pipeline — retrieval, sessions, routes — stays exercisable end to end.
"""
from __future__ import annotations

import re

from .base import LLMProvider


class EchoProvider(LLMProvider):
    name = "offline"

    def generate(self, api_key: str, system: str, history: list[dict], message: str) -> str:
        context = ""
        match = re.search(r"<knowledge>\n(.*?)\n</knowledge>", system, re.S)
        if match:
            context = match.group(1)
        excerpt = "\n".join(context.splitlines()[:14]).strip()
        return (
            "[offline mode — LLM provider unavailable or credentials not usable]\n\n"
            f"Question: {message}\n\n"
            "Most relevant passage from the ESI knowledge base "
            "(include the academic year, semester, and coefficient when shown):\n\n"
            f"{excerpt or '(no matching passage found)'}"
        )
