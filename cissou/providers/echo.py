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
        excerpt = context.strip()
        if re.search(r"[\u0600-\u06ff]", message):
            notice = "تعذر الاتصال بمزود الذكاء الاصطناعي، لذلك أعرض المقطع المطابق من وثائق ESI."
            question_label = "السؤال"
            passage_label = "المقطع الأكثر صلة من قاعدة معرفة ESI"
        elif re.search(r"\b(?:le|la|les|des|est|pour|quels?|quelles?|stage|stages)\b", message.lower()):
            notice = "Le fournisseur IA est indisponible; voici le passage correspondant des documents ESI."
            question_label = "Question"
            passage_label = "Passage le plus pertinent de la base documentaire ESI"
        else:
            notice = "The AI provider could not be reached, so here is the matching passage from the ESI documents."
            question_label = "Question"
            passage_label = "Most relevant passage from the ESI knowledge base"
        return (
            f"{notice}\n\n"
            f"{question_label}: {message}\n\n"
            f"{passage_label} (include the academic year, semester, and coefficient when shown):\n\n"
            f"{excerpt or '(no matching passage found)'}"
        )
