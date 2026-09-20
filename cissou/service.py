"""The chat pipeline: retrieve -> prompt -> generate -> remember."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from .prompts import build_system_prompt
from .retrieval import KnowledgeBase, normalize
from .router import LLMRouter
from .sessions import Session, SessionStore

log = logging.getLogger(__name__)

# Follow-ups ("and in 2CP?") carry too few words to retrieve on their own, so we
# prepend the previous user question to the retrieval query only.
FOLLOWUP_MAX_WORDS = 6
FOLLOWUP_RE = re.compile(r"^(?:and|et|what about|qu'en est-il|et pour)\b", re.IGNORECASE)
MODULE_LIST_RE = re.compile(
    r"\b(?:module|modules|matiere|matieres|cours|course|courses|programme|curriculum)\b",
    re.IGNORECASE,
)
ACADEMIC_LEVEL_RE = re.compile(r"\b(?:1|2)CP\b|\b(?:1|2|3)CS\b", re.IGNORECASE)
YEAR_HINTS = (
    (re.compile(r"\b(?:1ere|1re|premiere|first)(?!\s+semestre)(?:\s+annee?)?\b", re.IGNORECASE), "1CP"),
    (re.compile(r"\b(?:2eme|2e|deuxieme|second)(?!\s+semestre)(?:\s+annee?)?\b", re.IGNORECASE), "2CP"),
)
MODULE_HEADING_RE = re.compile(
    r"(?:^| > )([A-Z][A-Z0-9_]+)\s+[—-]\s+(.+?)\s+\((First|Second) Semester\)$"
)
COEFFICIENT_RE = re.compile(r"coefficient\s*:?\s*\*{0,2}\s*([0-9]+(?:[.,][0-9]+)?)", re.IGNORECASE)
CATALOG_ROW_RE = re.compile(r"^\|\s*(1CP|2CP|1CS|2CS)\s*\|\s*([^|]+)\|\s*(S1|S2)\s*\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*(.*?)\|\s*$")
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
FRENCH_RE = re.compile(
    r"\b(?:avec|dans|des|est|la|les|note|pour|quels?|quelles?|specialite|matiere|cours|donne|donnes|stage|stages|et|en|annee|premiere|deuxieme)\b",
    re.IGNORECASE,
)


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
        if (
            len(re.findall(r"\w+", message)) > FOLLOWUP_MAX_WORDS
            or ACADEMIC_LEVEL_RE.search(message)
            or not FOLLOWUP_RE.search(message.strip())
        ):
            query = message
        else:
            previous = [t["content"] for t in history if t["role"] == "user"]
            query = f"{previous[-1]} {message}" if previous else message

        normalized = normalize(query)
        if MODULE_LIST_RE.search(normalized) and not ACADEMIC_LEVEL_RE.search(normalized):
            level = self._academic_level(normalized)
            if level:
                query = f"{query} {level}"
        return query

    @staticmethod
    def _answer_language(message: str) -> str:
        if ARABIC_RE.search(message):
            return "ar"
        if (FRENCH_RE.search(normalize(message))
            or re.search(r"\by\s+a\s+t\s+il\b", normalize(message))
            or re.search(r"[àâçéèêëîïôùûüÿœ]", message.lower())):
            return "fr"
        return "en"

    @staticmethod
    def _academic_level(text: str) -> str | None:
        explicit = list(ACADEMIC_LEVEL_RE.finditer(text))
        if explicit:
            return explicit[-1].group(0).upper()
        hinted = [
            (match, level)
            for pattern, level in YEAR_HINTS
            for match in pattern.finditer(text)
        ]
        return max(hinted, key=lambda item: item[0].start())[1] if hinted else None

    def prepare(self, message: str, session: Session) -> tuple[str, list[dict], list[str]]:
        history = self.sessions.history(session)
        context, sources = self.kb.build_context(self._retrieval_query(message, history))
        return build_system_prompt(context), history, sources

    def structured_curriculum_answer(self, message: str, session: Session) -> str | None:
        """Format complete module lists directly from structured knowledge chunks."""
        query = self._retrieval_query(message, self.sessions.history(session))
        normalized = normalize(query)
        inferred_level = self._academic_level(normalized)
        if not MODULE_LIST_RE.search(normalized) or not inferred_level:
            if not inferred_level or not MODULE_LIST_RE.search(normalized):
                return None
        if not ACADEMIC_LEVEL_RE.search(normalized):
            query = f"{query} {inferred_level}"
            normalized = normalize(query)

        math_request = bool(re.search(r"\b(?:math|maths|mathematique|mathematiques|mathematics)\b", normalized))

        hits = self.kb.search(query)
        if math_request:
            prefixes = ("ALG", "ANAL", "PRST", "LOGM")
        else:
            prefixes = ()

        rows: list[tuple[str, str, str, str, str]] = []
        for chunk in self.kb.chunks:
            if chunk.source != "modules_explained":
                continue
            for line in chunk.text.splitlines():
                row = CATALOG_ROW_RE.match(line)
                if not row or row.group(1) != inferred_level:
                    continue
                code = row.group(4).strip()
                if prefixes and not code.startswith(prefixes):
                    continue
                semester = "First" if row.group(3) == "S1" else "Second"
                rows.append((semester, code, row.group(5).strip(), row.group(6).strip(), inferred_level))

        if not rows:
            modules = [hit.chunk for hit in hits if hit.chunk.source == "modules_explained"]
        else:
            modules = []
        entries: list[tuple[str, str, str, str, str]] = []
        if rows:
            entries = rows
        for chunk in modules:
            match = MODULE_HEADING_RE.search(chunk.heading)
            coefficient = COEFFICIENT_RE.search(chunk.text)
            level = ACADEMIC_LEVEL_RE.search(chunk.heading)
            if match and coefficient and level:
                entries.append((match.group(3), match.group(1), match.group(2),
                                coefficient.group(1), level.group(0).upper()))
        if not entries:
            return None

        language = self._answer_language(message)
        if language == "ar":
            title, scope_word, total_word = "هذه هي الوحدات المتاحة", "لـ", "وحدة إجمالاً"
        elif language == "fr":
            title, scope_word, total_word = "Voici les modules disponibles", "pour", "au total"
        else:
            title, scope_word, total_word = "Here are the available modules", "for", "total"
        lines = [f"{title} {scope_word} **{entries[0][4]}** ({len(entries)} {total_word}):"]
        for semester in ("First", "Second"):
            semester_entries = [entry for entry in entries if entry[0] == semester]
            if not semester_entries:
                continue
            if language == "ar":
                label = "السداسي الأول" if semester == "First" else "السداسي الثاني"
            elif language == "fr":
                label = "Premier semestre" if semester == "First" else "Deuxieme semestre"
            else:
                label = f"{semester} semester"
            lines.append(f"\n**{label}**")
            for _, code, name, coefficient, _ in semester_entries:
                coefficient_label = "المعامل" if language == "ar" else "coefficient"
                lines.append(f"- **{code}** - {name} ({coefficient_label}: {coefficient})")
        return "\n".join(lines)

    def structured_reference_answer(self, message: str, session: Session) -> str | None:
        """Return concise answers for high-confidence structured reference topics."""
        query = self._retrieval_query(message, self.sessions.history(session))
        normalized = normalize(query)
        if re.search(r"\b(?:stage|stages|internship|internships)\b", normalized):
            language = self._answer_language(message)
            if language == "fr":
                return ("Oui. L'ESI prévoit plusieurs stages :\n\n"
                        "- **Stage de découverte / stage industriel** : à la fin de la 1CP, "
                        "obligatoire, pendant environ **1 à 2 semaines**.\n"
                        "- **Stage en entreprise de 1CS** : obligatoire, pendant environ "
                        "**4 à 6 semaines**.")
            if language == "ar":
                return ("نعم. يتضمن تكوين ESI عدة تدريبات ميدانية:\n\n"
                        "- **تدريب اكتشافي / تدريب صناعي**: في نهاية السنة التحضيرية الأولى 1CP، "
                        "إجباري لمدة تقارب أسبوعاً إلى أسبوعين.\n"
                        "- **تدريب في مؤسسة خلال 1CS**: إجباري لمدة تقارب 4 إلى 6 أسابيع.")
            return ("Yes. ESI includes several internships:\n\n"
                    "- **Discovery / industrial internship**: at the end of 1CP, "
                    "mandatory, lasting about **1 to 2 weeks**.\n"
                    "- **1CS company internship**: mandatory, lasting about **4 to 6 weeks**.")

        if re.search(r"\b(?:specialit(?:e|es|y|ies)?|specialties|majors)\b", normalized):
            chunk = next(
                (item for item in self.kb.chunks
                 if item.source == "specialities_guide" and "four engineering majors" in normalize(item.heading)),
                None,
            )
            if chunk:
                majors = re.findall(r"^-\s+\*\*([A-Z]+)\*\*\s+[—-]\s+(.+)$", chunk.text, re.MULTILINE)
                language = self._answer_language(message)
                names = {
                    "SIT": {"fr": "Systemes d'information et technologies", "ar": "أنظمة المعلومات والتكنولوجيا"},
                    "SIQ": {"fr": "Systemes informatiques", "ar": "الأنظمة المعلوماتية"},
                    "SIL": {"fr": "Genie logiciel", "ar": "هندسة البرمجيات"},
                    "SID": {"fr": "Systemes intelligents et donnees", "ar": "الأنظمة الذكية والبيانات"},
                }
                if language == "fr":
                    title = "ESI propose quatre spécialités d'ingénierie :"
                elif language == "ar":
                    title = "تقدم المدرسة أربع تخصصات هندسية:"
                else:
                    title = "ESI offers four engineering majors:"
                lines = [title]
                for code, name in majors:
                    translated = names.get(code, {}).get(language, name)
                    lines.append(f"- **{code}** - {translated}")
                return "\n".join(lines)

        if (re.search(r"note eliminatoire|elimination threshold|eliminatory grade", normalized)
            or (ARABIC_RE.search(message) and re.search(r"علام|اقص", normalized))):
            chunk = next(
                (item for item in self.kb.chunks
                 if "note eliminatoire" in normalize(item.heading)),
                None,
            )
            if chunk:
                language = self._answer_language(message)
                if language == "fr":
                    return ("La note éliminatoire est la note minimale à ne pas dépasser vers le bas "
                            "pour valider un module. D'après le guide, elle correspond environ à "
                            "60 % de la moyenne de la classe pour le module et la période d'examen. "
                            "Ce n'est pas un nombre fixe comme 10/20.")
                if language == "ar":
                    return ("العلامة الإقصائية هي الحد الأدنى من العلامة اللازم لاجتياز الوحدة. "
                            "حسب الدليل، تساوي تقريباً 60٪ من متوسط القسم في الوحدة وفترة الامتحان، "
                            "وليست رقماً ثابتاً مثل 10 من 20.")
                return ("The elimination threshold is the minimum grade required to pass a module. "
                        "According to the guide, it is approximately 60% of the class average for "
                        "the relevant module and exam period. It is not a fixed number such as 10/20.")
        return None

    def structured_answer(self, message: str, session: Session) -> str | None:
        return (
            self.structured_curriculum_answer(message, session)
            or self.structured_reference_answer(message, session)
        )

    def answer(self, message: str, session_id: str | None = None) -> ChatResult:
        started = time.perf_counter()
        session = self.sessions.get(session_id)
        structured_reply = self.structured_curriculum_answer(message, session)
        if structured_reply is not None:
            history = self.sessions.history(session)
            _, _, sources = self.prepare(message, session)
            self.sessions.record(session, message, structured_reply)
            elapsed = int((time.perf_counter() - started) * 1000)
            return ChatResult(structured_reply, session.session_id, "knowledge", sources, elapsed)
        system, history, sources = self.prepare(message, session)
        reply, provider = self.router.generate(system, history, message)
        self.sessions.record(session, message, reply)
        elapsed = int((time.perf_counter() - started) * 1000)
        log.info("chat provider=%s sources=%d latency=%dms", provider, len(sources), elapsed)
        return ChatResult(reply.strip(), session.session_id, provider, sources, elapsed)
