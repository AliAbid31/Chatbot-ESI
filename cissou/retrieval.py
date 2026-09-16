"""Lexical retrieval over the ESI knowledge base.

Deliberately embedding-free: the corpus is ~66k characters, and a BM25 index
built at start-up answers in microseconds with no API call, no quota cost and no
vector store to deploy. What makes it accurate here is the domain layer on top —
accent-folded bilingual (FR/EN) normalisation plus an ESI/CSE alias table, so
"prépa", "1CP" and "first year" all reach the same passages.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .knowledge import Chunk

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    # English
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "its", "me", "of", "on", "or", "that",
    "the", "there", "they", "this", "to", "was", "what", "when", "where", "which",
    "who", "will", "with", "you", "your", "about", "tell", "give", "please",
    # French
    "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en",
    "est", "et", "eux", "il", "je", "la", "le", "les", "leur", "lui", "ma",
    "mais", "me", "mes", "moi", "mon", "ne", "nos", "notre", "nous", "on", "ou",
    "par", "pas", "pour", "qu", "que", "qui", "sa", "se", "ses", "son", "sur",
    "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos", "votre", "vous",
    "quoi", "quel", "quelle", "quels", "quelles", "comment", "pourquoi", "combien",
    "dis", "moi", "parle", "explique",
}

# Domain vocabulary: each key expands into extra query terms so a student's
# phrasing reaches the passage even when the document words it differently.
ALIASES: dict[str, tuple[str, ...]] = {
    "esi": ("ecole", "superieure", "informatique", "ini", "ceri", "school"),
    "cse": ("scientific", "club", "clubs"),
    "cissou": ("app", "application", "chatbot", "assistant"),
    "tresor": ("platform", "resources", "plateforme"),
    "1cp": ("first", "year", "preparatory", "premiere", "annee", "prepa", "cp"),
    "2cp": ("second", "year", "preparatory", "deuxieme", "annee", "prepa", "cp"),
    "1cs": ("third", "year", "common", "core", "tronc", "commun", "cs"),
    "2cs": ("fourth", "year", "specialization", "specialite", "cs"),
    "3cs": ("fifth", "year", "final", "pfe", "cs"),
    "prepa": ("preparatory", "1cp", "2cp", "cp", "classes"),
    "preparatoire": ("preparatory", "1cp", "2cp", "cp"),
    "sit": ("information", "technology", "systems", "specialization"),
    "siq": ("computer", "systems", "specialization"),
    "sil": ("software", "engineering", "systems", "specialization"),
    "sid": ("intelligent", "systems", "data", "specialization"),
    "pfe": ("final", "year", "project", "3cs", "memoire"),
    "bac": ("baccalaureate", "baccalaureat", "admission", "moyenne"),
    "concours": ("admission", "selective", "competition"),
    "stage": ("internship", "internships", "discovery", "entreprise"),
    "internship": ("stage", "discovery", "enterprise", "professional"),
    "job": ("career", "careers", "employment", "graduates", "opportunities"),
    "emploi": ("career", "job", "travail", "graduates"),
    "travail": ("career", "job", "work", "emploi"),
    "hackathon": ("hackt", "hack", "hackin", "datahack", "event"),
    "datathon": ("datahack", "data", "event"),
    "event": ("events", "evenement", "hackt", "datahack", "workshop"),
    "rejoindre": ("join", "membership", "recruitment", "inscription"),
    "join": ("rejoindre", "membership", "recruit", "registration"),
    "cours": ("course", "courses", "subject", "module", "curriculum"),
    "course": ("cours", "module", "subject", "curriculum", "matiere"),
    "matiere": ("subject", "course", "module", "cours"),
    "programme": ("curriculum", "program", "structure", "syllabus"),
    "curriculum": ("programme", "program", "structure", "courses"),
    "admission": ("admissions", "entry", "requirements", "bac", "moyenne", "inscription"),
    "difficile": ("difficult", "challenges", "hard", "pressure", "workload"),
    "difficulty": ("challenges", "difficile", "hard", "workload", "pressure"),
    "vie": ("life", "student", "campus", "social"),
    "life": ("vie", "student", "campus", "social"),
    "examen": ("exam", "exams", "cr", "rattrapage", "replacement"),
    "exam": ("examen", "exams", "cr", "replacement", "retake"),
    "note": ("grade", "moyenne", "average", "marks"),
    "diplome": ("diploma", "degree", "engineer", "ingenieur"),
    "etudiant": ("student", "students", "etudiants"),
    "student": ("etudiant", "students", "eleve"),
    "professeur": ("teacher", "faculty", "staff", "enseignant"),
    "histoire": ("history", "founded", "created", "ceri", "ini"),
    "history": ("histoire", "founded", "created", "ceri", "ini"),
    "creer": ("created", "founded", "developed", "cree"),
    "developpe": ("developed", "created", "sayah", "badreddine"),
    "ia": ("ai", "artificial", "intelligence", "intelligence artificielle"),
    "ai": ("ia", "artificial", "intelligence", "machine", "learning"),
    # FR -> EN bridges: the corpus is mostly English, students often ask in French.
    "debouches": ("career", "opportunities", "jobs", "graduates", "employment"),
    "obligatoire": ("mandatory", "required", "compulsory"),
    "decouverte": ("discovery", "internship", "stage"),
    "entreprise": ("company", "enterprise", "industry", "professional"),
    "specialite": ("specialization", "specializations", "2cs", "sit", "siq", "sil", "sid"),
    "annee": ("year", "years", "annual"),
    "semestre": ("semester", "semesters"),
    "module": ("course", "subject", "matiere", "curriculum"),
    "inscription": ("registration", "admission", "join", "apply"),
    "moyenne": ("average", "grade", "admission", "score"),
    "ecole": ("school", "esi", "institution"),
    "etudes": ("studies", "program", "curriculum", "training"),
    "duree": ("duration", "years", "length", "long"),
    "apres": ("after", "graduation", "career", "graduates"),
    "salaire": ("salary", "career", "employment"),
    "etranger": ("international", "foreign", "abroad"),
    "bourse": ("scholarship", "scholarships", "funding"),
    "horaire": ("schedule", "hours", "timetable"),
    "heures": ("hours", "volume", "schedule"),
    "enseignement": ("teaching", "education", "course", "training"),
    "reussir": ("succeed", "success", "tips", "advice"),
    "conseil": ("advice", "tips", "guidance", "help"),
    "nouveau": ("new", "newcomer", "freshman", "beginner"),
    "debutant": ("beginner", "beginners", "new", "starter"),
    "avantage": ("benefit", "benefits", "advantage", "advantages"),
    "fonde": ("founded", "created", "establishment", "history"),
    "situe": ("located", "location", "campus", "address"),
    "ou": ("where", "location", "located", "campus"),
}

_ALIASES_BY_STEM: dict[str, tuple[str, ...]] = {}

# Suffixes trimmed to fold plural / derived forms onto a shared stem. Order
# matters: longest first. Plurals are handled by the bare "s" only — stripping
# "es" too made stemming asymmetric ("courses" -> "cours" but "course" ->
# "course"), which silently broke matching between a query and the corpus.
_SUFFIXES = ("ements", "ement", "ations", "ation", "ances", "ance", "ities",
             "ity", "ings", "ing", "s")


def normalize(text: str) -> str:
    """Lowercase and strip accents so 'préparatoire' == 'preparatoire'."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def stem(token: str) -> str:
    if len(token) <= 4 or token[0].isdigit():
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _alias_index() -> dict[str, tuple[str, ...]]:
    """Alias keys folded through the stemmer, so plurals resolve too."""
    if not _ALIASES_BY_STEM:
        _ALIASES_BY_STEM.update({stem(k): v for k, v in ALIASES.items()})
    return _ALIASES_BY_STEM


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    tokens = TOKEN_RE.findall(normalize(text))
    out: list[str] = []
    for token in tokens:
        if not keep_stopwords and token in STOPWORDS:
            continue
        out.append(stem(token))
    return out


def expand_query(text: str) -> list[str]:
    """Tokenize, then append alias terms for any domain word we recognise."""
    raw = TOKEN_RE.findall(normalize(text))
    tokens = [stem(t) for t in raw if t not in STOPWORDS]
    for token in raw:
        # Try the literal word first, then its stem, so both "specialite" and
        # "specialites" resolve through the same alias entry.
        expansions = ALIASES.get(token) or _alias_index().get(stem(token)) or ()
        for extra in expansions:
            tokens.extend(stem(t) for t in TOKEN_RE.findall(extra))
    return tokens


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25Index:
    """Okapi BM25 with a heading-match bonus."""

    K1 = 1.5
    B = 0.75
    HEADING_BOOST = 1.6
    # Sections whose content is duplicated elsewhere in a cleaner form. They stay
    # searchable (they are the only source for a few details) but lose ties to the
    # curated sections.
    REDUNDANT_SECTIONS = ("Original PDF Content",)
    REDUNDANT_PENALTY = 0.75

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._docs: list[Counter[str]] = []
        self._heading_tokens: list[set[str]] = []
        self._lengths: list[int] = []
        self._weights: list[float] = []
        doc_freq: Counter[str] = Counter()

        for chunk in chunks:
            head = tokenize(chunk.heading)
            tokens = tokenize(chunk.text) + head * 2  # headings count double
            counts = Counter(tokens)
            self._docs.append(counts)
            self._heading_tokens.append(set(head))
            self._lengths.append(max(len(tokens), 1))
            trail = " > ".join(chunk.path)
            penalised = any(marker in trail for marker in self.REDUNDANT_SECTIONS)
            self._weights.append(self.REDUNDANT_PENALTY if penalised else 1.0)
            doc_freq.update(counts.keys())

        n = max(len(chunks), 1)
        self._avg_len = sum(self._lengths) / n
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def search(self, query: str, top_k: int = 10, min_score: float = 0.0) -> list[Hit]:
        terms = expand_query(query)
        if not terms:
            return []
        weights = Counter(terms)
        scored: list[Hit] = []

        for i, counts in enumerate(self._docs):
            length = self._lengths[i]
            score = 0.0
            for term, qty in weights.items():
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = tf + self.K1 * (1 - self.B + self.B * length / self._avg_len)
                contribution = idf * (tf * (self.K1 + 1)) / denom
                if term in self._heading_tokens[i]:
                    contribution *= self.HEADING_BOOST
                score += contribution * (1 + 0.1 * (qty - 1))
            score *= self._weights[i]
            if score > 0:
                scored.append(Hit(self.chunks[i], score))

        scored.sort(key=lambda h: h.score, reverse=True)
        if not scored:
            return []
        best = scored[0].score
        return [h for h in scored[:top_k] if h.score >= best * min_score]


class KnowledgeBase:
    """Chunked corpus + index + prompt-sized context assembly."""

    def __init__(self, document: str, *, top_k: int = 10, max_context_chars: int = 24_000,
                 min_score: float = 0.15) -> None:
        from .knowledge import chunk_document

        self.document = document
        self.chunks = chunk_document(document)
        self.index = BM25Index(self.chunks)
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.min_score = min_score

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        return self.index.search(query, top_k or self.top_k, self.min_score)

    def build_context(self, query: str) -> tuple[str, list[str]]:
        """Return (context text, source headings) capped at ``max_context_chars``.

        Falls back to the head of the document when nothing matches, so the model
        always has something grounded to work from.
        """
        hits = self.search(query)
        if not hits:
            return self.document[: self.max_context_chars], []

        parts: list[str] = []
        sources: list[str] = []
        budget = self.max_context_chars
        for hit in hits:
            rendered = hit.chunk.render()
            if len(rendered) > budget:
                break
            parts.append(rendered)
            if hit.chunk.heading not in sources:
                sources.append(hit.chunk.heading)
            budget -= len(rendered) + 2
        if not parts:  # single hit larger than the whole budget
            parts.append(hits[0].chunk.render()[: self.max_context_chars])
            sources.append(hits[0].chunk.heading)
        return "\n\n".join(parts), sources

    @property
    def stats(self) -> dict[str, int]:
        return {
            "characters": len(self.document),
            "chunks": len(self.chunks),
            "vocabulary": len(self.index._idf),
        }
