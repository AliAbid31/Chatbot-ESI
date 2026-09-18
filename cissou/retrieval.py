"""Lexical retrieval over the ESI knowledge base.

Deliberately embedding-free: the corpus is ~66k characters, and a BM25 index
built at start-up answers in microseconds with no API call, no quota cost and no
vector store to deploy. What makes it accurate here is the domain layer on top —
accent-folded bilingual (FR/EN) normalisation plus an ESI/CSE alias table, so
"prépa", "1CP" and "first year" all reach the same passages.
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .knowledge import Chunk

log = logging.getLogger(__name__)

# Latin letters/digits plus Arabic letters, digits, and the Arabic
# Supplement block's extended letters — deliberately narrower than the full
# 0600-06FF Arabic Unicode block, which also contains punctuation (0x061F
# '؟', 0x060C '،', 0x061B '؛') and tatweel (0x0640, a stretch character with
# no lexical meaning) that would otherwise get glued onto adjacent tokens.
# The original pattern only matched [a-z0-9], which silently dropped every
# Arabic token before it ever reached the tokenizer. Arabic harakat
# (diacritics) are handled separately by ``normalize`` below, since — like
# accented Latin — they're Unicode *combining marks* stripped by NFKD
# decomposition.
TOKEN_RE = re.compile(r"[a-z0-9\u0621-\u063a\u0641-\u064a\u0660-\u0669\u066e-\u06d3\u06d5\u0750-\u077f]+")

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
    # Common spelling used by students; the curriculum labels the module ARCH1.
    "archi1": ("arch1", "computer", "architecture", "first", "semester"),
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
    # Position of ``chunk`` in the knowledge base's chunk list. Populated by
    # BM25Index.search so hybrid retrieval can align BM25 hits with vector
    # store hits (which are index-based) for Reciprocal Rank Fusion, without
    # relying on Chunk equality (two FAQ entries could coincidentally have
    # identical text).
    index: int = -1


class BM25Index:
    """Okapi BM25 with a heading-match bonus and exact module-code detection."""

    K1 = 1.5
    B = 0.75
    HEADING_BOOST = 1.6
    # Sections whose content is duplicated elsewhere in a cleaner form. They stay
    # searchable (they are the only source for a few details) but lose ties to the
    # curated sections.
    REDUNDANT_SECTIONS = ("Original PDF Content",)
    REDUNDANT_PENALTY = 0.75

    # A query containing an exact module/specialization code (SIQ, PFE, ALSDS,
    # 1CP, POO...) should reliably surface the section documenting that exact
    # code, even if that section's surrounding prose doesn't share many other
    # terms with the query. Restricted to codes actually seen either as a
    # heading or in this corpus's "**CODE** — Description" list convention
    # (not any short word) so ordinary words never accidentally qualify.
    # Length >= 3 specifically excludes 2-letter false positives like "IS"
    # (which shows up embedded in several module full names, e.g. "MPSI —
    # IS Conduite de changement", and would otherwise fire on nearly every
    # query, since "is" appears in almost every question). The lookahead
    # requires at least one letter, so pure numbers (years, grades) never
    # qualify as a "code" either.
    CODE_RE = re.compile(r"\b(?=[A-Z0-9]{3,6}\b)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{3,6}\b")
    BOLD_CODE_RE = re.compile(r"\*\*([A-Z0-9]{3,6})\*\*\s*[—–-]")
    HEADING_CODE_BONUS = 20.0
    CONTENT_CODE_BONUS = 12.0

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._docs: list[Counter[str]] = []
        self._heading_tokens: list[set[str]] = []
        self._lengths: list[int] = []
        self._weights: list[float] = []
        self._raw_content: list[str] = []          # original-case, for exact code matching
        self._heading_codes: list[set[str]] = []
        doc_freq: Counter[str] = Counter()
        known_codes: set[str] = set()

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

            heading_codes = set(self.CODE_RE.findall(chunk.heading))
            self._heading_codes.append(heading_codes)
            self._raw_content.append(chunk.text)
            known_codes |= heading_codes | set(self.BOLD_CODE_RE.findall(chunk.text))

        # Only codes that actually label some heading in this corpus count as
        # "known" — this is what keeps the bonus from firing on any short
        # all-caps word a query happens to contain.
        self._known_codes = known_codes

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
        query_codes = set(self.CODE_RE.findall(query.upper())) & self._known_codes
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
            for code in query_codes:
                if code in self._heading_codes[i]:
                    score += self.HEADING_CODE_BONUS
                elif re.search(rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])", self._raw_content[i]):
                    score += self.CONTENT_CODE_BONUS
            score *= self._weights[i]
            if score > 0:
                scored.append(Hit(self.chunks[i], score, i))

        scored.sort(key=lambda h: h.score, reverse=True)
        if not scored:
            return []
        best = scored[0].score
        return [h for h in scored[:top_k] if h.score >= best * min_score]


class KnowledgeBase:
    """Chunked corpus + index + prompt-sized context assembly."""

    def __init__(
        self,
        document: str | None = None,
        *,
        chunks: list[Chunk] | None = None,
        top_k: int = 10,
        max_context_chars: int = 24_000,
        min_score: float = 0.15,
    ) -> None:
        if (document is None) == (chunks is None):
            raise ValueError("KnowledgeBase needs exactly one of `document` or `chunks`")

        if chunks is None:
            from .knowledge import chunk_document

            chunks = chunk_document(document)
        else:
            # Multi-document case: there's no single source text, so
            # reconstruct one for the "nothing matched" fallback in
            # build_context and for the character count in `.stats`.
            document = "\n\n".join(c.render() for c in chunks)

        self.document = document
        self.chunks = chunks
        self.index = BM25Index(self.chunks)
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.min_score = min_score

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        return self.index.search(query, top_k or self.top_k, self.min_score)

    def build_context(self, query: str) -> tuple[str, list[str]]:
        """Return (context text, source labels) capped at ``max_context_chars``.

        A source label is the chunk's ``source`` filename when the knowledge
        base was built from tagged multi-document chunks, or its heading
        breadcrumb otherwise — whichever actually distinguishes where the
        passage came from for the caller (e.g. the /api/chat citation list).

        Falls back to the head of the document when nothing matches, so the
        model always has something grounded to work from.
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
            label = hit.chunk.source or hit.chunk.heading
            if label not in sources:
                sources.append(label)
            budget -= len(rendered) + 2
        if not parts:  # single hit larger than the whole budget
            parts.append(hits[0].chunk.render()[: self.max_context_chars])
            sources.append(hits[0].chunk.source or hits[0].chunk.heading)
        return "\n\n".join(parts), sources

    @property
    def stats(self) -> dict[str, int]:
        return {
            "characters": len(self.document),
            "chunks": len(self.chunks),
            "vocabulary": len(self.index._idf),
        }


class HybridKnowledgeBase(KnowledgeBase):
    """BM25 + local semantic embeddings, fused with Reciprocal Rank Fusion.

    Falls back to BM25-only automatically if the vector store can't be built
    (e.g. the embedding model failed to download) — retrieval degrades
    gracefully rather than the app failing to start. This matters
    specifically on Render's free tier, where a transient network hiccup
    during a cold start shouldn't take the whole chatbot down.
    """

    def __init__(
        self,
        document: str | None = None,
        *,
        chunks: list[Chunk] | None = None,
        cache_dir,
        bm25_top_k: int = 10,
        vector_top_k: int = 10,
        enable_semantic: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(document, chunks=chunks, **kwargs)
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.vector_store = None
        if not enable_semantic:
            log.info("semantic retrieval disabled by config — BM25-only")
            return
        try:
            from .vector_store import VectorStore

            texts = [c.render() for c in self.chunks]
            self.vector_store = VectorStore.build_or_load(texts, cache_dir)
        except Exception:
            log.exception("semantic index unavailable — falling back to BM25-only retrieval")

    def _fused(self, query: str):
        from .hybrid import reciprocal_rank_fusion

        bm25_hits = self.index.search(query, self.bm25_top_k, min_score=0.0)
        vector_hits: list = []
        if self.vector_store is not None:
            from .embeddings import embed_query

            vector_hits = self.vector_store.search(embed_query(query), self.vector_top_k)
        fused = reciprocal_rank_fusion(self.chunks, bm25_hits, vector_hits)
        return bm25_hits, vector_hits, fused

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        top_k = top_k or self.top_k
        if self.vector_store is None:
            # No semantic layer available: behave exactly like plain BM25,
            # including its min_score cutoff.
            return self.index.search(query, top_k, self.min_score)

        _, _, fused = self._fused(query)
        return [Hit(f.chunk, f.score, f.index) for f in fused[:top_k]]

    def debug_search(self, query: str) -> dict:
        """Retrieval breakdown for /api/search — BM25, vector, and fused ranks."""
        bm25_hits, vector_hits, fused = self._fused(query)
        return {
            "bm25": [
                {"heading": h.chunk.heading, "score": round(h.score, 3)}
                for h in bm25_hits
            ],
            "vector": [
                {"heading": self.chunks[h.index].heading, "score": round(h.score, 3)}
                for h in vector_hits
            ],
            "fused": [
                {
                    "heading": f.chunk.heading,
                    "score": round(f.score, 4),
                    "bm25_rank": f.bm25_rank,
                    "vector_rank": f.vector_rank,
                }
                for f in fused[: self.top_k]
            ],
        }
