"""Load the ESI knowledge base and split it into retrievable chunks.

The source document is structured Markdown, so chunking follows its headings
instead of blindly slicing every N characters. Each chunk keeps the breadcrumb
of headings above it, which gives the model the context it needs to answer
questions such as "how many hours of algorithms in 1CP?" where the year only
appears in the heading.
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
FAQ_RE = re.compile(r"^\*\*Q:\s*(.+?)\*\*\s*$")

# The verbatim PDF dump is one enormous paragraph; windows keep it retrievable.
MAX_CHUNK_CHARS = 2_400
CHUNK_OVERLAP_CHARS = 300


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage plus the heading trail it was found under.

    ``source``, ``category`` and ``language`` are optional metadata for the
    multi-document knowledge base structure (data/documents/<category>/...),
    used once content is split across category folders rather than the
    single markdown file. They default to empty so existing single-document
    chunks keep working unchanged.
    """

    title: str
    text: str
    path: tuple[str, ...]
    source: str = ""
    category: str = ""
    language: str = ""

    @property
    def heading(self) -> str:
        # path[0] is the document title, identical on every chunk: drop it so the
        # breadcrumb we spend prompt tokens on carries only distinguishing levels.
        keep_first = bool(self.path and ACADEMIC_LEVEL_RE.search(self.path[0]))
        trail = self.path if keep_first else self.path[1:] if len(self.path) > 1 else self.path
        return " > ".join(trail) if trail else self.title

    def render(self) -> str:
        return f"### {self.heading}\n{self.text}"

    def __len__(self) -> int:
        return len(self.text)


ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
FRENCH_HINTS = re.compile(
    r"\b(le|la|les|des|est|une|dans|pour|avec|vous|nous|étudiant|étudiants)\b",
    re.IGNORECASE,
)
ACADEMIC_LEVEL_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth)\s+year\b|\b(?:1|2)CS\b|\b(?:1|2)CP\b|\bcommon\s+core\b|\bspecialit",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """Coarse per-chunk language tag: 'ar', 'fr', or 'en'.

    Good enough for metadata/debugging and optional retrieval filtering, not
    meant as a real language-ID model. Arabic script is unambiguous; French
    vs. English falls back to a small function-word heuristic since the
    corpus mixes both without markup indicating which is which.
    """
    if ARABIC_RE.search(text):
        return "ar"
    if len(FRENCH_HINTS.findall(text)) >= 2:
        return "fr"
    return "en"


def load_documents_from_dir(base_dir: Path) -> list[Chunk]:
    """Load every markdown file under category subfolders into tagged chunks.

    Expected layout::

        base_dir/
          academic/*.md
          residences/*.md
          campus/*.md
          clubs/*.md
          general/*.md
          ...

    The category is just the immediate subfolder name — add new folders
    freely, nothing here hardcodes a fixed category list, matching the
    "categories should remain flexible" requirement. Each file's chunks
    (via the same heading-aware ``chunk_document`` used for the single-file
    KB) are tagged with that category, the filename as ``source``, and a
    per-chunk language guess — so a multi-document knowledge base gets the
    same structure-aware splitting as the original single markdown file,
    plus the metadata multi-source retrieval and citation needs.

    Returns an empty list (not an error) if the directory doesn't exist yet,
    so callers can fall back to the single-file knowledge base.
    """
    chunks: list[Chunk] = []
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return chunks

    for path in sorted(base_dir.rglob("*.md")):
        category = path.parent.name if path.parent != base_dir else "other"
        source = path.stem
        text = path.read_text(encoding="utf-8")
        for chunk in chunk_document(text):
            chunks.append(dataclasses.replace(
                chunk,
                source=source,
                category=category,
                language=detect_language(chunk.text),
            ))
    return chunks


def load_knowledge_source(settings) -> tuple[str | None, list]:
    """Resolve the knowledge base source per current settings.

    Returns (document, chunks) with exactly one populated: the multi-document
    directory (data/documents/<category>/*.md) if it exists and has content,
    otherwise the single markdown file it's replacing. Shared by app_factory
    and scripts/ingest.py so both pick the same source the same way.
    """
    from .knowledge import load_documents_from_dir, load_knowledge

    chunks = load_documents_from_dir(Path(settings.knowledge_dir))
    if chunks:
        return None, chunks
    document = load_knowledge(Path(settings.knowledge_path), settings.fallback_pdf_path)
    return document, []


def load_knowledge(path: Path, pdf_fallback: Path | None = None) -> str:
    """Read the knowledge document, falling back to the source PDF if missing."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        if text.strip():
            return text
        raise ValueError(f"{path} is empty")
    except (OSError, ValueError):
        if pdf_fallback and Path(pdf_fallback).exists():
            return _extract_pdf(Path(pdf_fallback))
        raise


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader  # imported lazily: only needed for the fallback

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _split_long(text: str, title: str, path: tuple[str, ...]) -> list[Chunk]:
    """Window an oversized section on sentence boundaries, with overlap."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [Chunk(title=title, text=text, path=path)]

    # Prefer breaking after sentence enders / list items over mid-word cuts.
    pieces = re.split(r"(?<=[.!?])\s+|\n(?=[-•*]\s)|\n{2,}", text)
    chunks: list[Chunk] = []
    buffer = ""
    part = 1
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if buffer and len(buffer) + len(piece) + 1 > MAX_CHUNK_CHARS:
            chunks.append(Chunk(f"{title} (part {part})", buffer.strip(), path))
            part += 1
            buffer = buffer[-CHUNK_OVERLAP_CHARS:] if CHUNK_OVERLAP_CHARS else ""
        buffer = f"{buffer} {piece}".strip()
    if buffer.strip():
        chunks.append(Chunk(f"{title} (part {part})", buffer.strip(), path))
    return chunks


def chunk_document(document: str) -> list[Chunk]:
    """Split Markdown into heading-scoped chunks, isolating each FAQ entry."""
    chunks: list[Chunk] = []
    stack: list[str] = []          # current heading breadcrumb
    body: list[str] = []           # lines accumulated under the current heading
    faq_question: str | None = None
    faq_body: list[str] = []

    def flush_faq() -> None:
        nonlocal faq_question, faq_body
        if faq_question:
            answer = "\n".join(faq_body).strip()
            text = f"Q: {faq_question}\n{answer}" if answer else f"Q: {faq_question}"
            chunks.append(Chunk(faq_question, text, tuple(stack)))
        faq_question, faq_body = None, []

    def flush_body() -> None:
        text = "\n".join(body).strip()
        body.clear()
        if text:
            title = stack[-1] if stack else "Overview"
            chunks.extend(_split_long(text, title, tuple(stack)))

    for line in document.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            flush_faq()
            flush_body()
            level = len(heading.group(1))
            del stack[level - 1:]
            stack.append(heading.group(2))
            continue

        faq = FAQ_RE.match(line.strip())
        if faq:
            flush_faq()
            flush_body()
            faq_question = faq.group(1).strip()
            continue

        if faq_question is not None:
            faq_body.append(line)
        else:
            body.append(line)

    flush_faq()
    flush_body()
    return [c for c in chunks if c.text.strip()]
