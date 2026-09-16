"""Load the ESI knowledge base and split it into retrievable chunks.

The source document is structured Markdown, so chunking follows its headings
instead of blindly slicing every N characters. Each chunk keeps the breadcrumb
of headings above it, which gives the model the context it needs to answer
questions such as "how many hours of algorithms in 1CP?" where the year only
appears in the heading.
"""
from __future__ import annotations

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
    """A retrievable passage plus the heading trail it was found under."""

    title: str
    text: str
    path: tuple[str, ...]

    @property
    def heading(self) -> str:
        # path[0] is the document title, identical on every chunk: drop it so the
        # breadcrumb we spend prompt tokens on carries only distinguishing levels.
        trail = self.path[1:] if len(self.path) > 1 else self.path
        return " > ".join(trail) if trail else self.title

    def render(self) -> str:
        return f"### {self.heading}\n{self.text}"

    def __len__(self) -> int:
        return len(self.text)


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
