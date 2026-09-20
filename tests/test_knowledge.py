from cissou.knowledge import MAX_CHUNK_CHARS, chunk_document
from pathlib import Path

SAMPLE = """# Doc Title

## Section A
Intro line for A.

### Subsection A1
Details about A1.

## Frequently Asked Questions

**Q: Is the internship mandatory?**
A: No, it is optional.

**Q: Can I retake an exam?**
A: Yes, via a replacement exam.
"""


def test_headings_build_breadcrumbs():
    chunks = chunk_document(SAMPLE)
    paths = {c.heading for c in chunks}
    assert "Section A" in paths
    assert "Section A > Subsection A1" in paths


def test_faq_entries_become_individual_chunks():
    chunks = chunk_document(SAMPLE)
    faqs = [c for c in chunks if c.text.startswith("Q:")]
    assert len(faqs) == 2
    assert "optional" in faqs[0].text
    # Each FAQ keeps the section it belongs to.
    assert faqs[0].path[-1] == "Frequently Asked Questions"


def test_sibling_heading_pops_the_stack():
    """'## Section B' after '### Subsection A1' must not nest under A1."""
    chunks = chunk_document(SAMPLE + "\n## Section B\nBody B.\n")
    section_b = next(c for c in chunks if c.text.strip() == "Body B.")
    assert section_b.path == ("Doc Title", "Section B")


def test_long_sections_are_windowed(document):
    chunks = chunk_document(document)
    assert chunks, "real document must produce chunks"
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)


def test_root_title_is_stripped_from_heading():
    chunk = next(c for c in chunk_document(SAMPLE) if c.heading == "Section A")
    assert "Doc Title" not in chunk.heading


def test_academic_level_stays_in_module_breadcrumb():
    document = """# Guide

# 1. First Year Preparatory Class — 1CP
## Infrastructure & Networks
### ARCH1 — Computer Architecture 1 (First Semester)
**Coefficient:** 4
"""
    chunk = next(c for c in chunk_document(document) if "ARCH1" in c.heading)
    assert "1CP" in chunk.heading
    assert "First Semester" in chunk.heading


def test_structured_module_catalog_contains_all_academic_groups():
    catalog = Path(__file__).resolve().parent.parent / "data" / "documents" / "academic" / "modules_catalog.md"
    text = catalog.read_text(encoding="utf-8")

    assert text.count("| 1CP |") == 16
    assert text.count("| 2CP |") == 16
    assert text.count("| 1CS |") == 17
    assert text.count("| 2CS | SID |") == 21
    assert text.count("| 2CS | SIT |") == 29
    assert text.count("| 2CS | SIL |") == 31
    assert text.count("| 2CS | SIQ |") == 31
    assert all(column in text for column in ("Code", "Nom", "Semestre", "Coefficient", "Description"))
