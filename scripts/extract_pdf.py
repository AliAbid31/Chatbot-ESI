#!/usr/bin/env python3
"""Re-extract raw text from the source PDF.

The curated knowledge base at data/esi_knowledge.md is maintained by hand — this
script only dumps the PDF so you can diff it against the curated file and copy
across anything new. It deliberately does NOT overwrite the knowledge base.

    python scripts/extract_pdf.py                    # -> data/raw_pdf_text.txt
    python scripts/extract_pdf.py --stdout           # print instead
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = ROOT / "data" / "ESI101_data.pdf"
DEFAULT_OUT = ROOT / "data" / "raw_pdf_text.txt"


def extract(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        print(f"  page {number:>3}: {len(text):>6} chars", file=sys.stderr)
        parts.append(f"\n=== PAGE {number} ===\n{text}\n")
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    print(f"Extracting {args.pdf.name}", file=sys.stderr)
    text = extract(args.pdf)

    if args.stdout:
        print(text)
    else:
        args.out.write_text(text, encoding="utf-8")
        print(f"\n{len(text):,} characters -> {args.out}", file=sys.stderr)
        print("Diff this against data/esi_knowledge.md and port anything new by hand.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
