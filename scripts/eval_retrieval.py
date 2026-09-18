#!/usr/bin/env python3
"""Score retrieval against a bilingual question set.

Retrieval quality is what decides whether CISSOU can answer a question at all,
so it gets measured rather than eyeballed:

    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --verbose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cissou.knowledge import load_knowledge      # noqa: E402
from cissou.retrieval import KnowledgeBase       # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "esi_knowledge.md"

# (question, a string that must appear in the retrieved context)
CASES: list[tuple[str, str]] = [
    # --- English ---
    ("Who developed CISSOU?", "Badreddine"),
    ("What is the admission average for ESI?", "18.19"),
    ("How many spots are available each year?", "250"),
    ("What are the 2CS specializations?", "SID"),
    ("Which specialization covers artificial intelligence?", "Intelligent Systems"),
    ("What courses are in the first year preparatory cycle?", "ALSDS"),
    ("What is taught in 2CP?", "2CP"),
    ("Is the discovery internship mandatory?", "not mandatory"),
    ("Can I retake an exam after a justified absence?", "replacement exam"),
    ("How do I join CSE?", "join"),
    ("What events does CSE organize?", "DataHack"),
    ("What is Hack!T?", "hackathon"),
    ("What is Tresor?", "Trésor"),
    ("Where is ESI located?", "Oued Smar"),
    ("When was ESI founded?", "1984"),
    ("What jobs can ESI graduates get?", "career"),
    ("What challenges do ESI students face?", "workload"),
    ("Can international students join ESI?", "international"),
    ("What is the PFE?", "PFE"),
    ("How long is the whole programme?", "five"),
    # --- French ---
    ("Qui a développé CISSOU ?", "Badreddine"),
    ("Quelle est la moyenne d'admission ?", "18.19"),
    ("Quelles sont les spécialités en 2CS ?", "SID"),
    ("Quelle spécialité pour l'intelligence artificielle ?", "Intelligent Systems"),
    ("Quels sont les cours en 1CP ?", "ALSDS"),
    ("Le stage de découverte est-il obligatoire ?", "internship"),
    ("Comment rejoindre le CSE ?", "CSE"),
    ("Quels événements organise le CSE ?", "DataHack"),
    ("Où se trouve l'ESI ?", "Oued Smar"),
    ("Quels sont les débouchés après l'ESI ?", "career"),
    ("Combien d'années dure la formation ?", "five"),
    ("Quelles sont les difficultés à l'ESI ?", "challenge"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    kb = KnowledgeBase(load_knowledge(DATA))
    print(f"Knowledge base: {kb.stats['characters']:,} chars · "
          f"{kb.stats['chunks']} chunks · {kb.stats['vocabulary']:,} terms\n")

    failures: list[tuple[str, str]] = []
    context_sizes: list[int] = []

    for question, expected in CASES:
        context, sources = kb.build_context(question)
        context_sizes.append(len(context))
        hit = expected.lower() in context.lower()
        if not hit:
            failures.append((question, expected))
        if args.verbose or not hit:
            mark = "PASS" if hit else "FAIL"
            print(f"[{mark}] {question}")
            print(f"        expects {expected!r} · {len(sources)} sources · "
                  f"{len(context):,} chars")
            if args.verbose:
                for s in sources[:3]:
                    print(f"          - {s}")

    total = len(CASES)
    passed = total - len(failures)
    avg = sum(context_sizes) // max(len(context_sizes), 1)
    print(f"\n{'-' * 60}")
    print(f"Recall@context : {passed}/{total} ({100 * passed / total:.1f}%)")
    print(f"Avg context    : {avg:,} chars (~{avg // 4:,} tokens)")
    print(f"Full-doc cost  : {kb.stats['characters']:,} chars "
          f"(~{kb.stats['characters'] // 4:,} tokens) — what the old app sent every turn")
    print(f"Reduction      : {100 - 100 * avg / kb.stats['characters']:.1f}% fewer context tokens")
    if failures:
        print(f"\n{len(failures)} failing:")
        for question, expected in failures:
            print(f"  - {question}  (missing {expected!r})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
