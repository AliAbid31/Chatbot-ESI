import pytest

from cissou.retrieval import expand_query, normalize, stem, tokenize


def test_normalize_folds_french_accents():
    assert normalize("Préparatoire") == "preparatoire"
    assert normalize("découverte") == "decouverte"


def test_stopwords_are_dropped():
    assert "the" not in tokenize("what is the school")
    assert "quelle" not in tokenize("quelle est la moyenne")


def test_stem_folds_plurals():
    assert stem("specializations") == stem("specialization")


def test_alias_expansion_bridges_french_to_english():
    # expand_query returns stems, so compare against stemmed expectations.
    assert stem("mandatory") in expand_query("est-ce obligatoire")
    assert stem("career") in expand_query("les débouchés")


def test_alias_lookup_works_on_plural_forms():
    assert stem("specialization") in expand_query("les spécialités")


def test_archi1_alias_reaches_arch1_module():
    from cissou.knowledge import Chunk
    from cissou.retrieval import BM25Index

    chunks = [
        Chunk(title="Modules", text="Computer Architecture 1 (First Semester).",
              path=("Modules", "ARCH1 — Computer Architecture 1")),
        Chunk(title="Residence", text="The residence has a study room.",
              path=("Residence", "Is there a study room?")),
    ]
    hits = BM25Index(chunks).search("which semester will I study ARCHI1?", top_k=2)
    assert hits[0].chunk.path[-1].startswith("ARCH1")


def test_stemming_is_symmetric_for_singular_and_plural():
    """A query word and its corpus form must land on the same stem."""
    for singular, plural in [("course", "courses"), ("specialite", "specialites"),
                             ("matiere", "matieres"), ("year", "years")]:
        assert stem(singular) == stem(plural)


# (query, substring that must appear in one of the top-3 passages)
CASES = [
    ("Who developed CISSOU?", "Badreddine"),
    ("Qui a développé CISSOU ?", "Badreddine"),
    ("What is the admission average in 2025?", "18.19"),
    ("Quelle est la moyenne d'admission ?", "18.19"),
    ("What are the 2CS specializations?", "SID"),
    ("Quelles sont les spécialités en 2CS ?", "SID"),
    ("Is the discovery internship mandatory?", "not mandatory"),
    ("Le stage de découverte est-il obligatoire ?", "internship"),
    ("How can I join CSE?", "join"),
    ("Comment rejoindre le CSE ?", "CSE"),
    ("Can I retake an exam after a justified absence?", "replacement exam"),
    ("What courses are in 1CP?", "ALSDS"),
    ("Quels sont les débouchés après l'ESI ?", "career"),
    ("What is Hack!T?", "hackathon"),
    ("Where is ESI located?", "Oued Smar"),
    ("What is Trésor?", "Trésor"),
]


@pytest.mark.parametrize("query,expected", CASES)
def test_retrieved_context_contains_the_answer(kb, query, expected):
    """The context CISSOU actually receives must contain the supporting fact."""
    context, sources = kb.build_context(query)
    assert sources, f"no results for {query!r}"
    assert expected.lower() in context.lower(), f"{expected!r} missing for {query!r}"


@pytest.mark.parametrize("query,expected", CASES)
def test_answer_ranks_in_the_top_five(kb, query, expected):
    # render() is what reaches the model: heading breadcrumb + body. Several
    # facts (specialization acronyms, for one) live only in the heading.
    hits = kb.search(query, top_k=5)
    blob = " ".join(h.chunk.render() for h in hits).lower()
    assert expected.lower() in blob, f"{expected!r} not ranked top-5 for {query!r}"


def test_context_respects_the_char_budget(kb):
    kb.max_context_chars = 3_000
    context, sources = kb.build_context("What are the 2CS specializations?")
    assert len(context) <= 3_000
    assert sources
    kb.max_context_chars = 24_000


def test_unmatched_query_falls_back_to_document_head(kb):
    context, sources = kb.build_context("zzzz qqqq xxxx")
    assert context, "must never hand the model an empty context"
    assert sources == []


def test_redundant_section_loses_to_curated_section(kb):
    top = kb.search("Who developed CISSOU?", top_k=1)[0]
    assert "Original PDF Content" not in top.chunk.heading


# --- Exact module/specialization code bonus ---------------------------------

def test_exact_heading_code_ranks_first():
    from cissou.knowledge import Chunk
    from cissou.retrieval import BM25Index

    chunks = [
        Chunk(title="Modules", text="ALSDS — Algorithms and data structures.",
              path=("Modules", "ALSDS — Algorithms and data structures")),
        Chunk(title="Modules", text="A general module about many algorithmic topics "
                                     "and data handling techniques used across courses.",
              path=("Modules", "General overview")),
    ]
    index = BM25Index(chunks)
    hits = index.search("ALSDS", top_k=2)
    assert hits[0].chunk.path[-1].startswith("ALSDS")


def test_bold_list_code_gets_content_bonus():
    from cissou.knowledge import Chunk
    from cissou.retrieval import BM25Index

    chunks = [
        Chunk(title="Doc", text="- **COMPIL** — Compilation\n- **BDA** — Databases",
              path=("Doc", "First Semester Modules")),
        Chunk(title="Doc", text="Some unrelated section about student clubs and events.",
              path=("Doc", "Clubs")),
    ]
    index = BM25Index(chunks)
    hits = index.search("what is COMPIL", top_k=2)
    assert hits[0].chunk.path[-1] == "First Semester Modules"


def test_short_common_words_never_count_as_codes():
    # Regression test: "IS" (2 letters) used to leak into known_codes because
    # some real headings embed it (e.g. "MPSI — IS Conduite de changement"),
    # and "is" appears in almost every question — which meant nearly every
    # query got a spurious bonus toward IS-related sections. Codes now
    # require length >= 3 specifically to rule this out.
    from cissou.knowledge import Chunk
    from cissou.retrieval import BM25Index

    chunks = [
        Chunk(title="Doc", text="MPSI is the IS conduite de changement module.",
              path=("Doc", "MPSI — IS Conduite de changement")),
        Chunk(title="Doc", text="A note about clubs and student life at the school.",
              path=("Doc", "Student Life")),
    ]
    index = BM25Index(chunks)
    assert "IS" not in index._known_codes
    assert "AI" not in index._known_codes  # same length-3 rule applies


def test_pure_numbers_never_count_as_codes():
    from cissou.knowledge import Chunk
    from cissou.retrieval import BM25Index

    chunks = [Chunk(title="Doc", text="Admissions in 2025 totalled 310 places.",
                     path=("Doc", "2025 Admission Averages"))]
    index = BM25Index(chunks)
    assert "2025" not in index._known_codes
    assert "310" not in index._known_codes
