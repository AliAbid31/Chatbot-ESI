"""Tests for hybrid retrieval: RRF fusion, the FAISS vector store, Arabic
tokenization, and HybridKnowledgeBase's fallback behaviour.

None of these touch the network or download the embedding model — vectors
are injected directly, and the model download path is exercised only via
its *absence* (enable_semantic=False), the same way CI stays offline. This
mirrors the project's existing rule that the test suite needs no keys and no
network, extended to cover the new semantic layer without actually calling
out to fetch model weights.
"""
from __future__ import annotations

import numpy as np
import pytest

from cissou.hybrid import reciprocal_rank_fusion
from cissou.knowledge import Chunk
from cissou.retrieval import Hit, HybridKnowledgeBase, tokenize
from cissou.vector_store import VectorHit, VectorStore


def _chunks(n: int) -> list[Chunk]:
    return [Chunk(title=f"t{i}", text=f"text {i}", path=(f"t{i}",)) for i in range(n)]


def _unit_vectors(n: int, dim: int = 384, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype("float32")
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


# --- RRF fusion ---------------------------------------------------------

def test_rrf_ranks_items_in_both_lists_above_single_list_items():
    chunks = _chunks(5)
    bm25_hits = [Hit(chunks[0], 5.0, 0), Hit(chunks[1], 4.0, 1), Hit(chunks[2], 3.0, 2)]
    vector_hits = [VectorHit(2, 0.9), VectorHit(3, 0.8), VectorHit(4, 0.7)]

    fused = reciprocal_rank_fusion(chunks, bm25_hits, vector_hits)

    assert fused[0].index == 2  # only chunk appearing in both lists
    assert fused[0].bm25_rank == 3 and fused[0].vector_rank == 1


def test_rrf_handles_disjoint_lists_without_crashing():
    chunks = _chunks(4)
    bm25_hits = [Hit(chunks[0], 1.0, 0)]
    vector_hits = [VectorHit(3, 0.5)]

    fused = reciprocal_rank_fusion(chunks, bm25_hits, vector_hits)

    assert {f.index for f in fused} == {0, 3}
    assert all(f.score > 0 for f in fused)


def test_rrf_empty_inputs_return_empty():
    assert reciprocal_rank_fusion(_chunks(3), [], []) == []


# --- Vector store --------------------------------------------------------

def test_vector_store_self_match_scores_near_one():
    vecs = _unit_vectors(5)
    store = VectorStore(vecs)

    hits = store.search(vecs[2], top_k=3)

    assert hits[0].index == 2
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_vector_store_empty_index_returns_no_hits():
    store = VectorStore(np.zeros((0, 384), dtype="float32"))
    assert store.search(_unit_vectors(1)[0], top_k=5) == []


def test_vector_store_build_or_load_round_trips_through_cache(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_embed_texts(texts):
        calls["n"] += 1
        return _unit_vectors(len(texts), seed=1)

    monkeypatch.setattr("cissou.vector_store.embed_texts", fake_embed_texts)

    texts = ["a", "b", "c"]
    VectorStore.build_or_load(texts, tmp_path)
    VectorStore.build_or_load(texts, tmp_path)  # should hit the cache, not re-embed

    assert calls["n"] == 1


def test_vector_store_cache_invalidates_on_content_change(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_embed_texts(texts):
        calls["n"] += 1
        return _unit_vectors(len(texts), seed=1)

    monkeypatch.setattr("cissou.vector_store.embed_texts", fake_embed_texts)

    VectorStore.build_or_load(["a", "b"], tmp_path)
    VectorStore.build_or_load(["a", "b", "c"], tmp_path)  # content changed

    assert calls["n"] == 2


# --- Arabic tokenization ---------------------------------------------------

def test_arabic_text_is_not_dropped_entirely():
    # Regression test: TOKEN_RE used to be [a-z0-9]+ only, which silently
    # produced zero tokens for any Arabic-script query.
    tokens = tokenize("أين تقع الإقامة الجامعية؟")
    assert tokens, "Arabic text must not tokenize to an empty list"


def test_arabic_punctuation_is_not_glued_onto_tokens():
    tokens = tokenize("أين تقع الإقامة الجامعية؟")
    assert all("؟" not in t and "،" not in t for t in tokens)


def test_arabic_hamza_variants_fold_together():
    from cissou.retrieval import normalize
    # NFKD decomposes hamza-above/below forms onto a shared alef base.
    assert normalize("أحمد") == normalize("إحمد")


# --- HybridKnowledgeBase fallback behaviour --------------------------------

def test_hybrid_kb_with_semantic_disabled_behaves_like_bm25_only(tmp_path):
    document = "# Title\n## Section\nSome content about 3CS specialization.\n"
    kb = HybridKnowledgeBase(document, cache_dir=tmp_path, enable_semantic=False)

    assert kb.vector_store is None
    hits = kb.search("3CS")
    assert hits and hits[0].chunk.heading == "Section"


def test_hybrid_kb_falls_back_gracefully_if_vector_store_build_raises(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr("cissou.vector_store.VectorStore.build_or_load", boom)

    document = "# Title\n## Section\nSome content.\n"
    kb = HybridKnowledgeBase(document, cache_dir=tmp_path, enable_semantic=True)

    assert kb.vector_store is None  # caught, not raised
    assert kb.search("content")     # still functions via BM25


def test_hybrid_kb_debug_search_reports_all_three_rankings(tmp_path, monkeypatch):
    def fake_build_or_load(texts, cache_dir):
        return VectorStore(_unit_vectors(len(texts), seed=2))

    monkeypatch.setattr("cissou.vector_store.VectorStore.build_or_load", fake_build_or_load)
    # The vector store's *search* is the thing under test here; the query's
    # own embedding is irrelevant to verifying debug_search's output shape,
    # so it's stubbed too rather than hitting the real (network-dependent)
    # embedding model.
    monkeypatch.setattr("cissou.embeddings.embed_query", lambda text: _unit_vectors(1, seed=3)[0])

    document = "# Title\n## A\nContent A.\n## B\nContent B.\n"
    kb = HybridKnowledgeBase(document, cache_dir=tmp_path, enable_semantic=True)

    assert kb.vector_store is not None
    breakdown = kb.debug_search("content")
    assert set(breakdown.keys()) == {"bm25", "vector", "fused"}
    assert breakdown["fused"]


def test_module_list_query_returns_every_requested_semester_module():
    first = ["ALSDS", "ARCH1", "ANAL1", "ALG1", "ELECT", "SYST1", "DAIL", "AWPS"]
    second = ["ALSDD", "SYST2", "ANAL2", "ALG2"]
    chunks = [
        Chunk(
            title=code,
            text=f"**Coefficient:** 3\nDescription for {code}.",
            path=("1CP", f"{code} (First Semester)"),
            source="modules_explained",
        )
        for code in first
    ] + [
        Chunk(
            title=code,
            text=f"**Coefficient:** 3\nDescription for {code}.",
            path=("1CP", f"{code} (Second Semester)"),
            source="modules_explained",
        )
        for code in second
    ]
    kb = HybridKnowledgeBase(chunks=chunks, cache_dir=None, enable_semantic=False)

    context, _ = kb.build_context("donne-moi tous les modules de S1 1CP")

    assert all(code in context for code in first)
    assert not any(code in context for code in second)
