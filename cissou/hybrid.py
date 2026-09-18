"""Reciprocal Rank Fusion: combine BM25's lexical ranking with the vector
store's semantic ranking into one ordering.

RRF operates on rank position rather than raw scores, which sidesteps the
need to normalise BM25 scores (unbounded, corpus-dependent) against cosine
similarities (roughly 0..1) onto a shared scale — a common source of subtle
bugs in hand-rolled score blending. A chunk that ranks well in *either* list
gets a meaningful boost; a chunk ranking well in *both* rises to the top.
"""
from __future__ import annotations

from dataclasses import dataclass

from .knowledge import Chunk
from .vector_store import VectorHit

# Standard default from the original RRF paper; large enough that rank 1 vs.
# rank 2 isn't wildly more valuable than rank 9 vs. rank 10, which keeps a
# single list's noise from dominating the fused order.
RRF_K = 60


@dataclass(frozen=True)
class FusedHit:
    chunk: Chunk
    index: int
    score: float                 # fused RRF score
    bm25_rank: int | None        # 1-based rank in the BM25 list, or None if absent
    vector_rank: int | None      # 1-based rank in the vector list, or None if absent


def reciprocal_rank_fusion(
    chunks: list[Chunk],
    bm25_hits: list,          # list[retrieval.Hit] — must carry a populated .index
    vector_hits: list[VectorHit],
    *,
    k: int = RRF_K,
) -> list[FusedHit]:
    scores: dict[int, float] = {}
    bm25_rank: dict[int, int] = {}
    vector_rank: dict[int, int] = {}

    for rank, hit in enumerate(bm25_hits, start=1):
        scores[hit.index] = scores.get(hit.index, 0.0) + 1.0 / (k + rank)
        bm25_rank[hit.index] = rank

    for rank, hit in enumerate(vector_hits, start=1):
        scores[hit.index] = scores.get(hit.index, 0.0) + 1.0 / (k + rank)
        vector_rank[hit.index] = rank

    fused = [
        FusedHit(
            chunk=chunks[idx],
            index=idx,
            score=score,
            bm25_rank=bm25_rank.get(idx),
            vector_rank=vector_rank.get(idx),
        )
        for idx, score in scores.items()
    ]
    fused.sort(key=lambda h: h.score, reverse=True)
    return fused
