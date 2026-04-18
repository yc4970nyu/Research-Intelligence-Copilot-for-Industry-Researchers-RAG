import os
from backend.ingestion.chunker import Chunk
from backend.retrieval.bm25 import BM25Index
from backend.retrieval.vector_store import VectorStore

# RRF constant - 60 is the standard value from the original paper
# higher k means lower ranks matter more, reduces sensitivity to outliers
RRF_K = 60


def hybrid_search(
    query: str,
    store: VectorStore,
    top_k: int = None,
) -> list[tuple[Chunk, float]]:
    """
    Combine semantic search and BM25 keyword search using
    Reciprocal Rank Fusion (RRF).

    RRF formula: score(doc) = sum over each ranker of 1 / (k + rank)

    Why RRF instead of just averaging scores?
    BM25 scores and cosine similarity scores are on completely different scales
    so you can't just add them directly. RRF only uses rank positions,
    which makes the combination scale-invariant. Works really well in practice.
    """
    top_k = top_k or int(os.getenv("TOP_K", 5))

    if store.size == 0:
        return []

    chunks = store.chunks
    corpus = [c.text for c in chunks]

    # fetch more candidates than top_k so RRF has enough to work with
    candidate_k = min(top_k * 4, store.size)

    # --- semantic search ---
    semantic_results = store.search(query, top_k=candidate_k)
    # map chunk_id -> rank (0-indexed)
    semantic_ranks: dict[str, int] = {
        chunk.chunk_id: rank for rank, (chunk, _) in enumerate(semantic_results)
    }

    # --- keyword search (BM25) ---
    bm25_index = BM25Index(corpus)
    bm25_raw = bm25_index.search(query, top_k=candidate_k)
    # bm25_raw returns (doc_idx, score) pairs
    bm25_ranks: dict[str, int] = {
        chunks[doc_idx].chunk_id: rank
        for rank, (doc_idx, _) in enumerate(bm25_raw)
    }

    # --- RRF fusion ---
    # collect all chunk_ids that appear in either result set
    all_ids = set(semantic_ranks) | set(bm25_ranks)

    rrf_scores: dict[str, float] = {}
    for cid in all_ids:
        score = 0.0
        if cid in semantic_ranks:
            score += 1.0 / (RRF_K + semantic_ranks[cid])
        if cid in bm25_ranks:
            score += 1.0 / (RRF_K + bm25_ranks[cid])
        rrf_scores[cid] = score

    # sort by RRF score
    ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    # build a quick lookup from chunk_id -> Chunk object
    id_to_chunk = {c.chunk_id: c for c in chunks}

    results = []
    for cid in ranked_ids[:top_k]:
        chunk = id_to_chunk[cid]
        results.append((chunk, rrf_scores[cid]))

    return results
