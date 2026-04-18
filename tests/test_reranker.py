"""
Tests for the reranker and evidence threshold logic.

Run with:  python tests/test_reranker.py
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.ingestion.chunker import Chunk
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.hybrid import hybrid_search
from backend.retrieval.reranker import rerank, check_evidence, filter_by_threshold

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "Data", "sample_pdfs")


def _make_store_from_pdfs(pdf_map: dict) -> VectorStore:
    from backend.ingestion.pdf_extractor import extract_text_from_pdf
    from backend.ingestion.chunker import chunk_pages
    store = VectorStore()
    for rel_path, doc_id in pdf_map.items():
        path = os.path.join(SAMPLE_DIR, rel_path)
        pages = extract_text_from_pdf(path)
        chunks = chunk_pages(pages, filename=os.path.basename(rel_path), doc_id=doc_id)
        store.add_chunks(chunks)
    return store


def test_rerank_returns_sorted_scores():
    print("\n[CHECK] Reranker returns results sorted by score descending...")
    store = _make_store_from_pdfs({
        "foundations/attention_is_all_you_need.pdf": "attn",
    })
    query = "self-attention mechanism transformer"
    candidates = hybrid_search(query, store, top_k=10)
    reranked = rerank(query, candidates, top_k=5)

    assert len(reranked) > 0
    scores = [s for _, s in reranked]
    assert scores == sorted(scores, reverse=True), "Scores not descending"
    print(f"  PASS - {len(reranked)} results, top score={scores[0]:.4f}")


def test_rerank_top_k_respected():
    print("\n[CHECK] Reranker respects top_k limit...")
    store = _make_store_from_pdfs({
        "foundations/attention_is_all_you_need.pdf": "attn",
    })
    query = "attention mechanism"
    candidates = hybrid_search(query, store, top_k=20)
    reranked = rerank(query, candidates, top_k=3)
    assert len(reranked) <= 3
    print(f"  PASS - got {len(reranked)} results (top_k=3)")


def test_rerank_empty_candidates():
    print("\n[CHECK] Reranker handles empty candidate list...")
    result = rerank("any query", [], top_k=5)
    assert result == []
    print("  PASS - empty input returns []")


def test_sufficient_evidence_relevant_query():
    print("\n[CHECK] check_evidence returns True for a clearly relevant query...")
    store = _make_store_from_pdfs({
        "foundations/attention_is_all_you_need.pdf": "attn",
        "rag/rag_original.pdf": "rag",
    })
    query = "multi-head attention query key value projection matrices"
    candidates = hybrid_search(query, store, top_k=10)
    reranked = rerank(query, candidates, top_k=5)
    assert check_evidence(reranked), f"Expected True, top score={reranked[0][1]:.4f}"
    print(f"  PASS - sufficient evidence, top score={reranked[0][1]:.4f}")


def test_insufficient_evidence_off_topic():
    print("\n[CHECK] check_evidence returns False for completely off-topic query...")
    store = _make_store_from_pdfs({
        "foundations/attention_is_all_you_need.pdf": "attn",
        "rag/rag_original.pdf": "rag",
    })
    query = "best pizza recipe with mozzarella and tomato sauce"
    candidates = hybrid_search(query, store, top_k=10)
    reranked = rerank(query, candidates, top_k=5)
    assert not check_evidence(reranked), f"Expected False, top score={reranked[0][1]:.4f}"
    print(f"  PASS - insufficient evidence, top score={reranked[0][1]:.4f}")


def test_filter_by_threshold_removes_weak():
    print("\n[CHECK] filter_by_threshold removes low-score chunks...")
    dummy_chunk = Chunk(
        chunk_id=str(uuid.uuid4()),
        doc_id="d1",
        filename="x.pdf",
        page=1,
        text="some text",
    )
    results = [
        (dummy_chunk, 0.80),
        (dummy_chunk, 0.70),
        (dummy_chunk, 0.40),  # below threshold * 0.8
        (dummy_chunk, 0.30),  # below threshold * 0.8
    ]
    filtered = filter_by_threshold(results)
    # threshold=0.65, cutoff=0.65*0.8=0.52, so 0.80 and 0.70 pass, 0.40 and 0.30 don't
    assert len(filtered) == 2, f"Expected 2 results, got {len(filtered)}"
    print(f"  PASS - kept {len(filtered)}/4 chunks above cutoff")


def test_rerank_improves_over_hybrid():
    print("\n[CHECK] Reranker improves semantic precision over raw hybrid order...")
    store = _make_store_from_pdfs({
        "foundations/attention_is_all_you_need.pdf": "attn",
        "rag/rag_original.pdf": "rag",
    })
    query = "retrieval augmented generation dense passage retrieval"

    candidates = hybrid_search(query, store, top_k=10)
    reranked = rerank(query, candidates, top_k=5)

    # top result after reranking should be from the rag paper
    top_doc = reranked[0][0].doc_id
    assert top_doc == "rag", f"Expected rag paper on top after reranking, got {top_doc}"
    print(f"  PASS - top doc after reranking: {top_doc}, score={reranked[0][1]:.4f}")


def run_all():
    tests = [
        test_rerank_returns_sorted_scores,
        test_rerank_top_k_respected,
        test_rerank_empty_candidates,
        test_sufficient_evidence_relevant_query,
        test_insufficient_evidence_off_topic,
        test_filter_by_threshold_removes_weak,
        test_rerank_improves_over_hybrid,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
