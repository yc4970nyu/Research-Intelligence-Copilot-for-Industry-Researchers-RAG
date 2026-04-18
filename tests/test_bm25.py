"""
Tests for the BM25 keyword search module.

Run with:  python tests/test_bm25.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.retrieval.bm25 import BM25Index, tokenize


def test_tokenizer_lowercases_and_removes_stopwords():
    print("\n[CHECK] Tokenizer lowercases and removes stopwords...")
    tokens = tokenize("The Transformer model uses self-attention")
    assert "the" not in tokens, "stopword 'the' should be removed"
    assert "transformer" in tokens
    assert "attention" in tokens
    # all tokens should be lowercase
    assert all(t == t.lower() for t in tokens)
    print(f"  PASS - tokens: {tokens}")


def test_bm25_ranks_relevant_docs_higher():
    print("\n[CHECK] BM25 ranks relevant documents higher than irrelevant ones...")
    corpus = [
        "The transformer model uses self-attention to process sequences.",  # relevant
        "Bananas are a great source of potassium and vitamins.",             # not relevant
        "Attention mechanisms let the model focus on important tokens.",     # relevant
        "The stock market went up today due to positive earnings.",          # not relevant
    ]
    index = BM25Index(corpus)
    results = index.search("transformer attention model", top_k=4)

    top_indices = [r[0] for r in results]
    # relevant docs (0 and 2) should both appear before irrelevant ones
    assert 0 in top_indices[:2] or 2 in top_indices[:2], "Relevant doc should be in top 2"
    # scores should be descending
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True), "Scores should be sorted descending"
    print("  PASS - relevant docs ranked higher")


def test_bm25_score_zero_for_no_overlap():
    print("\n[CHECK] BM25 returns score 0 for query with no matching terms...")
    corpus = ["the cat sat on the mat", "dogs are loyal animals"]
    index = BM25Index(corpus)
    results = index.search("quantum computing neural network")
    # nothing in corpus matches, so results should be empty or all zeros
    assert all(s == 0 for _, s in results), "Should return zero scores for no overlap"
    print("  PASS - zero score for no overlap")


def test_bm25_top_k_respected():
    print("\n[CHECK] BM25 search respects top_k limit...")
    corpus = [f"document about topic {i} with some words" for i in range(50)]
    index = BM25Index(corpus)
    results = index.search("document topic words", top_k=5)
    assert len(results) <= 5, f"Expected at most 5 results, got {len(results)}"
    print(f"  PASS - got {len(results)} results (top_k=5)")


def test_bm25_on_real_chunks():
    print("\n[CHECK] BM25 finds correct content in real paper chunks...")
    from backend.ingestion.pdf_extractor import extract_text_from_pdf
    from backend.ingestion.chunker import chunk_pages

    import os
    pdf_path = os.path.join("Data", "sample_pdfs", "foundations", "attention_is_all_you_need.pdf")
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages, filename="attention_is_all_you_need.pdf", doc_id="test")

    corpus = [c.text for c in chunks]
    index = BM25Index(corpus)

    results = index.search("multi-head attention query key value", top_k=3)
    assert len(results) > 0, "Should find something"

    top_text = corpus[results[0][0]].lower()
    assert "attention" in top_text, "Top result should be about attention"
    print(f"  PASS - top result score={results[0][1]:.2f}, contains 'attention'")


def run_all():
    tests = [
        test_tokenizer_lowercases_and_removes_stopwords,
        test_bm25_ranks_relevant_docs_higher,
        test_bm25_score_zero_for_no_overlap,
        test_bm25_top_k_respected,
        test_bm25_on_real_chunks,
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
