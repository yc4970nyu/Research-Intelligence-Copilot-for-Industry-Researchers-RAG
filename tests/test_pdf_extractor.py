"""
Tests for the PDF extraction module.

Run with:  python -m pytest tests/test_pdf_extractor.py -v
Or just:   python tests/test_pdf_extractor.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.ingestion.pdf_extractor import extract_text_from_pdf, is_reference_section, PageContent

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "Data", "sample_pdfs")

# all 20 expected PDFs grouped by category
EXPECTED_PDFS = {
    "foundations": [
        "attention_is_all_you_need.pdf",
        "bert.pdf",
        "chain_of_thought.pdf",
        "gpt3.pdf",
        "instructgpt.pdf",
        "t5.pdf",
    ],
    "long-context-efficiency": [
        "flashattention.pdf",
        "longrope.pdf",
    ],
    "model-reports": [
        "claude_3_model_card.pdf",
        "gemini_1_5.pdf",
        "llama1.pdf",
        "llama2.pdf",
        "llama3.pdf",
        "mistral_7b.pdf",
        "mixtral.pdf",
        "phi3.pdf",
        "qwen2_5.pdf",
    ],
    "rag": [
        "rag_original.pdf",
        "rag_survey.pdf",
        "self_rag.pdf",
    ],
}


def get_all_pdf_paths():
    paths = []
    for category, filenames in EXPECTED_PDFS.items():
        for fname in filenames:
            paths.append((category, fname, os.path.join(SAMPLE_DIR, category, fname)))
    return paths


# ------------------------------------------------------------------ #
#  checkpoint 1: all expected PDF files actually exist in the repo
# ------------------------------------------------------------------ #
def test_all_pdfs_exist():
    print("\n[CHECK] All 20 PDFs exist on disk...")
    missing = []
    for category, fname, path in get_all_pdf_paths():
        if not os.path.isfile(path):
            missing.append(f"{category}/{fname}")

    if missing:
        raise AssertionError(f"Missing PDFs:\n" + "\n".join(f"  - {m}" for m in missing))

    print(f"  PASS - all {sum(len(v) for v in EXPECTED_PDFS.values())} PDFs found")


# ------------------------------------------------------------------ #
#  checkpoint 2: extraction runs without exceptions on every PDF
# ------------------------------------------------------------------ #
def test_extraction_no_errors():
    print("\n[CHECK] Extraction runs without errors on all PDFs...")
    errors = []
    for category, fname, path in get_all_pdf_paths():
        try:
            pages = extract_text_from_pdf(path)
            assert isinstance(pages, list), "should return a list"
        except Exception as e:
            errors.append(f"{category}/{fname}: {e}")

    if errors:
        raise AssertionError("Extraction failed on:\n" + "\n".join(f"  - {e}" for e in errors))

    print("  PASS - no extraction errors")


# ------------------------------------------------------------------ #
#  checkpoint 3: each PDF yields at least 1 page of content
# ------------------------------------------------------------------ #
def test_extraction_yields_content():
    print("\n[CHECK] Each PDF yields at least 1 page of non-empty text...")
    empty_results = []
    for category, fname, path in get_all_pdf_paths():
        pages = extract_text_from_pdf(path)
        if len(pages) == 0:
            empty_results.append(f"{category}/{fname}")

    if empty_results:
        raise AssertionError("Got zero pages from:\n" + "\n".join(f"  - {f}" for f in empty_results))

    print("  PASS - all PDFs yielded content")


# ------------------------------------------------------------------ #
#  checkpoint 4: no page has empty or whitespace-only text
# ------------------------------------------------------------------ #
def test_no_empty_pages_in_output():
    print("\n[CHECK] No extracted page has empty text...")
    bad = []
    for category, fname, path in get_all_pdf_paths():
        pages = extract_text_from_pdf(path)
        for p in pages:
            if not p.text or not p.text.strip():
                bad.append(f"{category}/{fname} page {p.page_num}")

    if bad:
        raise AssertionError("Empty page text found:\n" + "\n".join(f"  - {b}" for b in bad))

    print("  PASS - no empty pages")


# ------------------------------------------------------------------ #
#  checkpoint 5: page numbers are positive integers, in order
# ------------------------------------------------------------------ #
def test_page_numbers_are_valid():
    print("\n[CHECK] Page numbers are positive and ascending...")
    bad = []
    for category, fname, path in get_all_pdf_paths():
        pages = extract_text_from_pdf(path)
        for i, p in enumerate(pages):
            if not isinstance(p.page_num, int) or p.page_num < 1:
                bad.append(f"{category}/{fname}: invalid page_num={p.page_num}")
            if i > 0 and p.page_num <= pages[i - 1].page_num:
                bad.append(
                    f"{category}/{fname}: page order broken at index {i} "
                    f"({pages[i-1].page_num} -> {p.page_num})"
                )

    if bad:
        raise AssertionError("Page number issues:\n" + "\n".join(f"  - {b}" for b in bad))

    print("  PASS - page numbers valid")


# ------------------------------------------------------------------ #
#  checkpoint 6: reference section detection spot check
# ------------------------------------------------------------------ #
def test_reference_section_detection():
    print("\n[CHECK] Reference section detection spot check...")

    # mistral_7b last page is a known reference page
    mistral_path = os.path.join(SAMPLE_DIR, "model-reports", "mistral_7b.pdf")
    pages = extract_text_from_pdf(mistral_path)
    last = pages[-1].text
    assert is_reference_section(last), "Expected last page of mistral_7b to be detected as reference section"

    # first page of attention paper should NOT be reference section
    attn_path = os.path.join(SAMPLE_DIR, "foundations", "attention_is_all_you_need.pdf")
    pages2 = extract_text_from_pdf(attn_path)
    first = pages2[0].text
    assert not is_reference_section(first), "First page of attention paper should not be a reference section"

    print("  PASS - reference detection works")


# ------------------------------------------------------------------ #
#  summary report (also useful when running as plain script)
# ------------------------------------------------------------------ #
def run_all():
    tests = [
        test_all_pdfs_exist,
        test_extraction_no_errors,
        test_extraction_yields_content,
        test_no_empty_pages_in_output,
        test_page_numbers_are_valid,
        test_reference_section_detection,
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
