import re
import fitz  # pymupdf
from dataclasses import dataclass


@dataclass
class PageContent:
    page_num: int
    text: str


def extract_text_from_pdf(filepath: str) -> list[PageContent]:
    """
    Extract text from a PDF file, one PageContent per page.

    I tried a few approaches here. The simplest is just doc[i].get_text()
    but that gives terrible results for two-column papers (which is basically
    every academic PDF). So I'm using the "blocks" mode and sorting them
    by x-position to handle multi-column layout.
    """
    doc = fitz.open(filepath)
    pages = []

    for i, page in enumerate(doc):
        text = _extract_page_text(page)
        text = _clean_text(text)

        # skip pages with almost no content, probably just figures or blank pages
        if len(text.strip()) < 30:
            continue

        pages.append(PageContent(page_num=i + 1, text=text))

    doc.close()
    return pages


def _extract_page_text(page: fitz.Page) -> str:
    """
    Handle multi-column layout by splitting the page into left/right halves
    and extracting each separately, then combining in reading order.

    Most ML papers use two-column format so this matters a lot.
    The threshold I use is the midpoint of the page width - not perfect
    but works well enough for standard two-column academic PDFs.
    """
    page_width = page.rect.width
    mid_x = page_width / 2

    blocks = page.get_text("blocks")  # returns (x0, y0, x1, y1, text, block_no, block_type)

    # separate text blocks into left and right columns
    left_blocks = []
    right_blocks = []

    for block in blocks:
        x0, y0, x1, y1, text, *_ = block
        if not text.strip():
            continue
        # if the block center is on the left side, it's left column
        center_x = (x0 + x1) / 2
        if center_x < mid_x:
            left_blocks.append((y0, text))
        else:
            right_blocks.append((y0, text))

    # sort each column top to bottom
    left_blocks.sort(key=lambda b: b[0])
    right_blocks.sort(key=lambda b: b[0])

    combined = [text for _, text in left_blocks] + [text for _, text in right_blocks]
    return "\n".join(combined)


def _clean_text(text: str) -> str:
    """
    Remove common noise from academic PDFs.

    Things I noticed when testing on the sample papers:
    - page numbers sitting alone on a line
    - "arXiv:xxxx.xxxxx" headers
    - excessive whitespace / newlines
    - hyphenated line breaks like "atten-\ntion" -> "attention"
    - lines that are just a number (page numbers, figure numbers)
    """
    # fix hyphenated line breaks first - common in two-column papers
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # remove lines that are just a standalone number (page numbers etc)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # skip pure number lines and very short lines that are probably artifacts
        if re.match(r"^\d+$", stripped):
            continue
        # skip arxiv id lines
        if re.match(r"^arXiv:\d{4}\.\d+", stripped):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)

    # collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def is_reference_section(text: str) -> bool:
    """
    Check if a block of text is likely the references section.
    I want to skip or down-weight reference sections because they
    just add noise to retrieval - nobody asks "what papers are cited".

    Simple heuristic: if there are a lot of lines starting with [number]
    it's probably a reference list.
    """
    lines = text.strip().split("\n")
    ref_lines = sum(1 for l in lines if re.match(r"^\[\d+\]", l.strip()))
    return ref_lines > 5
