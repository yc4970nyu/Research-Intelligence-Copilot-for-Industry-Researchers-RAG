import os
import uuid
from dataclasses import dataclass

from backend.ingestion.pdf_extractor import PageContent, is_reference_section


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    text: str


def chunk_pages(
    pages: list[PageContent],
    filename: str,
    doc_id: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[Chunk]:
    """
    Turn a list of pages into overlapping text chunks.

    chunk_size and chunk_overlap are in characters (not tokens).
    I chose characters instead of tokens because we don't have a tokenizer
    running locally and it's simpler. The mistral-embed model handles
    up to 8192 tokens, so 512 chars is very safe.

    The overlap helps avoid cutting a sentence in half right at a boundary,
    which would hurt retrieval a lot.
    """
    chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", 512))
    chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 64))

    chunks = []

    for page in pages:
        # skip reference pages - they don't help with answering questions
        if is_reference_section(page.text):
            continue

        page_chunks = _split_text(page.text, chunk_size, chunk_overlap)

        for text in page_chunks:
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    filename=filename,
                    page=page.page_num,
                    text=text.strip(),
                )
            )

    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Sliding window over the text. I try to break at sentence boundaries
    instead of just at a fixed character count - cutting mid-sentence
    kills the semantic meaning of both chunks.

    The idea: advance by (chunk_size - overlap) each step, then look
    forward a bit to find a good split point like ". " or "\n".
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # try to find a clean break point near the end of this window
            end = _find_break_point(text, end, lookahead=80)

        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)

        start += step

        # if the remaining text is shorter than chunk_size, just grab it all
        if start < len(text) and len(text) - start < chunk_size // 2:
            tail = text[start:]
            if tail.strip():
                chunks.append(tail)
            break

    return chunks


def _find_break_point(text: str, pos: int, lookahead: int = 80) -> int:
    """
    From position `pos`, scan forward up to `lookahead` chars to find
    a sentence or paragraph boundary. This way chunks end at natural
    stopping points.

    Priority: paragraph break > sentence end > word boundary
    """
    window = text[pos: pos + lookahead]

    # paragraph break is the best split point
    para_idx = window.find("\n\n")
    if para_idx != -1:
        return pos + para_idx + 2

    # sentence end - period followed by space or newline
    for i, ch in enumerate(window):
        if ch in ".!?" and i + 1 < len(window) and window[i + 1] in " \n":
            return pos + i + 2

    # at least try to not cut mid-word
    space_idx = window.rfind(" ")
    if space_idx != -1:
        return pos + space_idx + 1

    # no good break point found, just cut here
    return pos
