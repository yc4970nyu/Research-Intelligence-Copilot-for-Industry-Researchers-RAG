import re
from dataclasses import dataclass

from backend.ingestion.chunker import Chunk
from backend.retrieval.intent import INTENT_CHITCHAT, INTENT_REFUSAL, needs_search
from backend.generation.claude_client import call_claude, is_api_available
from backend.generation.prompt_templates import (
    get_system_prompt,
    build_user_message,
    REFUSAL_MESSAGE,
    INSUFFICIENT_EVIDENCE_MESSAGE,
)


@dataclass
class GenerationResult:
    answer: str
    sufficient_evidence: bool
    citations: list[tuple[Chunk, float]]
    hallucination_flagged: bool = False


def generate(
    query: str,
    intent: str,
    chunks: list[tuple[Chunk, float]],
    sufficient_evidence: bool,
) -> GenerationResult:
    """
    Main generation function. Handles all intent types and evidence states.

    Flow:
      refusal intent     -> return canned refusal, no LLM call
      chitchat           -> call LLM with no context
      insufficient evidence -> return canned message, no LLM call
      factual/synthesis/structured -> call LLM with context, run hallucination check
    """
    # refusal - don't even call the LLM
    if intent == INTENT_REFUSAL:
        return GenerationResult(
            answer=REFUSAL_MESSAGE,
            sufficient_evidence=False,
            citations=[],
        )

    # chitchat - call LLM but no KB context needed
    if intent == INTENT_CHITCHAT:
        answer = _call_llm(query, intent, [])
        return GenerationResult(
            answer=answer,
            sufficient_evidence=True,
            citations=[],
        )

    # knowledge base query but evidence too weak
    if not sufficient_evidence:
        return GenerationResult(
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            sufficient_evidence=False,
            citations=[],
        )

    # main path: generate answer grounded in retrieved chunks
    answer = _call_llm(query, intent, chunks)

    # post-hoc hallucination check
    flagged = _hallucination_check(answer, chunks)

    return GenerationResult(
        answer=answer,
        sufficient_evidence=True,
        citations=chunks,
        hallucination_flagged=flagged,
    )


def _call_llm(
    query: str,
    intent: str,
    chunks: list[tuple[Chunk, float]],
) -> str:
    """
    Call Claude if available, otherwise return a fallback message.
    The fallback is honest about what happened - better than returning garbage.
    """
    if not is_api_available():
        return _llm_unavailable_fallback(query, intent, chunks)

    system = get_system_prompt(intent)
    user_msg = build_user_message(query, intent, chunks)

    try:
        return call_claude(
            system=system,
            user=user_msg,
            max_tokens=1024,
            temperature=0.1,  # slight warmth for more natural answers
        )
    except Exception as e:
        # if the API call fails mid-request, don't crash the whole pipeline
        return f"[Generation error: {str(e)[:120]}]"


def _llm_unavailable_fallback(
    query: str,
    intent: str,
    chunks: list[tuple[Chunk, float]],
) -> str:
    """
    When no LLM is available, return the raw top chunks so the user still
    gets something useful. Not a great answer but shows the retrieval is working.
    """
    if intent == INTENT_CHITCHAT:
        return (
            "Hello! I'm a research assistant. You can upload PDF papers and ask me questions about them. "
            "(Note: LLM generation is currently unavailable - please configure a valid API key.)"
        )

    if not chunks:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    top_chunks = chunks[:3]
    parts = ["[LLM unavailable - showing raw retrieved passages]\n"]
    for i, (chunk, score) in enumerate(top_chunks, 1):
        parts.append(f"[{i}] {chunk.filename}, page {chunk.page}:\n{chunk.text.strip()[:300]}")

    return "\n\n".join(parts)


def _hallucination_check(answer: str, chunks: list[tuple[Chunk, float]]) -> bool:
    """
    Simple post-hoc hallucination filter.

    Splits the answer into sentences and checks if each sentence that makes
    a specific claim can be traced back to at least one retrieved chunk.

    This is a heuristic - not perfect. The main thing it catches is when
    the model starts talking about something not in the context at all.

    Returns True if potential hallucination detected (flagged), False if clean.
    """
    if not chunks:
        return False

    # combine all chunk text into one big context string for lookup
    full_context = " ".join(c.text.lower() for c, _ in chunks)

    sentences = _split_sentences(answer)
    flagged_count = 0

    for sent in sentences:
        sent_lower = sent.lower().strip()

        # skip short sentences, citations, and metadata lines
        if len(sent_lower) < 40:
            continue
        if re.search(r"\[\d+\]", sent):
            continue

        # extract the "content words" - nouns, numbers, proper nouns
        content_words = _extract_content_words(sent_lower)
        if not content_words:
            continue

        # if less than half the content words appear anywhere in the context, flag it
        found = sum(1 for w in content_words if w in full_context)
        if len(content_words) > 2 and found / len(content_words) < 0.4:
            flagged_count += 1

    # flag if more than 1 sentence seems unsupported
    return flagged_count > 1


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on period/exclamation/question mark."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _extract_content_words(text: str) -> list[str]:
    """
    Extract meaningful words (length > 4, not stopwords) for the hallucination check.
    We don't want to flag on articles and prepositions.
    """
    stopwords = {
        "this", "that", "with", "from", "they", "their", "have", "been",
        "which", "about", "also", "more", "into", "than", "when", "where",
        "some", "such", "each", "these", "those", "other", "while", "after",
        "paper", "model", "method", "approach", "show", "shows", "shown",
        "using", "used", "based", "results", "table", "figure",
    }
    words = re.findall(r"[a-z]{5,}", text)
    return [w for w in words if w not in stopwords]
