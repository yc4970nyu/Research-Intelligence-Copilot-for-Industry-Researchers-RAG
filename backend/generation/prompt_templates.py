from backend.ingestion.chunker import Chunk
from backend.retrieval.intent import (
    INTENT_CHITCHAT,
    INTENT_FACTUAL,
    INTENT_SYNTHESIS,
    INTENT_STRUCTURED,
    INTENT_REFUSAL,
)


def build_context_block(chunks: list[tuple[Chunk, float]]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.
    Each chunk gets a citation number so the model can reference them.
    """
    parts = []
    for i, (chunk, score) in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Source: {chunk.filename}, page {chunk.page} (score={score:.3f})\n"
            f"{chunk.text.strip()}"
        )
    return "\n\n".join(parts)


def get_system_prompt(intent: str) -> str:
    """
    Return the system prompt for each intent type.
    Each template enforces grounded answering and citation.
    """
    base_citation_rule = (
        "When you use information from a source, cite it inline using [1], [2], etc. "
        "matching the source numbers in the context. "
        "Do not make up information that is not in the provided context. "
        "If the context does not contain enough information to answer, say so clearly."
    )

    if intent == INTENT_FACTUAL:
        return (
            "You are a research assistant helping industry researchers find information in technical papers. "
            "Answer the question using ONLY the information provided in the context below. "
            f"{base_citation_rule} "
            "Be concise and precise. Answer in 2-4 sentences unless more detail is needed."
        )

    elif intent == INTENT_SYNTHESIS:
        return (
            "You are a research assistant synthesizing information across multiple technical papers. "
            "Answer the question by drawing on ALL relevant sources provided in the context. "
            f"{base_citation_rule} "
            "Structure your answer to clearly show where different papers agree or differ."
        )

    elif intent == INTENT_STRUCTURED:
        return (
            "You are a research assistant creating structured summaries from technical papers. "
            "Format your answer as a well-organized markdown table or list as appropriate. "
            f"{base_citation_rule} "
            "Use markdown formatting. For tables, include clear column headers."
        )

    elif intent == INTENT_CHITCHAT:
        return (
            "You are a friendly research assistant. "
            "Respond naturally to the user's message. "
            "If they seem to be asking about research documents, let them know they can upload PDFs and ask questions."
        )

    elif intent == INTENT_REFUSAL:
        # shouldn't reach generation for refusal, but just in case
        return "You must politely decline this request."

    else:
        return (
            "You are a research assistant. Answer based only on the provided context. "
            f"{base_citation_rule}"
        )


def build_user_message(
    query: str,
    intent: str,
    chunks: list[tuple[Chunk, float]],
) -> str:
    """
    Build the user message - either with context (for KB intents) or without (for chitchat).
    """
    if intent == INTENT_CHITCHAT:
        return query

    context = build_context_block(chunks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer (cite sources inline with [1], [2], etc.):"
    )


# canned responses that don't need an LLM call
REFUSAL_MESSAGE = (
    "I can't help with that. This system is designed for research questions about "
    "uploaded technical documents. Please avoid sharing personal information or "
    "asking for personal legal or medical advice."
)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find sufficient evidence in the uploaded documents to answer this question. "
    "The retrieved passages are not relevant enough to give a reliable answer. "
    "Please try rephrasing your question, or make sure the relevant documents have been uploaded."
)
