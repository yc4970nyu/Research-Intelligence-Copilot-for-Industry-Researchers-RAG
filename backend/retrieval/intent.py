import re
from backend.generation.claude_client import call_claude, is_api_available

# intent labels - used here and in prompt_templates.py to pick generation template
INTENT_CHITCHAT = "chitchat"
INTENT_FACTUAL = "factual"
INTENT_SYNTHESIS = "synthesis"
INTENT_STRUCTURED = "structured"
INTENT_REFUSAL = "refusal"

ALL_INTENTS = {INTENT_CHITCHAT, INTENT_FACTUAL, INTENT_SYNTHESIS, INTENT_STRUCTURED, INTENT_REFUSAL}

# intents that require a knowledge base search
SEARCH_INTENTS = {INTENT_FACTUAL, INTENT_SYNTHESIS, INTENT_STRUCTURED}

_INTENT_SYSTEM = """You are an intent classifier for a research document assistant.

Classify the user query into exactly one of these categories:
- chitchat: greetings, small talk, or questions unrelated to research documents (e.g. "hello", "how are you", "what's the weather")
- factual: a specific factual question that can be answered from research documents (e.g. "what dataset did this paper use", "what is the BLEU score")
- synthesis: requires synthesizing or comparing information across multiple documents (e.g. "which papers focus on RAG", "compare llama and mistral")
- structured: explicitly asks for structured output like a table, list, or comparison (e.g. "make a table comparing", "list all models mentioned")
- refusal: contains PII, requests personal legal/medical advice, or is clearly harmful

Reply with ONLY the category name, nothing else."""


def detect_intent(query: str) -> str:
    """
    Classify query intent. Tries Claude first, falls back to rule-based if API unavailable.
    Returns one of the INTENT_* constants.
    """
    if is_api_available():
        try:
            response = call_claude(
                system=_INTENT_SYSTEM,
                user=query,
                max_tokens=20,
                temperature=0.0,
            )
            intent = response.strip().lower().rstrip(".,;:")
            if intent in ALL_INTENTS:
                return intent
        except Exception:
            pass

    # rule-based fallback when LLM is not available
    return _rule_based_intent(query)


def _rule_based_intent(query: str) -> str:
    """
    Heuristic intent classification. Good enough for demos when the LLM API is down.
    Not perfect but covers the obvious cases well.
    """
    q = query.lower().strip()

    # refusal: PII patterns or personal advice requests
    pii_patterns = [
        r"\b(ssn|social security|passport|credit card|password|my address|my phone)\b",
        r"\bshould i (take|use|buy|invest|see a doctor)\b",
        r"\bam i (sick|pregnant|dying|at risk)\b",
    ]
    if any(re.search(p, q) for p in pii_patterns):
        return INTENT_REFUSAL

    # chitchat: greetings and meta questions
    chitchat_patterns = [
        r"^(hi|hello|hey|howdy|good (morning|afternoon|evening))\b",
        r"^(how are you|what's up|sup|how's it going)\b",
        r"^(thank(s| you)|bye|goodbye|see you)\b",
        r"^(what can you do|who are you|what are you)\b",
        r"^(nice to meet|pleased to meet)\b",
    ]
    if any(re.search(p, q) for p in chitchat_patterns):
        return INTENT_CHITCHAT

    # structured: wants a table, list, or comparison output
    structured_keywords = ["table", "list all", "compare", "versus", " vs ", "summarize all", "enumerate"]
    if any(kw in q for kw in structured_keywords):
        return INTENT_STRUCTURED

    # synthesis: cross-document questions
    synthesis_keywords = ["which papers", "across papers", "all documents", "each paper", "these papers"]
    if any(kw in q for kw in synthesis_keywords):
        return INTENT_SYNTHESIS

    # default: treat as a factual question
    return INTENT_FACTUAL


def needs_search(intent: str) -> bool:
    """True if this intent should trigger a knowledge base search."""
    return intent in SEARCH_INTENTS


_REWRITE_SYSTEM = """You are a query optimizer for a semantic search system over research papers.

Rewrite the user's question into a better search query:
- Expand abbreviations (LLM -> large language model, RAG -> retrieval augmented generation)
- Remove filler words ("can you tell me", "I want to know")
- Add relevant technical context if the query is vague
- Keep it to 1-2 sentences

Reply with ONLY the rewritten query, no explanation."""

# common abbreviations to expand even without the LLM
_ABBREV = {
    r"\bllm\b": "large language model",
    r"\bllms\b": "large language models",
    r"\brag\b": "retrieval augmented generation",
    r"\bnlp\b": "natural language processing",
    r"\bsft\b": "supervised fine-tuning",
    r"\brlhf\b": "reinforcement learning from human feedback",
    r"\bpeft\b": "parameter efficient fine-tuning",
    r"\bmoe\b": "mixture of experts",
    r"\bkv\b": "key value",
    r"\bvram\b": "GPU memory",
}

_FILLER = re.compile(
    r"^(can you (tell me|explain|describe)|i want to know|what (is|are|was|were) the|please explain)\s+",
    re.IGNORECASE,
)


def rewrite_query(query: str) -> str:
    """
    Rewrite query for better retrieval. Tries Claude, falls back to rule-based expansion.
    """
    if is_api_available():
        try:
            rewritten = call_claude(
                system=_REWRITE_SYSTEM,
                user=query,
                max_tokens=100,
                temperature=0.0,
            )
            if 0 < len(rewritten) < 500:
                return rewritten
        except Exception:
            pass

    return _rule_based_rewrite(query)


def _rule_based_rewrite(query: str) -> str:
    """
    Simple query cleanup: remove filler phrases and expand common abbreviations.
    Not as good as LLM rewriting but better than nothing.
    """
    q = query.strip()
    q = _FILLER.sub("", q)
    q_lower = q.lower()
    for pattern, expansion in _ABBREV.items():
        q_lower = re.sub(pattern, expansion, q_lower)
    # preserve original casing for the rest of the string
    return q_lower.strip()
