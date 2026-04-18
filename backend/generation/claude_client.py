import os
import httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_CHAT_MODEL = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

# cached result so we don't ping the API on every request just to check availability
_api_available: Optional[bool] = None


def is_api_available() -> bool:
    """
    Check if the Anthropic API key is set and working.
    Caches the result after the first successful call.
    """
    global _api_available
    if _api_available is not None:
        return _api_available
    if not ANTHROPIC_API_KEY:
        _api_available = False
        return False
    # we'll assume it's available if the key exists and let call_claude handle errors
    _api_available = True
    return True


def call_claude(
    system: str,
    user: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """
    Simple wrapper around the Anthropic messages API.
    Returns the text content of the first response block.

    temperature=0 by default because for intent detection and query rewriting
    we want deterministic outputs, not creative ones.
    """
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": ANTHROPIC_CHAT_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(ANTHROPIC_URL, json=payload, headers=headers)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Anthropic API error {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    return data["content"][0]["text"].strip()
