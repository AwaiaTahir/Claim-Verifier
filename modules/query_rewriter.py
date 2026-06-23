"""Query rewriting for neutral evidence retrieval."""

from __future__ import annotations

import json
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from config import (
    QUERY_REWRITE_PROMPT_TEMPLATE,
    STOPWORD_LANGUAGE,
)
from modules.llm_client import generate


def _call_ollama(prompt: str) -> str | None:
    """Send a query rewriting prompt to Ollama and return text, or None on failure."""
    return generate(prompt, task="query rewriting")


def _extract_json_array(text: str) -> list[str] | None:
    """Parse a JSON array of strings from an Ollama response."""
    try:
        payload = json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]", text or "", re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return None
    if not isinstance(payload, list):
        return None
    queries = [str(item).strip() for item in payload if str(item).strip()]
    return queries or None


def _content_words(claim: str) -> list[str]:
    """Extract content words from a claim for fallback query generation."""
    try:
        stop_words = set(stopwords.words(STOPWORD_LANGUAGE))
    except Exception:
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on"}
    try:
        tokens = word_tokenize(claim)
    except Exception:
        tokens = re.findall(r"[A-Za-z0-9]+", claim)
    words = []
    for token in tokens:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", token).lower()
        if cleaned and cleaned not in stop_words and len(cleaned) > 2:
            words.append(cleaned)
    return words


def _fallback_queries(claim: str) -> list[str]:
    """Build three deterministic fallback queries when Ollama is unavailable."""
    words = _content_words(claim)
    keyword_query = " ".join(words[:5]) or claim
    return [
        claim,
        keyword_query,
        f"{keyword_query} evidence research",
    ]


def _normalize_queries(queries: list[str] | None, claim: str) -> list[str]:
    """Return exactly three usable query strings."""
    if not queries:
        return _fallback_queries(claim)
    cleaned = []
    seen = set()
    for query in queries:
        value = re.sub(r"\s+", " ", query).strip(" ?\n\t")
        key = value.lower()
        if value and key not in seen:
            cleaned.append(value)
            seen.add(key)
        if len(cleaned) == 3:
            break
    while len(cleaned) < 3:
        for query in _fallback_queries(claim):
            if query.lower() not in seen:
                cleaned.append(query)
                seen.add(query.lower())
            if len(cleaned) == 3:
                break
    return cleaned[:3]


def rewrite_to_queries(claim: str) -> list[str]:
    """Rewrite a claim into three neutral search queries for retrieval."""
    prompt = QUERY_REWRITE_PROMPT_TEMPLATE.format(claim=claim)
    response_text = _call_ollama(prompt)
    if response_text is None:
        queries = _fallback_queries(claim)
        print(f"[Query Rewriter] Fallback queries: {queries}")
        return queries
    queries = _extract_json_array(response_text)
    normalized = _normalize_queries(queries, claim)
    print(f"[Query Rewriter] Queries: {normalized}")
    return normalized
