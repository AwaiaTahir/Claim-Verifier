"""Explanation generation and final output formatting."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from config import (
    DECISION_RELATIVE_SCORE_THRESHOLD,
    EXPLANATION_PROMPT_TEMPLATE,
    NOT_ENOUGH_EVIDENCE,
    RETRIEVAL_CONFIDENCE_THRESHOLD,
    SOURCE_LOCAL,
    TOP_K_FINAL,
)
from modules.llm_client import generate


def _source_name(passage: dict[str, Any]) -> str:
    """Return a readable source label for a passage."""
    url = passage.get("url", "")
    if url:
        try:
            domain = urlparse(url).netloc.replace("www.", "")
            return domain or passage.get("source", "")
        except Exception:
            return passage.get("source", "")
    return passage.get("source", SOURCE_LOCAL)


def _build_evidence_text(passages: list[dict[str, Any]]) -> str:
    """Build a numbered evidence block for the explanation prompt."""
    lines = []
    for index, passage in enumerate(passages[:3], start=1):
        source = _source_name(passage)
        text = passage.get("text", "")
        lines.append(f"{index}. [{source}] {text}")
    return "\n".join(lines)


def _call_ollama(prompt: str) -> str | None:
    """Send an explanation prompt to Ollama and return text, or None on failure."""
    return generate(prompt, task="explanation")


def _fallback_explanation(verdict_result: dict[str, Any], passages: list[dict[str, Any]]) -> str:
    """Create a short evidence-grounded fallback explanation."""
    if not passages:
        if verdict_result.get("verdict") == NOT_ENOUGH_EVIDENCE:
            return "The system did not retrieve enough relevant evidence to evaluate the claim."
        return "The system could not generate an explanation because no evidence passages were available."
    top_passage = passages[0]
    source = _source_name(top_passage)
    text = top_passage.get("text", "")[:200].strip()
    return f"Based on retrieved evidence from {source}: {text}..."


def generate_explanation(claim: str, verdict_result: dict[str, Any], passages: list[dict[str, Any]]) -> str:
    """Generate a concise evidence-grounded explanation for the final verdict."""
    evidence = _build_evidence_text(passages)
    prompt = EXPLANATION_PROMPT_TEMPLATE.format(
        verdict=verdict_result.get("verdict", NOT_ENOUGH_EVIDENCE),
        claim=claim,
        evidence=evidence,
    )
    response_text = _call_ollama(prompt)
    if not response_text:
        print("[Explainer] Fallback explanation used")
        return _fallback_explanation(verdict_result, passages)
    print("[Explainer] Ollama explanation used")
    return response_text


def _display_passages(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select passages worth displaying while avoiding zero-score filler evidence."""
    if not passages:
        return []
    max_score = max(float(passage.get("reranker_score", 0.0)) for passage in passages)
    minimum_score = max(
        RETRIEVAL_CONFIDENCE_THRESHOLD,
        max_score * DECISION_RELATIVE_SCORE_THRESHOLD,
    )
    filtered = [
        passage
        for passage in passages
        if float(passage.get("reranker_score", 0.0)) >= minimum_score
    ]
    return filtered or passages[:1]


def format_output(
    claim: str,
    verdict_result: dict[str, Any],
    passages: list[dict[str, Any]],
    explanation: str,
) -> dict[str, Any]:
    """Format the claim, verdict, explanation, and top evidence for UI and evaluation."""
    evidence = []
    for rank, passage in enumerate(_display_passages(passages)[:TOP_K_FINAL], start=1):
        evidence.append(
            {
                "rank": rank,
                "text": passage.get("text", ""),
                "source": _source_name(passage),
                "url": passage.get("url", ""),
                "stance": passage.get("stance", ""),
                "relevance_score": round(float(passage.get("reranker_score", 0.0)), 4),
                "credibility_weight": passage.get("credibility_weight", 1.0),
                "credibility_label": passage.get("credibility_label", "standard"),
            }
        )

    return {
        "claim": claim,
        "verdict": verdict_result.get("verdict", NOT_ENOUGH_EVIDENCE),
        "confidence": round(float(verdict_result.get("confidence", 0.0)), 2),
        "explanation": explanation,
        "evidence": evidence,
        "queries_used": verdict_result.get("queries_used", []),
        "claim_type": verdict_result.get("claim_type", ""),
        "processing_time_seconds": verdict_result.get("processing_time_seconds", 0.0),
        "reason": verdict_result.get("reason", ""),
        "supports_count": verdict_result.get("supports_count", 0),
        "refutes_count": verdict_result.get("refutes_count", 0),
        "neutral_count": verdict_result.get("neutral_count", 0),
        "is_numerical": verdict_result.get("is_numerical", False),
    }
