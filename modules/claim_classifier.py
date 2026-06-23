"""Claim classification before retrieval."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from config import (
    CLAIM_AMBIGUOUS,
    CLAIM_COMPOUND,
    CLAIM_FACTUAL,
    CLAIM_OPINION,
    CLAIM_PREDICTION,
    CLAIM_TYPES,
    CLASSIFIER_PROMPT_TEMPLATE,
    DEFAULT_CLASSIFICATION,
    NOT_ENOUGH_EVIDENCE,
)
from modules.llm_client import generate

# Common English verbs and assertion markers — a factual claim must have at least one
_CLAIM_VERBS = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having",
    "does", "did", "do",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "cause", "causes", "caused", "causing",
    "make", "makes", "made", "making",
    "produce", "produces", "produced", "producing",
    "contain", "contains", "contained", "containing",
    "walked", "walks", "walk", "walking",
    "fell", "falls", "fall", "falling",
    "says", "said", "say", "saying",
    "shows", "showed", "show", "showing", "shown",
    "found", "finds", "find", "finding",
    "states", "stated", "state", "stating",
    "proves", "proved", "prove", "proving", "proven",
    "exists", "existed", "exist", "existing",
    "leads", "led", "lead", "leading",
    "created", "creates", "create", "creating",
    "discovered", "discovers", "discover",
    "invented", "invents", "invent",
    "built", "builds", "build", "building",
    "killed", "kills", "kill",
    "won", "wins", "win", "winning",
    "lost", "loses", "lose", "losing",
    "became", "becomes", "become", "becoming",
    "born", "died", "dies", "die", "lives", "live",
    "wrote", "writes", "write", "writing", "written",
    "known", "called", "named", "considered",
    "located", "founded", "established",
    "visible", "flat", "round", "true", "false",
}


def _is_gibberish_claim(claim: str) -> bool:
    """Detect nonsensical claims that lack verb structure or factual assertion."""
    words = re.findall(r"[a-zA-Z]+", claim.lower())
    if len(words) < 2:
        return True
    return not any(w in _CLAIM_VERBS for w in words)


def _default_classification() -> dict[str, Any]:
    """Return a fresh fallback classification dictionary."""
    return copy.deepcopy(DEFAULT_CLASSIFICATION)


def _call_ollama(prompt: str) -> str | None:
    """Send a prompt to Ollama and return the response text, or None on failure."""
    return generate(prompt, task="claim classification")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from a raw model response."""
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def _validate_classification(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize and validate an Ollama classification payload."""
    if not isinstance(payload, dict):
        return _default_classification()

    claim_type = str(payload.get("type", CLAIM_FACTUAL)).strip().lower()
    if claim_type not in CLAIM_TYPES:
        claim_type = CLAIM_FACTUAL

    sub_claims = payload.get("sub_claims", [])
    if not isinstance(sub_claims, list):
        sub_claims = []
    sub_claims = [str(item).strip() for item in sub_claims if str(item).strip()]

    is_checkable = bool(payload.get("is_checkable", True))
    if claim_type in {CLAIM_OPINION, CLAIM_PREDICTION}:
        is_checkable = False
    if claim_type in {CLAIM_FACTUAL, CLAIM_AMBIGUOUS, CLAIM_COMPOUND}:
        is_checkable = True

    message = str(payload.get("message", "") or "").strip()
    if not is_checkable and not message:
        message = "This input is not currently verifiable with evidence."

    return {
        "type": claim_type,
        "is_checkable": is_checkable,
        "sub_claims": sub_claims,
        "message": message,
    }


def classify_claim(claim: str) -> dict[str, Any]:
    """Classify a user claim into a checkability type before retrieval."""
    # Pre-check: gibberish claims without verbs are not verifiable
    if _is_gibberish_claim(claim):
        print(f"[Classifier] Gibberish detected (no verb/assertion): {claim[:60]}")
        return {
            "type": CLAIM_AMBIGUOUS,
            "is_checkable": False,
            "sub_claims": [],
            "message": "This does not appear to be a verifiable factual claim.",
        }

    prompt = CLASSIFIER_PROMPT_TEMPLATE.format(claim=claim)
    response_text = _call_ollama(prompt)
    if response_text is None:
        fallback = _default_classification()
        print(f"[Classifier] Fallback classification: {fallback['type']}")
        return fallback
    payload = _extract_json_object(response_text)
    classification = _validate_classification(payload)
    print(
        "[Classifier] "
        f"type={classification['type']} "
        f"checkable={classification['is_checkable']} "
        f"sub_claims={len(classification['sub_claims'])}"
    )
    return classification
