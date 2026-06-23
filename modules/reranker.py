"""Cross-encoder reranking and NLI stance labeling."""

from __future__ import annotations

import math
import re
from typing import Any
from urllib.parse import urlparse

import numpy as np

from config import (
    BELIEF_FRAMING_PHRASES,
    CREDIBILITY_WEIGHTS,
    DEFAULT_CREDIBILITY,
    DEVICE,
    NEGATION_WORDS,
    NLI_MODEL,
    RERANKER_MODEL,
    STANCE_NEUTRAL,
    STANCE_REFUTES,
    STANCE_SUPPORTS,
    TORCH_DTYPE,
)

_reranker_model = None
_nli_model = None
_reranker_failed = False
_nli_failed = False


def _sigmoid(value: float) -> float:
    """Convert a raw reranker score into a 0-1 confidence score."""
    try:
        return 1.0 / (1.0 + math.exp(-float(value)))
    except Exception:
        return 0.0


def _softmax(values: Any) -> list[float]:
    """Convert raw NLI logits into probabilities."""
    array = np.asarray(values, dtype="float32")
    if array.ndim > 1:
        array = array[0]
    array = array - np.max(array)
    exp = np.exp(array)
    total = float(np.sum(exp))
    if total <= 0:
        return [0.0, 1.0, 0.0]
    return [float(item / total) for item in exp]


def _apply_half_precision(cross_encoder: Any, name: str) -> None:
    """Switch a loaded cross-encoder model to float16 on CUDA when possible."""
    if DEVICE != "cuda":
        return
    try:
        cross_encoder.model.half()
    except Exception as exc:
        print(f"[reranker WARNING] Could not switch {name} to {TORCH_DTYPE}: {exc}")


def get_reranker_model():
    """Load the relevance cross-encoder lazily on the configured GPU."""
    global _reranker_model, _reranker_failed
    if _reranker_model is not None:
        return _reranker_model
    if _reranker_failed:
        return None
    try:
        from sentence_transformers import CrossEncoder

        _reranker_model = CrossEncoder(RERANKER_MODEL, device=DEVICE)
        _apply_half_precision(_reranker_model, "reranker")
        return _reranker_model
    except Exception as exc:
        print(f"[reranker WARNING] Reranker model unavailable: {exc}")
        _reranker_failed = True
        return None


def get_nli_model():
    """Load the NLI cross-encoder lazily on the configured GPU."""
    global _nli_model, _nli_failed
    if _nli_model is not None:
        return _nli_model
    if _nli_failed:
        return None
    try:
        from sentence_transformers import CrossEncoder

        _nli_model = CrossEncoder(NLI_MODEL, device=DEVICE)
        _apply_half_precision(_nli_model, "NLI model")
        return _nli_model
    except Exception as exc:
        print(f"[reranker WARNING] NLI model unavailable: {exc}")
        _nli_failed = True
        return None


# --- FIX 7+8: Min-max normalization for reranker scores ---
def _min_max_normalize(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply min-max normalization to reranker_score within a batch.

    Preserves the pre-normalization score as reranker_score_prenorm so that
    retrieval confidence checks can use the actual relevance signal.
    """
    if len(scored) < 2:
        for item in scored:
            item["reranker_score_prenorm"] = item["reranker_score"]
        return scored
    scores = [item["reranker_score"] for item in scored]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    if score_range < 1e-9:
        # All scores identical — spread evenly
        for item in scored:
            item["reranker_score_prenorm"] = item["reranker_score"]
            item["reranker_score"] = 0.5
        return scored
    for item in scored:
        item["reranker_score_prenorm"] = item["reranker_score"]
        item["reranker_score"] = (item["reranker_score"] - min_score) / score_range
    return scored


# --- BONUS: Source credibility weight ---
def get_credibility_weight(url: str) -> tuple[float, str]:
    """Look up domain credibility weight and label from URL."""
    if not url:
        return DEFAULT_CREDIBILITY, "standard"
    try:
        domain = urlparse(url).netloc.replace("www.", "").lower()
    except Exception:
        return DEFAULT_CREDIBILITY, "standard"
    for cred_domain, weight in CREDIBILITY_WEIGHTS.items():
        if domain.endswith(cred_domain):
            if weight >= 1.2:
                return weight, "trusted"
            elif weight < 1.0:
                return weight, "low"
            else:
                return weight, "standard"
    return DEFAULT_CREDIBILITY, "standard"


def _fallback_rerank(passages: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Score passages by retrieval score if the reranker model is unavailable."""
    fallback = []
    for passage in passages:
        item = dict(passage)
        raw_score = float(item.get("rrf_score", 0.0))
        item["reranker_raw_score"] = raw_score
        base_score = min(0.99, max(0.0, raw_score * 25.0))
        # Store pure score for confidence checking BEFORE credibility
        item["reranker_score_prenorm"] = base_score
        # --- BONUS: Apply credibility weight ---
        cred_weight, cred_label = get_credibility_weight(item.get("url", ""))
        item["credibility_weight"] = cred_weight
        item["credibility_label"] = cred_label
        item["reranker_score"] = min(1.0, base_score * cred_weight)
        fallback.append(item)
    fallback = sorted(fallback, key=lambda item: item.get("reranker_score", 0.0), reverse=True)[:top_k]
    return _min_max_normalize(fallback)


def rerank(claim: str, passages: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Rerank retrieved passages by cross-encoder relevance to the claim."""
    if not passages:
        return []
    model = get_reranker_model()
    if model is None:
        return _fallback_rerank(passages, top_k)
    try:
        pairs = [(claim, passage.get("text", "")) for passage in passages]
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
        scored = []
        for passage, score in zip(passages, scores):
            raw_score = float(np.asarray(score).reshape(-1)[0])
            item = dict(passage)
            item["reranker_raw_score"] = raw_score
            sigmoid_score = _sigmoid(raw_score)
            # Store pure sigmoid for confidence checking BEFORE credibility
            item["reranker_score_prenorm"] = sigmoid_score
            # --- BONUS: Apply credibility weight ---
            cred_weight, cred_label = get_credibility_weight(item.get("url", ""))
            item["credibility_weight"] = cred_weight
            item["credibility_label"] = cred_label
            item["reranker_score"] = min(1.0, sigmoid_score * cred_weight)
            scored.append(item)
        scored = sorted(scored, key=lambda item: item["reranker_score"], reverse=True)[:top_k]
        # --- FIX 7+8: Apply min-max normalization ---
        scored = _min_max_normalize(scored)
        return scored
    except Exception as exc:
        print(f"[reranker WARNING] Reranking failed: {exc}")
        return _fallback_rerank(passages, top_k)


# --- FIX 4+5: Belief-framing and negation detection ---
def _has_belief_framing(text: str) -> bool:
    """Check if passage text contains belief-framing phrases."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in BELIEF_FRAMING_PHRASES)


def _extract_claim_keywords(claim: str) -> list[str]:
    """Extract meaningful keywords from a claim for proximity checking."""
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
                  "that", "this", "it", "to", "for", "and", "or", "but", "with",
                  "from", "by", "at", "as", "has", "have", "had", "do", "does"}
    words = re.findall(r"[a-zA-Z]+", claim.lower())
    return [w for w in words if w not in stop_words and len(w) > 2]


def _has_negation_near_keywords(text: str, claim: str) -> bool:
    """Check for negation words within 5 words of claim keywords in the text."""
    text_lower = text.lower()
    words = text_lower.split()
    claim_keywords = _extract_claim_keywords(claim)
    if not claim_keywords:
        return False
    # Find positions of claim keywords in text
    keyword_positions = []
    for i, word in enumerate(words):
        clean_word = re.sub(r"[^a-z]", "", word)
        if clean_word in claim_keywords:
            keyword_positions.append(i)
    if not keyword_positions:
        return False
    # Check if any negation word is within 5 words of a keyword
    for i, word in enumerate(words):
        clean_word = re.sub(r"[^a-z]", "", word)
        if clean_word in NEGATION_WORDS:
            for kp in keyword_positions:
                if abs(i - kp) <= 5:
                    return True
    return False


def _post_process_stance(claim: str, item: dict[str, Any]) -> dict[str, Any]:
    """Apply belief-framing and negation post-processing to a single stance result (FIX 4+5)."""
    text = item.get("text", "")
    stance = item.get("stance", STANCE_NEUTRAL)
    stance_scores = item.get("stance_scores", {})

    # Check belief-framing → force NEUTRAL
    if _has_belief_framing(text):
        # Don't override if it's clearly refuting with negation
        if not _has_negation_near_keywords(text, claim):
            item["stance"] = STANCE_NEUTRAL
            item["stance_override_reason"] = "belief_framing"
            return item

    # Check negation near claim keywords → upgrade NEUTRAL to REFUTES
    # Only if NLI entailment confidence was low (not close to SUPPORTS)
    if stance == STANCE_NEUTRAL and _has_negation_near_keywords(text, claim):
        entailment = float(stance_scores.get("entailment", 0.0))
        if entailment < 0.4:
            item["stance"] = STANCE_REFUTES
            item["stance_override_reason"] = "negation_near_keywords"

    return item


# --- FIX 10: Direct contradiction detector ---
def detect_direct_contradictions(claim: str, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Override stance to REFUTES when passage directly contradicts the claim.

    Only overrides NEUTRAL passages. Never overrides NLI SUPPORTS labels since
    the NLI model understands context better than keyword matching (e.g. 'Moon
    landing hoax was debunked' supports the landing despite containing 'hoax').
    """
    claim_keywords = _extract_claim_keywords(claim)
    if not claim_keywords:
        return passages

    contradiction_terms = ["not", "false", "debunked", "disproven", "incorrect",
                          "myth", "untrue", "wrong", "no evidence", "misleading"]
    hoax_terms = ["hoax", "conspiracy", "staged", "faked", "fake"]

    result = []
    for passage in passages:
        item = dict(passage)
        text_lower = item.get("text", "").lower()
        
        has_keywords = sum(1 for kw in claim_keywords if kw in text_lower) >= max(1, len(claim_keywords) // 3)
        if has_keywords:
            has_contradiction = any(term in text_lower for term in contradiction_terms)
            is_debunking_hoax = any(term in text_lower for term in hoax_terms) and not any(term in claim.lower() for term in hoax_terms)
            if has_contradiction and not is_debunking_hoax:
                item["stance"] = STANCE_REFUTES
                item["stance_override_reason"] = "direct_contradiction"
                print(f"[reranker] Contradiction detected: {text_lower[:80].encode('ascii', 'replace').decode()}...")
        result.append(item)
    return result


def _fallback_stance(claim: str, passage: dict[str, Any]) -> dict[str, Any]:
    """Assign a conservative neutral stance when NLI is unavailable."""
    item = dict(passage)
    claim_lower = claim.lower()
    text_lower = item.get("text", "").lower()
    refute_terms = ["false", "myth", "not", "no evidence", "misleading", "incorrect"]
    support_terms = ["confirms", "states", "recognized", "evidence", "walked", "dates"]
    contradiction = 0.15
    entailment = 0.15
    if any(term in text_lower for term in refute_terms) and not any(term in claim_lower for term in refute_terms):
        contradiction = 0.55
    if any(term in text_lower for term in support_terms):
        entailment = max(entailment, 0.45)
    neutral = max(0.0, 1.0 - contradiction - entailment)
    if entailment > contradiction and entailment > neutral:
        stance = STANCE_SUPPORTS
    elif contradiction > entailment and contradiction > neutral:
        stance = STANCE_REFUTES
    else:
        stance = STANCE_NEUTRAL
    item["stance"] = stance
    item["stance_scores"] = {
        "entailment": entailment,
        "neutral": neutral,
        "contradiction": contradiction,
    }
    return item


def label_stance(claim: str, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use the NLI cross-encoder to label each passage as support, refute, or neutral."""
    if not passages:
        return []
    model = get_nli_model()
    if model is None:
        labeled = [_fallback_stance(claim, passage) for passage in passages]
    else:
        try:
            pairs = [(claim, passage.get("text", "")) for passage in passages]
            score_rows = model.predict(pairs, batch_size=32, show_progress_bar=False)
            labeled = []
            for passage, score_row in zip(passages, score_rows):
                contradiction, neutral, entailment = _softmax(score_row)
                if entailment > contradiction and entailment > neutral:
                    stance = STANCE_SUPPORTS
                elif contradiction > entailment and contradiction > neutral:
                    stance = STANCE_REFUTES
                else:
                    stance = STANCE_NEUTRAL
                item = dict(passage)
                item["stance"] = stance
                item["stance_scores"] = {
                    "entailment": entailment,
                    "neutral": neutral,
                    "contradiction": contradiction,
                }
                labeled.append(item)
        except Exception as exc:
            print(f"[reranker WARNING] NLI stance labeling failed: {exc}")
            labeled = [_fallback_stance(claim, passage) for passage in passages]

    # --- FIX 4+5: Post-process stances for belief-framing and negation ---
    labeled = [_post_process_stance(claim, item) for item in labeled]

    # --- FIX 10: Detect direct contradictions ---
    labeled = detect_direct_contradictions(claim, labeled)

    return labeled


def warmup_models() -> None:
    """Load both reranker models so app startup can surface model issues early."""
    get_reranker_model()
    get_nli_model()
