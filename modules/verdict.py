"""Decision logic for converting evidence stances into final verdicts."""

from __future__ import annotations

import re
from typing import Any

from config import (
    CONFLICT_RATIO_THRESHOLD,
    CONFLICTING_EVIDENCE,
    CONTRADICTION_MIN_PASSAGES,
    CONTRADICTION_RERANKER_THRESHOLD,
    DECISION_RELATIVE_SCORE_THRESHOLD,
    NOT_ENOUGH_EVIDENCE,
    NUMERICAL_PATTERN,
    NUMERICAL_VERDICT_PROMPT_TEMPLATE,
    REFUTED,
    RETRIEVAL_CONFIDENCE_THRESHOLD,
    STANCE_NEUTRAL,
    STANCE_REFUTES,
    STANCE_SUPPORTS,
    SUPPORTED,
    VERDICT_LABELS,
)
from modules.llm_client import generate


def _is_numerical_claim(claim: str) -> bool:
    """Return True when a claim contains numbers, dates, percentages, or quantities."""
    return bool(re.search(NUMERICAL_PATTERN, claim or "", flags=re.IGNORECASE))


def _call_ollama_for_numerical_verdict(claim: str, passages: list[dict[str, Any]]) -> str | None:
    """Ask Ollama to compare numerical values in a claim against top evidence passages."""
    evidence = "\n".join(
        f"{index}. {passage.get('text', '')}"
        for index, passage in enumerate(passages[:3], start=1)
    )
    prompt = NUMERICAL_VERDICT_PROMPT_TEMPLATE.format(claim=claim, evidence=evidence)
    response_text = generate(prompt, task="numerical verdict")
    if not response_text:
        return None

    text = response_text.upper()
    for label in VERDICT_LABELS:
        if label in text and label != CONFLICTING_EVIDENCE:
            return label
    return None


# --- FIX 6: Weighted stance scoring ---
def _weighted_stances(passages: list[dict[str, Any]]) -> tuple[float, float, float, int, int, int]:
    """Compute weighted support/refute/neutral scores and raw counts.

    Weight = reranker_score * stance_confidence (the max NLI probability).
    Returns (weighted_support, weighted_refute, weighted_neutral,
             supports_count, refutes_count, neutral_count).
    """
    weighted_support = 0.0
    weighted_refute = 0.0
    weighted_neutral = 0.0
    supports_count = 0
    refutes_count = 0
    neutral_count = 0

    for passage in passages:
        stance = passage.get("stance", STANCE_NEUTRAL)
        reranker_score = float(passage.get("reranker_score", 0.0))
        stance_scores = passage.get("stance_scores", {})

        if stance == STANCE_SUPPORTS:
            confidence = float(stance_scores.get("entailment", 0.5))
            weighted_support += reranker_score * confidence
            supports_count += 1
        elif stance == STANCE_REFUTES:
            confidence = float(stance_scores.get("contradiction", 0.5))
            weighted_refute += reranker_score * confidence
            refutes_count += 1
        else:
            confidence = float(stance_scores.get("neutral", 0.5))
            weighted_neutral += reranker_score * confidence
            neutral_count += 1

    return weighted_support, weighted_refute, weighted_neutral, supports_count, refutes_count, neutral_count


def _decision_passages(passages: list[dict[str, Any]], max_score: float) -> list[dict[str, Any]]:
    """Keep only passages relevant enough to vote on the final verdict."""
    minimum_score = max(
        RETRIEVAL_CONFIDENCE_THRESHOLD,
        max_score * DECISION_RELATIVE_SCORE_THRESHOLD,
    )
    filtered = [
        passage
        for passage in passages
        if float(passage.get("reranker_score_prenorm", passage.get("reranker_score", 0.0))) >= minimum_score
    ]
    return filtered or passages[:1]


def passes_retrieval_confidence(passages: list[dict[str, Any]]) -> bool:
    """Check whether retrieved passages are relevant enough before NLI is run.

    Uses pre-normalization scores when available so that min-max normalized
    scores (which always max at 1.0) don't bypass the threshold.
    """
    max_score = max(
        (float(passage.get("reranker_score_prenorm", passage.get("reranker_score", 0.0)))
         for passage in passages),
        default=0.0,
    )
    return max_score >= RETRIEVAL_CONFIDENCE_THRESHOLD


def _base_result(
    verdict: str,
    confidence: float,
    supports_count: int,
    refutes_count: int,
    neutral_count: int,
    is_numerical: bool,
    reason: str,
) -> dict[str, Any]:
    """Build a normalized verdict result dictionary."""
    return {
        "verdict": verdict,
        "confidence": round(float(max(0.0, min(1.0, confidence))), 4),
        "supports_count": supports_count,
        "refutes_count": refutes_count,
        "neutral_count": neutral_count,
        "is_numerical": is_numerical,
        "reason": reason,
    }


def _print_verdict_result(result: dict[str, Any]) -> None:
    """Print a concise verdict summary for terminal diagnostics."""
    print(
        "[Verdict] "
        f"verdict={result['verdict']} confidence={result['confidence']:.2f} "
        f"supports={result.get('supports_count', 0)} "
        f"refutes={result.get('refutes_count', 0)} "
        f"neutral={result.get('neutral_count', 0)} "
        f"numerical={result.get('is_numerical', False)} "
        f"reason={result.get('reason', '')}"
    )
    return None


# --- FIX 10: Early exit for strong refutation ---
def _check_strong_refutation(passages: list[dict[str, Any]]) -> bool:
    """Return True if 2+ passages have reranker_score > threshold and stance REFUTES."""
    strong_refutes = [
        p for p in passages
        if float(p.get("reranker_score_prenorm", p.get("reranker_score", 0.0))) > CONTRADICTION_RERANKER_THRESHOLD
        and p.get("stance") == STANCE_REFUTES
    ]
    return len(strong_refutes) >= CONTRADICTION_MIN_PASSAGES


def predict_verdict(claim: str, passages: list[dict[str, Any]]) -> dict[str, Any]:
    """Predict a final verdict from reranked passages and stance labels."""
    is_numerical = _is_numerical_claim(claim)
    if not passages:
        result = _base_result(
            NOT_ENOUGH_EVIDENCE,
            0.0,
            0,
            0,
            0,
            is_numerical,
            "No evidence passages were retrieved for this claim.",
        )
        _print_verdict_result(result)
        return result

    max_score = max(float(passage.get("reranker_score_prenorm", passage.get("reranker_score", 0.0))) for passage in passages)
    if max_score < RETRIEVAL_CONFIDENCE_THRESHOLD:
        result = _base_result(
            NOT_ENOUGH_EVIDENCE,
            max_score,
            0,
            0,
            len(passages),
            is_numerical,
            "No sufficiently relevant evidence was found for this claim.",
        )
        _print_verdict_result(result)
        return result

    # --- FIX 10: Early exit for strong refutation ---
    if _check_strong_refutation(passages):
        _, _, _, supports_count, refutes_count, neutral_count = _weighted_stances(passages)
        result = _base_result(
            REFUTED,
            max_score,
            supports_count,
            refutes_count,
            neutral_count,
            is_numerical,
            f"Strong refutation detected: {refutes_count} high-confidence refuting passage(s).",
        )
        print("[Verdict] Early exit: strong refutation detected")
        _print_verdict_result(result)
        return result

    voting_passages = _decision_passages(passages, max_score)

    # --- FIX 6: Use weighted scoring instead of raw counts ---
    weighted_support, weighted_refute, weighted_neutral, \
        supports_count, refutes_count, neutral_count = _weighted_stances(voting_passages)

    total_weight = weighted_support + weighted_refute + weighted_neutral
    total = len(voting_passages)

    if supports_count > 0 and refutes_count > 0:
        conflict_ratio = min(supports_count, refutes_count) / total if total else 0.0
        if conflict_ratio >= CONFLICT_RATIO_THRESHOLD:
            result = _base_result(
                CONFLICTING_EVIDENCE,
                max_score,
                supports_count,
                refutes_count,
                neutral_count,
                is_numerical,
                f"Found {supports_count} supporting and {refutes_count} refuting passages.",
            )
            _print_verdict_result(result)
            return result

    if supports_count == 0 and refutes_count == 0:
        result = _base_result(
            NOT_ENOUGH_EVIDENCE,
            max_score,
            supports_count,
            refutes_count,
            neutral_count,
            is_numerical,
            "Retrieved evidence was relevant but did not clearly support or refute the claim.",
        )
        _print_verdict_result(result)
        return result

    # --- FIX 6: Use weighted ratios for majority decision ---
    if weighted_support > weighted_refute:
        majority = SUPPORTED
    elif weighted_refute > weighted_support:
        majority = REFUTED
    else:
        # Tie-break: use counts
        majority = SUPPORTED if supports_count >= refutes_count else REFUTED

    if is_numerical:
        ollama_verdict = _call_ollama_for_numerical_verdict(claim, passages)
        if ollama_verdict in {SUPPORTED, REFUTED, NOT_ENOUGH_EVIDENCE}:
            majority = ollama_verdict

    if majority == NOT_ENOUGH_EVIDENCE:
        agreement = weighted_neutral / total_weight if total_weight else 0.0
        reason = "The numerical comparison did not find enough evidence to decide the claim."
    elif majority == SUPPORTED:
        agreement = weighted_support / total_weight if total_weight else 0.0
        reason = f"The strongest evidence leans supportive across {supports_count} passage(s) (weighted score: {weighted_support:.2f})."
    else:
        agreement = weighted_refute / total_weight if total_weight else 0.0
        reason = f"The strongest evidence leans refuting across {refutes_count} passage(s) (weighted score: {weighted_refute:.2f})."

    confidence = max_score * agreement
    result = _base_result(
        majority,
        confidence,
        supports_count,
        refutes_count,
        neutral_count,
        is_numerical,
        reason,
    )
    _print_verdict_result(result)
    return result
