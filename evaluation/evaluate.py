"""Evaluation script for label accuracy, Precision@3, and failure analysis."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import verify_claim  # noqa: E402
from config import REFUTED, STANCE_REFUTES, STANCE_SUPPORTS, SUPPORTED  # noqa: E402
from evaluation.test_queries import TEST_CLAIMS  # noqa: E402


def expected_to_stance(expected: str) -> str:
    """Map a verdict label to the evidence stance expected for Precision@K."""
    if expected == SUPPORTED:
        return STANCE_SUPPORTS
    if expected == REFUTED:
        return STANCE_REFUTES
    return ""


def majority_stance(evidence: list[dict[str, Any]], k: int) -> str:
    """Return the majority stance among the top K evidence passages."""
    stances = [item.get("stance", "") for item in evidence[:k] if item.get("stance")]
    if not stances:
        return ""
    counts = Counter(stances)
    return counts.most_common(1)[0][0]


def compute_precision_at_k(results: list[dict[str, Any]], k: int = 3) -> float:
    """Compute whether the top-K evidence majority stance matches the expected label."""
    if not results:
        return 0.0
    correct = 0
    for result in results:
        expected_stance = expected_to_stance(result["expected"])
        if expected_stance and majority_stance(result.get("evidence", []), k) == expected_stance:
            correct += 1
    return correct / len(results)


def print_failure_analysis(results: list[dict[str, Any]]) -> None:
    """Print queries and top evidence for every failed evaluation claim."""
    failures = [result for result in results if not result["correct"]]
    print("\nFailure case analysis:")
    if not failures:
        print("  No failures found in this evaluation run.")
        return

    for failure in failures:
        print(f"\nClaim: {failure['claim']}")
        print(f"Expected: {failure['expected']}")
        print(f"Predicted: {failure['predicted']}")
        print("Queries used:")
        for query in failure.get("queries_used", []):
            print(f"  - {query}")
        print("Top evidence:")
        for item in failure.get("evidence", [])[:3]:
            print(
                f"  #{item.get('rank')} [{item.get('stance')}] "
                f"{item.get('source')} - {item.get('text', '')[:180]}"
            )


def run_evaluation() -> list[dict[str, Any]]:
    """Run the fixed test set and print rubric-oriented metrics."""
    results = []
    for item in TEST_CLAIMS:
        output = verify_claim(item["claim"])
        predicted = output["verdict"]
        expected = item["expected"]
        correct = predicted == expected
        results.append(
            {
                "claim": item["claim"],
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "confidence": output["confidence"],
                "queries_used": output.get("queries_used", []),
                "evidence": output.get("evidence", []),
            }
        )

    label_accuracy = sum(result["correct"] for result in results) / len(results)
    precision_at_3 = compute_precision_at_k(results, k=3)

    print(f"\nLabel Accuracy: {label_accuracy:.2%}")
    print(f"Precision@3: {precision_at_3:.2%}")
    print("\nPer-claim results:")
    for result in results:
        status = "OK" if result["correct"] else "FAIL"
        print(f"  {status:4} [{result['expected']:25}] {result['claim'][:60]}")

    print_failure_analysis(results)
    return results


if __name__ == "__main__":
    run_evaluation()
