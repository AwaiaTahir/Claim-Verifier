"""
FEVER Dev Set Evaluation (Groq Backend)
Samples 10 claims from the FEVER labelled_dev split and evaluates the
claim verification system using the Groq API (llama-3.3-70b-versatile).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "NOT ENOUGH EVIDENCE",
}

SAMPLE_SIZE = 10
RANDOM_SEED = 42


def load_fever_dev_claims(sample_size: int = SAMPLE_SIZE, seed: int = RANDOM_SEED) -> list[dict]:
    """Load a random sample of claims from the FEVER labelled_dev split."""
    print(f"[Eval] Loading FEVER labelled_dev split (sample={sample_size})...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("fever", "v1.0", split="labelled_dev", trust_remote_code=True)
        print(f"[Eval] Total dev claims: {len(dataset)}")

        # Filter to only claims with clear labels (skip ambiguous)
        valid = [
            item for item in dataset
            if item.get("label") in LABEL_MAP
            and item.get("claim", "").strip()
        ]
        
        # Stratified sample — equal distribution across labels
        by_label: dict[str, list] = {}
        for item in valid:
            lbl = item["label"]
            by_label.setdefault(lbl, []).append(item)

        rng = random.Random(seed)
        per_label = sample_size // len(by_label)
        sampled = []
        for lbl, items in by_label.items():
            rng.shuffle(items)
            sampled.extend(items[:per_label])

        # Fill remaining slots if uneven
        remaining = sample_size - len(sampled)
        if remaining > 0:
            extras = [i for i in valid if i not in sampled]
            rng.shuffle(extras)
            sampled.extend(extras[:remaining])

        rng.shuffle(sampled)
        return sampled

    except Exception as exc:
        print(f"[Eval ERROR] Failed to load FEVER dev set: {exc}")
        sys.exit(1)


def run_fever_eval():
    """Run the FEVER dev set evaluation and report metrics."""
    import config
    # Force Groq backend for this eval
    config.LLM_BACKEND = "groq"
    
    from app import verify_claim
    from modules.llm_client import set_llm_available

    claims = load_fever_dev_claims()

    results = []
    for i, item in enumerate(claims, start=1):
        # Reset LLM availability before each claim so transient errors don't cascade
        set_llm_available(True)
        
        claim_text = item["claim"].strip()
        fever_label = item["label"]
        expected = LABEL_MAP[fever_label]

        print(f"\n[{i}/{len(claims)}] Claim: {claim_text[:70]}")
        print(f"  Expected: {expected}")

        try:
            output = verify_claim(claim_text)
            predicted = output["verdict"]
            correct = predicted == expected
            results.append({
                "claim": claim_text,
                "fever_label": fever_label,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "confidence": output.get("confidence", 0.0),
            })
            status = "CORRECT" if correct else "WRONG"
            print(f"  Predicted: {predicted} {status}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({
                "claim": claim_text,
                "fever_label": fever_label,
                "expected": expected,
                "predicted": "ERROR",
                "correct": False,
                "confidence": 0.0,
            })

    # --- Metrics ---
    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    label_accuracy = correct_count / total if total else 0.0

    print("\n" + "=" * 72)
    print("GROQ FEVER EVALUATION RESULTS")
    print("=" * 72)
    print(f"Total claims evaluated : {total}")
    print(f"Correct predictions    : {correct_count}")
    print(f"Label Accuracy         : {label_accuracy:.2%}")

    print("\nPer-label breakdown:")
    for expected_label in ["SUPPORTED", "REFUTED", "NOT ENOUGH EVIDENCE"]:
        subset = [r for r in results if r["expected"] == expected_label]
        if not subset:
            continue
        correct_sub = sum(1 for r in subset if r["correct"])
        print(f"  {expected_label:25s}: {correct_sub}/{len(subset)} ({correct_sub/len(subset):.0%})")

    print("\nPrediction distribution:")
    pred_counts = Counter(r["predicted"] for r in results)
    for label, count in pred_counts.most_common():
        print(f"  {label:25s}: {count}")

if __name__ == "__main__":
    run_fever_eval()
