"""Parses the fever_eval log file to extract accuracy and metrics without waiting for the full run to complete."""

import sys
import re
from collections import Counter

def parse_log(log_path: str, max_claims: int = 50):
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find each claim block
    # Looks for:
    # [X/100] Claim: ...
    #   Expected: LABEL
    # ...
    #   Predicted: LABEL ✓/✗
    
    results = []
    
    # We can split by "[XX/100] Claim:"
    blocks = re.split(r'\[\d+/100\] Claim:', content)[1:]  # Skip everything before first claim
    
    for block in blocks[:max_claims]:
        expected_match = re.search(r'Expected:\s+(.+)', block)
        predicted_match = re.search(r'Predicted:\s+(.+?)\s+[✓✗]', block)
        
        if expected_match and predicted_match:
            expected = expected_match.group(1).strip()
            predicted = predicted_match.group(1).strip()
            correct = (expected == predicted)
            results.append({
                "expected": expected,
                "predicted": predicted,
                "correct": correct
            })
            
    total = len(results)
    if total == 0:
        print("No completed claims found in the log yet.")
        return
        
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / total
    
    print("==================================================")
    print(f"INTERMEDIATE FEVER EVALUATION RESULTS ({total} claims)")
    print("==================================================")
    print(f"Total claims evaluated : {total}")
    print(f"Correct predictions    : {correct_count}")
    print(f"Label Accuracy         : {accuracy:.2%}")
    
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
    if len(sys.argv) > 1:
        parse_log(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50)
    else:
        print("Please provide log file path.")
