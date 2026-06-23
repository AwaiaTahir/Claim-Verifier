"""Run the 4 required test claims and report pass/fail."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import verify_claim

REQUIRED_CLAIMS = [
    ("The Earth is flat", "REFUTED"),
    ("Neil Armstrong walked on the Moon", "SUPPORTED"),
    ("Vaccines cause autism", "REFUTED"),
    ("asdfjkl qwerty nonsense xyzzy", "NOT ENOUGH EVIDENCE"),
]

def main():
    results = []
    for claim, expected in REQUIRED_CLAIMS:
        output = verify_claim(claim)
        verdict = output["verdict"]
        passed = verdict == expected
        results.append((claim, expected, verdict, passed))

    print("\n" + "=" * 72)
    print("REQUIRED TEST RESULTS")
    print("=" * 72)
    all_passed = True
    for claim, expected, verdict, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {status} | Expected: {expected:25s} | Got: {verdict:25s}")
        print(f"       | Claim: {claim}")

    total = sum(1 for _, _, _, p in results if p)
    print(f"\n{total}/4 passed")

    if all_passed:
        print("\nALL 4 REQUIRED TESTS PASSED!")
    else:
        print("\nSOME TESTS FAILED - needs debugging")
        sys.exit(1)

if __name__ == "__main__":
    main()
