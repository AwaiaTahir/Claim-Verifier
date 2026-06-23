"""Fixed evaluation claims and expected labels for the project rubric."""

TEST_CLAIMS = [
    # --- Required test claims (must all pass) ---
    {"claim": "The Earth is flat", "expected": "REFUTED"},
    {"claim": "Neil Armstrong walked on the Moon", "expected": "SUPPORTED"},
    {"claim": "Vaccines cause autism", "expected": "REFUTED"},
    {"claim": "asdfjkl qwerty nonsense xyzzy", "expected": "NOT ENOUGH EVIDENCE"},
    # --- Original test claims ---
    {"claim": "The Great Wall of China is visible from space with naked eye", "expected": "REFUTED"},
    {"claim": "COVID-19 vaccines contain microchips", "expected": "REFUTED"},
    {"claim": "Neil Armstrong walked on the Moon in 1969", "expected": "SUPPORTED"},
    {"claim": "Albert Einstein failed mathematics in school", "expected": "REFUTED"},
    {"claim": "The Amazon rainforest produces 20% of Earth's oxygen", "expected": "REFUTED"},
    {"claim": "The Earth is approximately 4.5 billion years old", "expected": "SUPPORTED"},
    {"claim": "Humans only use 10% of their brain", "expected": "REFUTED"},
    {"claim": "Mount Everest is the tallest mountain on Earth", "expected": "SUPPORTED"},
    {"claim": "The Berlin Wall fell in 1989", "expected": "SUPPORTED"},
    {"claim": "Sharks are mammals", "expected": "REFUTED"},
]
