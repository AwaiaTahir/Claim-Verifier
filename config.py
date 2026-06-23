"""Central configuration for the local claim verification system."""

from pathlib import Path
import os
from dotenv import load_dotenv

# load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

import torch


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "fever_index"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CORPUS_PATH = INDEX_DIR / "corpus.pkl"
METADATA_PATH = INDEX_DIR / "metadata.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
OLLAMA_MODEL = "gemma4:e4b"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT_SECONDS = 30
OLLAMA_REQUEST_TIMEOUT_SECONDS = 30
OLLAMA_OK_PROMPT = "reply with the word ok"

# LLM Backend selection
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq") # Default to groq now
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

FEVER_DATASET = "fever"
FEVER_CONFIG = "v1.0"
FEVER_SPLIT = "train"
MAX_FEVER_DOCS = 50000
CHUNK_SIZE = 3
CHUNK_OVERLAP = 1

TOP_K_BM25 = 50
TOP_K_DENSE = 50
TOP_K_RERANK = 10
TOP_K_FINAL = 5
WEB_TOP_K_PER_QUERY = 5
WEB_SEARCH_BACKEND = "duckduckgo"
WEB_SEARCH_REGION = "wt-wt"
WEB_SEARCH_DELAY_SECONDS = 1.0
WEB_SEARCH_TIMEOUT_SECONDS = 15

RETRIEVAL_CONFIDENCE_THRESHOLD = 0.15
DECISION_RELATIVE_SCORE_THRESHOLD = 0.5
CONFLICT_RATIO_THRESHOLD = 0.35

SUPPORTED = "SUPPORTED"
REFUTED = "REFUTED"
NOT_ENOUGH_EVIDENCE = "NOT ENOUGH EVIDENCE"
CONFLICTING_EVIDENCE = "CONFLICTING EVIDENCE"
NOT_VERIFIABLE = "NOT A VERIFIABLE CLAIM"
VERDICT_LABELS = [
    SUPPORTED,
    REFUTED,
    NOT_ENOUGH_EVIDENCE,
    CONFLICTING_EVIDENCE,
]

STANCE_SUPPORTS = "SUPPORTS"
STANCE_REFUTES = "REFUTES"
STANCE_NEUTRAL = "NEUTRAL"
STANCE_LABELS = [STANCE_SUPPORTS, STANCE_REFUTES, STANCE_NEUTRAL]

CLAIM_FACTUAL = "factual"
CLAIM_OPINION = "opinion"
CLAIM_PREDICTION = "prediction"
CLAIM_AMBIGUOUS = "ambiguous"
CLAIM_COMPOUND = "compound"
CLAIM_TYPES = [
    CLAIM_FACTUAL,
    CLAIM_OPINION,
    CLAIM_PREDICTION,
    CLAIM_AMBIGUOUS,
    CLAIM_COMPOUND,
]

SOURCE_LOCAL = "local_fever"
SOURCE_WEB = "web"

DATE_PENALTY_YEARS = 2
DATE_PENALTY_FACTOR = 0.75

MAX_HOPS = 2
RRF_K = 60
HOP_ONE_BOOST = 1.2

ENCODE_BATCH_SIZE = 64
RERANK_BATCH_SIZE = 32
NLI_BATCH_SIZE = 32

NUMERICAL_PATTERN = (
    r"\b\d+[\.,]?\d*\s*(%|percent|million|billion|thousand|km|kg|"
    r"years?|days?|months?)\b|\b\d{4}\b"
)
ENTITY_PATTERN = (
    r"\b([A-Z][a-z]+ (?:[A-Z][a-z]+ )*[A-Z][a-z]+|[A-Z][a-z]+)\b"
)
ENTITY_STOPLIST = {
    "Earth",
    "November",
    "The",
    "Wall",
    "China",
    "Space",
}

STOPWORD_LANGUAGE = "english"
NLTK_PACKAGES = ["punkt", "punkt_tab", "stopwords"]

DEFAULT_CLASSIFICATION = {
    "type": CLAIM_FACTUAL,
    "is_checkable": True,
    "sub_claims": [],
    "message": "",
}

CLASSIFIER_PROMPT_TEMPLATE = """Classify this claim for automated fact-checking.

Claim: "{claim}"

Respond ONLY with a JSON object. No preamble. No markdown. Example:
{{"type": "factual", "is_checkable": true, "sub_claims": [], "message": ""}}

Types:
- factual: an objective statement that can be verified with evidence
- opinion: a subjective judgment or preference (cannot be fact-checked)
- prediction: a claim about the future (cannot be verified yet)
- compound: multiple distinct factual claims in one sentence (split them)
- ambiguous: unclear or too vague to verify precisely

If compound, populate sub_claims with each atomic claim as a separate string.
If not checkable, populate message explaining why to the user.
"""

QUERY_REWRITE_PROMPT_TEMPLATE = """You are an expert fact-checker preparing search queries to verify a claim.

Claim: "{claim}"

Generate exactly 3 search queries that would help find evidence to verify or refute this claim.

Rules:
- Each query targets a DIFFERENT angle (e.g. scientific evidence, historical record, statistics)
- Queries are NEUTRAL - do not assume the claim is true or false
- If the claim contains negation (not, never, no), at least one query must test that negative case
- Queries are 4-8 words, search-engine style, no question marks
- Respond ONLY with a JSON array of 3 strings. No preamble. No markdown.

Example output: ["query one here", "query two here", "query three here"]
"""

NUMERICAL_VERDICT_PROMPT_TEMPLATE = """You are checking whether a numerical claim matches the evidence.

Claim: "{claim}"

Evidence:
{evidence}

Return ONLY one label from this set:
SUPPORTED
REFUTED
NOT ENOUGH EVIDENCE
"""

EXPLANATION_PROMPT_TEMPLATE = """You are a fact-checker. Based ONLY on the evidence passages below, explain why the claim is {verdict}.

Claim: "{claim}"

Evidence passages:
{evidence}

Instructions:
- Write 2-3 sentences maximum
- Cite the evidence (e.g. "According to [source]...")
- Do NOT add any information not present in the passages
- Do NOT repeat the verdict label - just explain the reasoning
- If verdict is NOT ENOUGH EVIDENCE, explain what kind of evidence is missing
"""

EXAMPLE_CLAIMS = [
    "The Great Wall of China is visible from space",
    "COVID-19 vaccines contain microchips",
    "Albert Einstein failed mathematics in school",
    "The Amazon rainforest produces 20% of Earth's oxygen",
    "Neil Armstrong walked on the Moon in 1969",
]

VERDICT_COLORS = {
    SUPPORTED: "#16803c",
    REFUTED: "#b42318",
    CONFLICTING_EVIDENCE: "#b54708",
    NOT_ENOUGH_EVIDENCE: "#667085",
    NOT_VERIFIABLE: "#667085",
}

STANCE_COLORS = {
    STANCE_SUPPORTS: "#16803c",
    STANCE_REFUTES: "#b42318",
    STANCE_NEUTRAL: "#667085",
}

# --- FIX 3: DuckDuckGo retry configuration ---
DDG_MAX_RETRIES = 3
DDG_BACKOFF_BASE = 1.0

# --- FIX 4+5: Belief-framing and negation detection ---
BELIEF_FRAMING_PHRASES = [
    "believed that",
    "ancient belief",
    "some people believe",
    "commonly believed",
    "once thought",
    "legend has it",
    "it was believed",
    "popular misconception",
    "widely believed",
    "traditional belief",
    "folk belief",
]
NEGATION_WORDS = [
    "not",
    "no",
    "never",
    "false",
    "disproven",
    "debunked",
    "incorrect",
    "untrue",
    "myth",
    "wrong",
    "refuted",
    "denied",
    "contrary",
    "unlikely",
    "impossible",
]

# --- FIX 9: Hop-2 filter terms ---
HOP2_FILTER_TERMS = ["history of", "ancient", "belief", "mythology"]

# --- FIX 10: Contradiction detection thresholds ---
CONTRADICTION_RERANKER_THRESHOLD = 0.85
CONTRADICTION_MIN_PASSAGES = 3

# --- BONUS: Source credibility weights ---
CREDIBILITY_WEIGHTS = {
    "nasa.gov": 1.5,
    "cdc.gov": 1.5,
    "nih.gov": 1.5,
    "snopes.com": 1.5,
    "who.int": 1.5,
    "britannica.com": 1.2,
    "nature.com": 1.3,
    "science.org": 1.3,
    "wikipedia.org": 1.1,
    "reddit.com": 0.5,
    "twitter.com": 0.5,
    "x.com": 0.5,
    "facebook.com": 0.5,
    "tiktok.com": 0.4,
    "quora.com": 0.6,
}
DEFAULT_CREDIBILITY = 1.0
