"""Build and save BM25 and FAISS indexes for the local evidence corpus."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (  # noqa: E402
    BM25_INDEX_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CORPUS_PATH,
    DEVICE,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FEVER_CONFIG,
    FEVER_DATASET,
    FEVER_SPLIT,
    INDEX_DIR,
    MAX_FEVER_DOCS,
    METADATA_PATH,
    SOURCE_LOCAL,
    TORCH_DTYPE,
)

# --- FIX 1: Minimum index size assertion ---
MIN_INDEX_CHUNKS = 10000


def ensure_index_dir() -> None:
    """Create the index directory if it does not already exist."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase BM25 tokens."""
    try:
        return [token.lower() for token in word_tokenize(text or "") if token.strip()]
    except Exception:
        return [token.lower() for token in (text or "").split() if token.strip()]


def split_into_sentences(text: str) -> list[str]:
    """Split a document into sentences with a simple fallback tokenizer."""
    try:
        sentences = sent_tokenize(text or "")
    except Exception:
        sentences = [part.strip() for part in (text or "").replace("\n", " ").split(".")]
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_sentences(sentences: list[str]) -> list[str]:
    """Create overlapping chunks from a list of sentences."""
    if not sentences:
        return []
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    chunks = []
    for start in range(0, len(sentences), step):
        chunk = " ".join(sentences[start : start + CHUNK_SIZE]).strip()
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_SIZE >= len(sentences):
            break
    return chunks


def extract_row_text(row: dict[str, Any]) -> str:
    """Extract the best available text field from a FEVER dataset row."""
    text_fields = ["evidence", "evidence_text", "text", "claim"]
    for field in text_fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            flattened = " ".join(str(item) for item in value if item)
            if flattened.strip():
                return flattened.strip()
    return str(row).strip()


def extract_row_title(row: dict[str, Any], index: int) -> str:
    """Extract a readable title for a FEVER row."""
    title_fields = ["title", "wikipedia_title", "wikipedia_url", "id"]
    for field in title_fields:
        value = row.get(field)
        if value:
            return str(value)
    return f"fever_doc_{index}"


def load_fever_rows() -> list[dict[str, Any]]:
    """Load FEVER rows from HuggingFace. Fails loudly if unavailable (FIX 1+2)."""
    from datasets import load_dataset

    print("[build_index] Loading FEVER dataset from HuggingFace...")
    dataset = load_dataset(
        FEVER_DATASET,
        FEVER_CONFIG,
        split=f"{FEVER_SPLIT}[:{MAX_FEVER_DOCS}]",
        trust_remote_code=True,
    )
    rows = [dict(row) for row in dataset]
    print(f"[build_index] Loaded {len(rows)} rows from FEVER dataset")
    if len(rows) < 1000:
        raise RuntimeError(
            f"FEVER dataset returned only {len(rows)} rows — expected at least 1000. "
            "Check your internet connection and the 'datasets' library installation."
        )
    return rows


def build_corpus(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert source rows into searchable chunk records."""
    corpus = []
    for doc_index, row in enumerate(tqdm(rows, desc="Chunking documents")):
        text = extract_row_text(row)
        title = extract_row_title(row, doc_index)
        sentences = split_into_sentences(text)
        chunks = chunk_sentences(sentences) or [text]
        for chunk_id, chunk in enumerate(chunks):
            corpus.append(
                {
                    "text": chunk,
                    "doc_id": str(row.get("id", doc_index)),
                    "title": title,
                    "chunk_id": chunk_id,
                    "source": SOURCE_LOCAL,
                    "url": "",
                    "date": "",
                }
            )
    return corpus


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize embedding vectors for cosine similarity via inner product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def load_embedding_model():
    """Load the sentence-transformer embedding model on the configured device."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    if DEVICE == "cuda":
        try:
            model.half()
        except Exception as exc:
            print(f"[build_index WARNING] Could not switch embedding model to {TORCH_DTYPE}: {exc}")
    return model


def build_bm25_index(corpus: list[dict[str, Any]]):
    """Build a BM25 index from the corpus chunks."""
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(item["text"]) for item in tqdm(corpus, desc="Tokenizing BM25")]
    return BM25Okapi(tokenized_corpus)


def build_faiss_index(corpus: list[dict[str, Any]]):
    """Build a normalized FAISS IndexFlatIP dense index from corpus chunks."""
    import faiss

    texts = [item["text"] for item in corpus]
    try:
        model = load_embedding_model()
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            device=DEVICE,
            convert_to_numpy=True,
        )
    except Exception as exc:
        print(f"[build_index WARNING] Embedding model failed: {exc}")
        print("[build_index] Using deterministic fallback embeddings")
        embeddings = build_fallback_embeddings(texts)
    embeddings = normalize_vectors(embeddings.astype("float32"))
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def build_fallback_embeddings(texts: list[str], dimension: int = 384) -> np.ndarray:
    """Create deterministic hashed embeddings when transformer encoding is unavailable."""
    vectors = np.zeros((len(texts), dimension), dtype="float32")
    for row, text in enumerate(texts):
        tokens = tokenize(text)
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            column = int(digest[:8], 16) % dimension
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vectors[row, column] += sign
    return vectors


def save_indexes(corpus: list[dict[str, Any]], bm25_index: Any, faiss_index: Any) -> None:
    """Persist corpus, BM25, FAISS, and metadata files to disk."""
    import faiss

    ensure_index_dir()
    with BM25_INDEX_PATH.open("wb") as file:
        pickle.dump(bm25_index, file)
    with CORPUS_PATH.open("wb") as file:
        pickle.dump(corpus, file)
    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))

    unique_docs = {item["doc_id"] for item in corpus}
    metadata = {
        "num_documents": len(unique_docs),
        "num_chunks": len(corpus),
        "embedding_model": EMBEDDING_MODEL,
        "build_date": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_indexes() -> None:
    """Build all local retrieval indexes and save them under data/fever_index."""
    ensure_index_dir()
    rows = load_fever_rows()
    corpus = build_corpus(rows)

    # --- FIX 1: Assert minimum index size ---
    assert len(corpus) >= MIN_INDEX_CHUNKS, (
        f"[build_index FATAL] Index too small: {len(corpus)} chunks, "
        f"need at least {MIN_INDEX_CHUNKS}. "
        "The FEVER dataset may not have loaded correctly."
    )
    print(f"[build_index] Corpus size check passed: {len(corpus)} chunks >= {MIN_INDEX_CHUNKS}")

    bm25_index = build_bm25_index(corpus)
    faiss_index = build_faiss_index(corpus)
    save_indexes(corpus, bm25_index, faiss_index)
    print(f"[build_index] Saved {len(corpus)} chunks to {INDEX_DIR}")


def main() -> None:
    """CLI entry point for index construction."""
    try:
        build_indexes()
    except Exception as exc:
        print(f"[build_index ERROR] Index build failed: {exc}")
        raise


if __name__ == "__main__":
    main()
