"""Hybrid local, web, and iterative retrieval for evidence passages."""

from __future__ import annotations

import json
import pickle
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import numpy as np
from nltk.tokenize import word_tokenize

from config import (
    BM25_INDEX_PATH,
    CORPUS_PATH,
    DATE_PENALTY_FACTOR,
    DATE_PENALTY_YEARS,
    DDG_BACKOFF_BASE,
    DDG_MAX_RETRIES,
    DEVICE,
    EMBEDDING_MODEL,
    ENTITY_PATTERN,
    ENTITY_STOPLIST,
    FAISS_INDEX_PATH,
    HOP_ONE_BOOST,
    HOP2_FILTER_TERMS,
    MAX_HOPS,
    METADATA_PATH,
    RRF_K,
    SOURCE_LOCAL,
    SOURCE_WEB,
    TOP_K_BM25,
    TOP_K_DENSE,
    TORCH_DTYPE,
    WEB_SEARCH_BACKEND,
    WEB_SEARCH_DELAY_SECONDS,
    WEB_SEARCH_REGION,
    WEB_SEARCH_TIMEOUT_SECONDS,
    WEB_TOP_K_PER_QUERY,
)

_embedding_model = None
_embedding_model_failed = False


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase tokens for BM25."""
    try:
        return [token.lower() for token in word_tokenize(text or "") if token.strip()]
    except Exception:
        return [token.lower() for token in (text or "").split() if token.strip()]


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize dense vectors for inner-product cosine search."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def reciprocal_rank_score(rank: int) -> float:
    """Compute the Reciprocal Rank Fusion contribution for a 1-based rank."""
    return 1.0 / (RRF_K + rank)


def result_key(result: dict[str, Any]) -> str:
    """Create a stable deduplication key for a retrieval result."""
    if result.get("url"):
        return f"url::{result['url']}"
    return f"local::{result.get('doc_id', '')}::{result.get('chunk_id', '')}::{result.get('title', '')}"


def get_embedding_model():
    """Load the sentence-transformer embedding model lazily on the configured GPU."""
    global _embedding_model, _embedding_model_failed
    if _embedding_model is not None:
        return _embedding_model
    if _embedding_model_failed:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        if DEVICE == "cuda":
            try:
                _embedding_model.half()
            except Exception as exc:
                print(f"[retriever WARNING] Could not switch embedding model to {TORCH_DTYPE}: {exc}")
        return _embedding_model
    except Exception as exc:
        print(f"[retriever WARNING] Embedding model unavailable: {exc}")
        _embedding_model_failed = True
        return None


def domain_from_url(url: str) -> str:
    """Extract a readable domain name from a URL."""
    try:
        domain = urlparse(url).netloc
        return domain.replace("www.", "") or SOURCE_WEB
    except Exception:
        return SOURCE_WEB


def summarize_domains(results: list[dict[str, Any]], limit: int = 5) -> list[str]:
    """Summarize unique domains from web retrieval results."""
    domains = []
    seen = set()
    for result in results:
        domain = domain_from_url(result.get("url", ""))
        if domain and domain not in seen:
            domains.append(domain)
            seen.add(domain)
        if len(domains) >= limit:
            break
    return domains


def parse_result_year(date_text: str) -> int | None:
    """Extract a four-digit year from a web result date string."""
    match = re.search(r"\b(19|20)\d{2}\b", date_text or "")
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def date_penalty(date_text: str) -> float:
    """Return the score multiplier for an older web result date."""
    year = parse_result_year(date_text)
    if year is None:
        return 1.0
    current_year = datetime.now().year
    if current_year - year > DATE_PENALTY_YEARS:
        return DATE_PENALTY_FACTOR
    return 1.0


def merge_results(result_lists: list[list[dict[str, Any]]], boosts: list[float] | None = None) -> list[dict[str, Any]]:
    """Merge ranked result lists with RRF-style score summation and deduplication."""
    merged: dict[str, dict[str, Any]] = {}
    if boosts is None:
        boosts = [1.0 for _ in result_lists]
    for results, boost in zip(result_lists, boosts):
        for rank, item in enumerate(results, start=1):
            key = result_key(item)
            score = float(item.get("rrf_score", reciprocal_rank_score(rank))) * boost
            if key not in merged:
                merged[key] = dict(item)
                merged[key]["rrf_score"] = 0.0
            merged[key]["rrf_score"] += score
    return sorted(merged.values(), key=lambda item: item.get("rrf_score", 0.0), reverse=True)


def extract_keywords(text: str, max_words: int = 5) -> str:
    """Extract simple lowercase keywords for second-hop retrieval."""
    words = []
    for token in re.findall(r"[A-Za-z0-9]+", text or ""):
        lowered = token.lower()
        if len(lowered) > 2 and lowered not in {"the", "and", "that", "with", "from"}:
            words.append(lowered)
    return " ".join(words[:max_words])


def normalize_search_query(query: str) -> str:
    """Clean a search query without changing its meaning."""
    return re.sub(r"\s+", " ", (query or "").strip(" ?\n\t"))


def dedupe_queries(queries: list[str], limit: int) -> list[str]:
    """Deduplicate search queries while preserving order."""
    cleaned = []
    seen = set()
    for query in queries:
        value = normalize_search_query(query)
        key = value.lower()
        if value and key not in seen:
            cleaned.append(value)
            seen.add(key)
        if len(cleaned) >= limit:
            break
    return cleaned


def build_local_queries(claim: str, queries: list[str], limit: int = 6) -> list[str]:
    """Prefer direct claim terms, then add rewritten queries for local retrieval."""
    keyword_query = extract_keywords(claim, max_words=8)
    return dedupe_queries([claim, keyword_query, *queries], limit)


def build_web_queries(claim: str, queries: list[str], limit: int = 5) -> list[str]:
    """Build web queries that start with simple DDG-friendly wording."""
    keyword_query = extract_keywords(claim, max_words=8)
    direct_claim = normalize_search_query(claim)
    direct_variants = [
        query
        for query in [
            f"{direct_claim} fact check",
            direct_claim,
            f"{keyword_query} fact check",
            f"{keyword_query} evidence",
        ]
        if query.strip() not in {"fact check", "evidence"}
    ]
    return dedupe_queries([*direct_variants, *queries], limit)


def extract_entities(texts: list[str], claim: str, max_entities: int = 3) -> list[str]:
    """Extract capitalized entity candidates not already present in the original claim."""
    claim_lower = claim.lower()
    entities = []
    seen = set()
    for text in texts:
        for match in re.findall(ENTITY_PATTERN, text or ""):
            entity = match.strip()
            key = entity.lower()
            if entity in ENTITY_STOPLIST:
                continue
            if len(entity.split()) == 1 and len(entity) < 5:
                continue
            if key and key not in seen and key not in claim_lower:
                entities.append(entity)
                seen.add(key)
            if len(entities) >= max_entities:
                return entities
    return entities


# --- FIX 9: Filter hop-2 results containing misleading terms ---
def _filter_hop2_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove hop-2 results whose URL or title contains misleading terms."""
    filtered = []
    for item in results:
        url_lower = str(item.get("url", "")).lower()
        title_lower = str(item.get("title", "")).lower()
        combined = f"{url_lower} {title_lower}"
        if any(term in combined for term in HOP2_FILTER_TERMS):
            print(f"[IterativeRetriever] Filtered hop2 result: {item.get('title', '')[:60]}")
            continue
        filtered.append(item)
    return filtered


class LocalRetriever:
    """Lazy local retriever backed by BM25 and FAISS indexes."""

    def __init__(self) -> None:
        """Initialize unloaded index holders."""
        self.bm25_index = None
        self.faiss_index = None
        self.corpus: list[dict[str, Any]] = []
        self.loaded = False

    def load_indexes(self) -> None:
        """Load local indexes from disk. If missing, log error and return empty (FIX 2)."""
        if self.loaded:
            return
        try:
            with BM25_INDEX_PATH.open("rb") as file:
                self.bm25_index = pickle.load(file)
            with CORPUS_PATH.open("rb") as file:
                self.corpus = pickle.load(file)
            try:
                import faiss

                self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
            except Exception as exc:
                print(f"[retriever WARNING] FAISS index unavailable: {exc}")
                self.faiss_index = None
            self.loaded = True
            self._print_index_status()
        except Exception as exc:
            # --- FIX 2: No fallback corpus — just log error and return empty ---
            print(
                f"[retriever ERROR] Local index files not found: {exc}\n"
                "  Run 'python data/build_index.py' to build the FEVER index."
            )
            self.corpus = []
            self.bm25_index = None
            self.faiss_index = None
            self.loaded = True

    def _print_index_status(self) -> None:
        """Print loaded index statistics."""
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
        num_chunks = int(metadata.get("num_chunks", len(self.corpus)) or len(self.corpus))
        print(
            "[LocalRetriever] Loaded local corpus "
            f"chunks={len(self.corpus)} metadata_chunks={num_chunks}"
        )

    def _bm25_search(self, query: str) -> list[dict[str, Any]]:
        """Run one BM25 search and return ranked result dictionaries."""
        if self.bm25_index is None or not self.corpus:
            return []
        try:
            scores = self.bm25_index.get_scores(tokenize(query))
            top_indices = np.argsort(scores)[::-1][:TOP_K_BM25]
            results = []
            for rank, index in enumerate(top_indices, start=1):
                item = dict(self.corpus[int(index)])
                item["rrf_score"] = reciprocal_rank_score(rank)
                item["bm25_score"] = float(scores[int(index)])
                results.append(item)
            return results
        except Exception as exc:
            print(f"[retriever WARNING] BM25 search failed: {exc}")
            return []

    def _dense_search(self, query: str) -> list[dict[str, Any]]:
        """Run one dense FAISS search and return ranked result dictionaries."""
        if self.faiss_index is None or not self.corpus:
            return []
        model = get_embedding_model()
        if model is None:
            return []
        try:
            query_vector = model.encode(
                [query],
                show_progress_bar=False,
                device=DEVICE,
                convert_to_numpy=True,
            ).astype("float32")
            query_vector = normalize_vectors(query_vector)
            scores, indices = self.faiss_index.search(query_vector, TOP_K_DENSE)
            results = []
            for rank, index in enumerate(indices[0], start=1):
                if index < 0:
                    continue
                item = dict(self.corpus[int(index)])
                item["rrf_score"] = reciprocal_rank_score(rank)
                item["dense_score"] = float(scores[0][rank - 1])
                results.append(item)
            return results
        except Exception as exc:
            print(f"[retriever WARNING] Dense search failed: {exc}")
            return []

    def retrieve_local(self, queries: list[str], top_k: int) -> list[dict[str, Any]]:
        """Retrieve local FEVER evidence for rewritten queries."""
        self.load_indexes()
        per_query_results = []
        for query in queries:
            bm25_results = self._bm25_search(query)
            dense_results = self._dense_search(query)
            per_query_results.append(merge_results([bm25_results, dense_results]))
        merged = merge_results(per_query_results)
        cleaned = []
        for item in merged[:top_k]:
            cleaned.append(
                {
                    "text": item.get("text", ""),
                    "title": item.get("title", ""),
                    "doc_id": item.get("doc_id", ""),
                    "chunk_id": item.get("chunk_id", ""),
                    "source": SOURCE_LOCAL,
                    "url": item.get("url", ""),
                    "rrf_score": float(item.get("rrf_score", 0.0)),
                    "date": item.get("date", ""),
                }
            )
        print(f"[LocalRetriever] Retrieved {len(cleaned)} local passages from {len(queries)} queries")
        return cleaned


class WebRetriever:
    """DuckDuckGo web retriever with retry logic and defensive fallbacks (FIX 3)."""

    def __init__(self) -> None:
        """Initialize web retrieval state."""
        self.rate_limited = False
        self.cache: dict[str, list[dict[str, Any]]] = {}

    def _load_ddgs_class(self):
        """Load the current DDGS package first, falling back to the legacy package."""
        try:
            from ddgs import DDGS

            return DDGS
        except Exception:
            try:
                from duckduckgo_search import DDGS

                return DDGS
            except Exception as exc:
                print(f"[retriever WARNING] DDGS package unavailable: {exc}")
                return None

    def _search_wikipedia(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Fallback to Wikipedia API if DDG fails."""
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json&srlimit={top_k}"
            req = urllib.request.Request(url, headers={'User-Agent': 'ClaimVerifierBot/1.0 (test@example.com)'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                results = []
                for item in data.get('query', {}).get('search', []):
                    # Clean HTML tags from snippet
                    snippet = re.sub(r'<[^>]+>', '', item.get('snippet', ''))
                    title = item.get('title', '')
                    results.append({
                        "href": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                        "title": title,
                        "body": snippet
                    })
                return results
        except Exception as exc:
            print(f"[retriever WARNING] Wikipedia fallback failed for '{query}': {exc}")
            return []

    def _search_query_with_retry(
        self, ddgs_class: Any, query: str, top_k_per_query: int
    ) -> list[dict[str, Any]]:
        """Run one web search with retry logic and exponential backoff (FIX 3).

        Only retries on real transient errors (timeouts, connection errors).
        Does NOT retry on 'No results found' — that is a valid empty response.
        Falls back to Wikipedia API only on connection failures.
        """
        last_exc = None
        connection_failed = False
        for attempt in range(1, DDG_MAX_RETRIES + 1):
            try:
                with ddgs_class(timeout=WEB_SEARCH_TIMEOUT_SECONDS) as ddgs:
                    try:
                        results = list(
                            ddgs.text(
                                query,
                                region=WEB_SEARCH_REGION,
                                safesearch="moderate",
                                backend="lite",
                                max_results=top_k_per_query,
                            )
                        )
                    except TypeError:
                        results = list(ddgs.text(query, max_results=top_k_per_query))
                return results  # Success (even if empty)
            except Exception as exc:
                exc_str = str(exc).lower()
                # "No results found" is not transient — it's a valid empty response, don't retry
                if "no results" in exc_str:
                    return []
                last_exc = exc
                connection_failed = True
                if attempt < DDG_MAX_RETRIES:
                    backoff = DDG_BACKOFF_BASE * (2 ** (attempt - 1))
                    print(
                        f"[retriever WARNING] DDG attempt {attempt}/{DDG_MAX_RETRIES} "
                        f"failed for '{query}': {exc}. Retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
                else:
                    print(
                        f"[retriever WARNING] DDG all {DDG_MAX_RETRIES} attempts "
                        f"failed for '{query}': {last_exc}"
                    )

        # Only fallback to Wikipedia on real connection failures (not empty results)
        if connection_failed:
            print(f"[retriever] DDG connection failed, falling back to Wikipedia for '{query}'")
            return self._search_wikipedia(query, top_k_per_query)

        return []

    def retrieve_web(self, queries: list[str], top_k_per_query: int = WEB_TOP_K_PER_QUERY) -> list[dict[str, Any]]:
        """Retrieve web snippets for rewritten queries without crashing on search failure."""
        if self.rate_limited:
            return []

        collected: dict[str, dict[str, Any]] = {}
        ddgs_class = self._load_ddgs_class()
        if ddgs_class is None:
            return []

        print(
            "[WebRetriever] "
            f"backend=lite region=wt-wt "
            f"timeout={WEB_SEARCH_TIMEOUT_SECONDS}s queries={queries}"
        )
        for query in queries:
            if query in self.cache:
                results = self.cache[query]
                print(f"[WebRetriever] Cache hit for '{query}' -> {len(results)} results")
            else:
                # --- FIX 3: Sequential with 1s sleep between queries ---
                if collected:
                    time.sleep(WEB_SEARCH_DELAY_SECONDS)
                try:
                    print(f"[WebRetriever] Searching '{query}'")
                    results = self._search_query_with_retry(ddgs_class, query, top_k_per_query)
                    self.cache[query] = results
                    print(f"[WebRetriever] Query returned {len(results)} results")
                except Exception as exc:
                    print(f"[retriever WARNING] DuckDuckGo query failed for '{query}': {exc}")
                    if "ratelimit" in str(exc).lower() or "rate limit" in str(exc).lower() or "202" in str(exc):
                        self.rate_limited = True
                        print("[retriever WARNING] DuckDuckGo is rate-limited; skipping web retrieval for this session")
                        return sorted(collected.values(), key=lambda item: item.get("rrf_score", 0.0), reverse=True)
                    continue

            for rank, result in enumerate(results, start=1):
                url = str(result.get("href", "") or result.get("url", "") or "")
                if not url:
                    continue
                date_text = str(result.get("date", "") or result.get("published", "") or "")
                score = reciprocal_rank_score(rank) * date_penalty(date_text)
                item = {
                    "text": str(result.get("body", "") or ""),
                    "title": str(result.get("title", "") or domain_from_url(url)),
                    "doc_id": url,
                    "chunk_id": 0,
                    "source": SOURCE_WEB,
                    "url": url,
                    "rrf_score": score,
                    "date": date_text,
                }
                if url not in collected or score > collected[url].get("rrf_score", 0.0):
                    collected[url] = item
        web_results = sorted(collected.values(), key=lambda item: item.get("rrf_score", 0.0), reverse=True)
        print(f"[WebRetriever] Returning {len(web_results)} web passages; domains={summarize_domains(web_results)}")
        return web_results


class IterativeRetriever:
    """Multi-hop retriever combining local and web evidence sources."""

    def __init__(self) -> None:
        """Create local and web retrievers used by each pipeline run."""
        self.local = LocalRetriever()
        self.web = WebRetriever()
        self.last_queries_used: list[str] = []

    def retrieve(self, claim: str, queries: list[str]) -> list[dict[str, Any]]:
        """Retrieve evidence for a claim through one or two retrieval hops."""
        local_queries = build_local_queries(claim, queries)
        web_queries = build_web_queries(claim, queries)
        self.last_queries_used = [
            *(f"web: {query}" for query in web_queries),
            *(f"local: {query}" for query in local_queries),
        ]
        print(f"[IterativeRetriever] local queries={local_queries}")
        print(f"[IterativeRetriever] web queries={web_queries}")

        local_results = self.local.retrieve_local(local_queries, top_k=TOP_K_BM25)
        web_results = self.web.retrieve_web(web_queries, top_k_per_query=WEB_TOP_K_PER_QUERY)
        hop_one = merge_results([local_results, web_results])
        print(
            "[IterativeRetriever] "
            f"hop1 local={len(local_results)} web={len(web_results)} merged={len(hop_one)}"
        )

        if MAX_HOPS < 2 or not hop_one:
            return hop_one
        if not web_results and not self.local.corpus:
            print("[IterativeRetriever] Skipping hop2: no web evidence and local index is empty")
            return hop_one

        # --- FIX 9: Fact-focused hop-2 queries instead of entity-based ---
        subject = extract_keywords(claim, max_words=6)
        if not subject.strip():
            return hop_one

        follow_up_queries = [
            f"{subject} scientific consensus",
            f"{subject} fact check",
            f"is it true that {subject}",
        ]
        follow_up_queries = dedupe_queries(follow_up_queries, limit=3)
        self.last_queries_used.extend(f"local hop2: {query}" for query in follow_up_queries)

        hop_two_local = self.local.retrieve_local(follow_up_queries, top_k=TOP_K_DENSE)
        # --- FIX 9: Filter out misleading hop-2 results ---
        hop_two_local = _filter_hop2_results(hop_two_local)

        merged = merge_results([hop_one, hop_two_local], boosts=[HOP_ONE_BOOST, 1.0])
        print(
            "[IterativeRetriever] "
            f"hop2 queries={follow_up_queries} local={len(hop_two_local)} final={len(merged)}"
        )
        return merged
