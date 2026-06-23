"""Shared LLM client state and safe generation helper for both Ollama and Groq."""

from __future__ import annotations

import time
import requests

import config
from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    OLLAMA_MODEL,
    OLLAMA_REQUEST_TIMEOUT_SECONDS,
    OLLAMA_URL,
)

try:
    import groq as groq_lib
    _groq_client = groq_lib.Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except ImportError:
    _groq_client = None

_llm_available = True
_consecutive_failures = 0
_MAX_CONSECUTIVE_FAILURES = 5  # Only disable after 5 consecutive failures


def set_llm_available(is_available: bool) -> None:
    """Set whether pipeline modules should attempt LLM requests."""
    global _llm_available, _consecutive_failures
    _llm_available = is_available
    if is_available:
        _consecutive_failures = 0


def is_llm_available() -> bool:
    """Return whether LLM requests are currently enabled."""
    return _llm_available


def generate(prompt: str, timeout: int = OLLAMA_REQUEST_TIMEOUT_SECONDS, task: str = "generation") -> str | None:
    """Generate text through the selected backend, or return None quickly when unavailable."""
    global _llm_available, _consecutive_failures
    if not _llm_available:
        print(f"[LLM] Skipping {task}: fallback mode is active")
        return None

    backend = config.LLM_BACKEND.lower()

    if backend == "groq":
        if not _groq_client:
            print(f"[LLM WARNING] {task}: Groq selected but API key missing or package not installed.")
            return None

        try:
            print(f"[LLM] {task}: sending prompt to Groq ({GROQ_MODEL})")
            completion = _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=256,
            )
            result = str(completion.choices[0].message.content).strip()
            print(f"[LLM] {task}: received {len(result)} characters from Groq")
            _consecutive_failures = 0  # Reset on success
            return result
        except Exception as exc:
            _consecutive_failures += 1
            print(f"[LLM WARNING] {task}: Groq request failed ({_consecutive_failures}/{_MAX_CONSECUTIVE_FAILURES}): {exc}")
            if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                print("[LLM] Too many consecutive failures — disabling LLM.")
                _llm_available = False
            # Brief backoff before returning None
            time.sleep(min(_consecutive_failures * 2, 10))
            return None

    else:
        # Default to Ollama
        try:
            print(f"[Ollama] {task}: sending prompt to {OLLAMA_MODEL}")
            response = requests.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            response.raise_for_status()
            result = str(response.json().get("response", "")).strip()
            print(f"[Ollama] {task}: received {len(result)} characters")
            _consecutive_failures = 0
            return result
        except Exception as exc:
            _consecutive_failures += 1
            print(f"[Ollama WARNING] {task}: request failed ({_consecutive_failures}/{_MAX_CONSECUTIVE_FAILURES}): {exc}")
            if _consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                print("[Ollama] Too many consecutive failures — disabling LLM.")
                _llm_available = False
            return None
