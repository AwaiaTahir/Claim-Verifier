"""Project setup helper for downloading reusable local resources."""

from __future__ import annotations

from typing import Iterable

import nltk

from config import FEVER_CONFIG, FEVER_DATASET, FEVER_SPLIT, NLTK_PACKAGES


def download_nltk_packages(packages: Iterable[str]) -> None:
    """Download required NLTK resources if they are not already available."""
    for package in packages:
        try:
            print(f"[setup] Checking NLTK package: {package}")
            nltk.download(package, quiet=True)
        except Exception as exc:
            print(f"[setup WARNING] Could not download NLTK package {package}: {exc}")


def warm_fever_cache() -> None:
    """Touch the FEVER dataset cache without building indexes or duplicating data."""
    try:
        from datasets import load_dataset

        print("[setup] Checking FEVER dataset availability")
        load_dataset(FEVER_DATASET, FEVER_CONFIG, split=f"{FEVER_SPLIT}[:1]")
        print("[setup] FEVER dataset cache is ready")
    except Exception as exc:
        print(f"[setup WARNING] Could not prepare FEVER dataset cache: {exc}")


def main() -> None:
    """Run idempotent setup tasks for NLTK and the FEVER dataset cache."""
    download_nltk_packages(NLTK_PACKAGES)
    warm_fever_cache()
    print("[setup] Setup complete")


if __name__ == "__main__":
    main()
