from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple


def load_corpus(data_dir: Path) -> List[Tuple[str, str]]:
    """
    Load all .txt files from data_dir.

    Returns list of (raw_text, doc_id) where doc_id is the file stem.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    docs: List[Tuple[str, str]] = []
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        docs.append((text, path.stem))
    return docs


def iter_doc_ids(data_dir: Path) -> Iterable[str]:
    """Yield doc_ids (stems) for .txt files in data_dir."""
    for path in sorted(data_dir.glob("*.txt")):
        yield path.stem

