"""Interactive REPL with session history, response cache, and request logging."""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from utils.config import CACHE_TTL_SEC, MAX_HISTORY_TURNS


def _cache_key(query: str, last_assistant: str) -> str:
    return f"{query.strip().lower()}|{last_assistant}"


def _format_refs(refs: List[Dict]) -> str:
    if not refs:
        return ""
    lines = [f"  [{r.get('doc_id', '')}:{r.get('section_path', '')}]: {r.get('section_title', '')}" for r in refs]
    return "References:\n" + "\n".join(lines)


def _format_risks(risks: List[Dict]) -> str:
    if not risks:
        return ""
    return "Risks:\n" + "\n".join(f"  ⚠ {r.get('label', '')} ({r.get('reference', '')})" for r in risks)


def _format_inconsistencies(items: List[Dict]) -> str:
    if not items:
        return ""
    lines = []
    for it in items:
        lines.append(f"  [{it.get('topic', '')}] {it.get('doc_a', '')} vs {it.get('doc_b', '')}:")
        lines.append(f"    A: {(it.get('claim_a', '') or '')[:80]}")
        lines.append(f"    B: {(it.get('claim_b', '') or '')[:80]}")
    return "Inconsistencies:\n" + "\n".join(lines)


def run_loop(
    process_fn: Callable[[str, Optional[List[Dict]], Dict], Dict],
    retriever=None,
    chunks=None,
) -> None:
    """Run interactive loop. process_fn(question, history, extra) -> result dict."""
    history: List[Dict[str, str]] = []
    cache: Dict[str, tuple] = {}
    extra = {"retriever": retriever, "chunks": chunks, "chunk_by_id": {c["chunk_id"]: c for c in (chunks or [])}}

    print("Legal Contract Q&A")
    print("Ask questions about the contracts. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            q = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        last_asst = history[-1]["content"] if history and history[-1].get("role") == "assistant" else ""
        key = _cache_key(q, last_asst)
        now = time.time()
        if key in cache:
            cached_val, cached_ts = cache[key]
            if now - cached_ts < CACHE_TTL_SEC:
                result = cached_val
            else:
                del cache[key]
                result = process_fn(q, history, extra)
                cache[key] = (result, now)
        else:
            result = process_fn(q, history, extra)
            cache[key] = (result, now)

        _display(result)

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": result.get("answer", "")})
        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-MAX_HISTORY_TURNS * 2 :]


def _display(result: Dict) -> None:
    print("\n--- Answer ---")
    print(result.get("answer", ""))

    refs = _format_refs(result.get("references", []))
    risks = _format_risks(result.get("risks", []))
    incon = _format_inconsistencies(result.get("inconsistencies", []))

    if refs:
        print(f"\n{refs}")
    if risks:
        print(f"\n{risks}")
    if incon:
        print(f"\n{incon}")
    print()
