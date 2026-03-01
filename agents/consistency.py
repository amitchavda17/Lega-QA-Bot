from __future__ import annotations

from typing import Dict, List, Optional

from utils.llm import OllamaChat
from utils.prompts import CONSISTENCY


def find_inconsistencies(
    chunks: List[Dict],
    client: Optional[OllamaChat] = None,
) -> List[Dict[str, str]]:
    """Detect cross-document conflicts/inconsistencies."""
    chat = client or OllamaChat()

    doc_ids = {c.get("metadata", {}).get("doc_id", "") for c in chunks}
    if len([d for d in doc_ids if d]) < 2:
        return []

    excerpts = []
    for idx, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        doc_id = meta.get("doc_id", "")
        sec = meta.get("section_path", "")
        text = c.get("text", "")[:800]
        excerpts.append(f"[{idx}] {doc_id}:{sec}\n{text}")
    ctx = "\n\n".join(excerpts)

    user_content = (
        "Contract excerpts from multiple documents:\n"
        + ctx
        + "\n\nReturn only the JSON object describing any conflicts."
    )

    messages = [{"role": "system", "content": CONSISTENCY}, {"role": "user", "content": user_content}]
    out = chat.complete_json(messages)

    if not isinstance(out, dict) or "error" in out:
        return []

    issues = out.get("issues", [])
    results: List[Dict[str, str]] = []
    if isinstance(issues, list):
        for it in issues:
            if not isinstance(it, dict):
                continue
            a, b, ca, cb = it.get("doc_a"), it.get("doc_b"), it.get("claim_a"), it.get("claim_b")
            if a and b and ca and cb:
                results.append({
                    "doc_a": str(a),
                    "doc_b": str(b),
                    "claim_a": str(ca),
                    "claim_b": str(cb),
                    "topic": str(it.get("topic", "general")),
                })
    return results
