from __future__ import annotations

from typing import Dict, List, Optional

from utils.llm import OllamaChat
from utils.prompts import RISKS


def extract_risks(
    answer: str,
    chunks: List[Dict],
    client: Optional[OllamaChat] = None,
) -> List[Dict[str, str]]:
    """Return a list of risk items derived from answer + context."""
    chat = client or OllamaChat()

    excerpts = []
    for idx, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        doc_id = meta.get("doc_id", "")
        sec = meta.get("section_path", "")
        text = c.get("text", "")[:600]
        excerpts.append(f"[{idx}] {doc_id}:{sec}\n{text}")
    ctx = "\n\n".join(excerpts)

    user_content = (
        "Answer given to the user:\n"
        + answer
        + "\n\nSupporting excerpts:\n"
        + ctx
        + "\n\nReturn only the JSON object."
    )

    messages = [{"role": "system", "content": RISKS}, {"role": "user", "content": user_content}]
    out = chat.complete_json(messages)

    if not isinstance(out, dict) or "error" in out:
        return []

    items = out.get("items", [])
    results: List[Dict[str, str]] = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict) and it.get("label") and it.get("reference"):
                results.append({"label": str(it["label"]), "reference": str(it["reference"])})
    return results
