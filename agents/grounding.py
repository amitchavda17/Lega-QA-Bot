from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils.llm import OllamaChat
from utils.prompts import GROUNDING


def assess_grounding(
    answer: str,
    chunks: List[Dict],
    client: Optional[OllamaChat] = None,
) -> Dict[str, Any]:
    """Check if the answer is grounded in the given chunks."""
    chat = client or OllamaChat()

    excerpts = []
    for idx, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        doc_id = meta.get("doc_id", "")
        sec = meta.get("section_path", "")
        text = c.get("text", "")
        excerpts.append(f"[{idx}] {doc_id}:{sec}\n{text}")
    ctx = "\n\n".join(excerpts)

    user_content = (
        "Answer to verify:\n"
        + answer
        + "\n\nExcerpts used as evidence:\n"
        + ctx
        + "\n\nReturn only the JSON object."
    )

    messages = [{"role": "system", "content": GROUNDING}, {"role": "user", "content": user_content}]
    out = chat.complete_json(messages)

    if not isinstance(out, dict) or "error" in out:
        return {"all_supported": True, "unsupported_claims": [], "suggested_query": ""}

    return {
        "all_supported": bool(out.get("all_supported", True)),
        "unsupported_claims": list(out.get("unsupported_claims", [])),
        "suggested_query": str(out.get("suggested_query", "")),
    }
