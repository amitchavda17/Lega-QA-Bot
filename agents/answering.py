from __future__ import annotations

from typing import Dict, List, Optional

from utils.llm import OllamaChat
from utils.prompts import ANSWER


def _format_context(chunks: List[Dict]) -> str:
    parts = []
    for idx, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        doc_id = meta.get("doc_id", "")
        sec = meta.get("section_path", "")
        title = meta.get("section_title", "")
        text = c.get("text", "")
        parts.append(f"[{idx}] doc_id={doc_id} section={sec} title={title}\n{text}")
    return "\n\n".join(parts)


def draft_answer(
    question: str,
    chunks: List[Dict],
    history: Optional[List[Dict[str, str]]] = None,
    client: Optional[OllamaChat] = None,
) -> str:
    """Produce a grounded answer with inline citations."""
    chat = client or OllamaChat()

    history_block = ""
    if history:
        last = history[-4:]
        history_block = "\n".join(f"{m['role']}: {m['content']}" for m in last)

    ctx = _format_context(chunks)
    user_msg = (
        (f"Recent dialogue:\n{history_block}\n\n" if history_block else "")
        + "Relevant contract excerpts:\n"
        + ctx
        + "\n\nQuestion:\n"
        + question
        + "\n\nWrite a concise answer following the rules."
    )

    messages = [{"role": "system", "content": ANSWER}, {"role": "user", "content": user_msg}]
    return chat.complete(messages)
