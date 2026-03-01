from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils.llm import OllamaChat
from utils.prompts import ROUTING


def plan_retrieval(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    client: Optional[OllamaChat] = None,
) -> Dict[str, Any]:
    """Classify intent, document scope, and generate refined and HyDE queries."""
    chat = client or OllamaChat()

    history_block = ""
    if history:
        last = history[-4:]
        fragments = [f"{m['role']}: {m['content']}" for m in last]
        history_block = "\n".join(fragments)

    user_prompt = (
        "Conversation (most recent first):\n"
        f"{history_block or '(none)'}\n\n"
        f"User question:\n{question}\n\n"
        "Return only the JSON object."
    )

    messages = [{"role": "system", "content": ROUTING}, {"role": "user", "content": user_prompt}]
    out = chat.complete_json(messages)

    if not isinstance(out, dict) or "error" in out:
        return {"intent": "fact", "doc_scope": "all", "refined_query": question, "hyde_query": question}

    return {
        "intent": out.get("intent", "fact"),
        "doc_scope": out.get("doc_scope", "all"),
        "refined_query": out.get("refined_query", question),
        "hyde_query": out.get("hyde_query", question),
    }
