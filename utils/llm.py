"""Ollama chat client for LLM calls."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import ollama

from utils.config import LLM_MODEL


class OllamaChat:
    """Minimal wrapper around ollama.chat with JSON helper."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or LLM_MODEL

    def complete(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """Return raw string content."""
        try:
            resp = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "seed": 42,
                    "top_p": 1.0,
                    "repeat_penalty": 1.05,
                },
            )
            return resp.get("message", {}).get("content", "")
        except Exception as exc:
            return f"[LLM error: {exc}]"

    def complete_json(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> Dict[str, Any]:
        """Ask for a JSON object; parse and return dict or {'error': ..., 'raw': ...}."""
        if messages and messages[0].get("role") == "system":
            messages = messages.copy()
            sys_msg = messages[0].copy()
            sys_msg["content"] = sys_msg["content"].rstrip() + (
                "\n\nYou must respond with a single JSON object and nothing else. "
                "Do not use markdown formatting."
            )
            messages[0] = sys_msg

        raw = self.complete(messages, temperature=temperature)
        if raw.startswith("[LLM error:"):
            return {"error": "llm_unavailable", "raw": raw}
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end <= start:
                raise ValueError("no JSON object found")
            return json.loads(raw[start:end])
        except Exception:
            return {"error": "json_parse_failed", "raw": raw}
