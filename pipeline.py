"""Pipeline: plan → retrieve → answer → verify (1 retry) → risks → conflicts."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agents.answering import draft_answer
from agents.consistency import find_inconsistencies
from agents.grounding import assess_grounding
from agents.risks import extract_risks
from agents.routing import plan_retrieval
from utils.config import LOG_REQUESTS_TO_CONSOLE, TOP_K
from utils.llm import OllamaChat
from utils.prompts import GREETING, OUT_OF_SCOPE
from retrieval.retriever import HybridRetriever

_EMPTY = {"answer": "", "references": [], "risks": [], "inconsistencies": [], "out_of_scope": False}


def process_turn(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    retriever: Optional[HybridRetriever] = None,
    chunks: Optional[List[Dict]] = None,
    chunk_by_id: Optional[Dict[str, Dict]] = None,
    client: Optional[OllamaChat] = None,
) -> Dict[str, Any]:
    """Run full pipeline and return {answer, references, risks, inconsistencies, out_of_scope}."""
    t0 = time.time()
    chat = client or OllamaChat()

    plan = plan_retrieval(question, history=history, client=chat)
    intent = plan.get("intent", "fact")
    doc_scope = plan.get("doc_scope", "all")
    refined = plan.get("refined_query", question)
    hyde = plan.get("hyde_query", question)

    if intent == "greeting":
        return _early_exit(GREETING, t0, out_of_scope=False)

    if intent == "out_of_scope":
        return _early_exit(OUT_OF_SCOPE, t0, out_of_scope=True)

    if not retriever or not chunks:
        return {**_EMPTY, "answer": "Index not loaded. Run: python build_index.py"}

    by_id = chunk_by_id or {c["chunk_id"]: c for c in chunks}
    doc_filter = None if doc_scope == "all" else doc_scope

    retrieved = retriever.retrieve(refined, k=TOP_K, vector_query=hyde, doc_id_filter=doc_filter)
    if not retrieved:
        return _early_exit("No relevant contract sections were found.", t0)

    context = _enrich(retrieved, by_id)
    answer = draft_answer(question, context, history=history, client=chat)

    grade = assess_grounding(answer, context, client=chat)
    if not grade.get("all_supported") and grade.get("suggested_query"):
        retry_q = grade["suggested_query"]
        retry_retrieved = retriever.retrieve(retry_q, k=TOP_K, vector_query=retry_q, doc_id_filter=doc_filter)
        if retry_retrieved:
            context = _enrich(retry_retrieved, by_id)
            answer = draft_answer(question, context, history=history, client=chat)
            retrieved = retry_retrieved

    _RISK_INTENTS = {"risk", "cross_document", "summary"}
    risks = extract_risks(answer, context, client=chat) if intent in _RISK_INTENTS else []

    doc_ids = {r.get("metadata", {}).get("doc_id", "") for r in retrieved}
    inconsistencies = find_inconsistencies(context, client=chat) if intent in _RISK_INTENTS and len([d for d in doc_ids if d]) >= 2 else []

    refs = _build_refs(retrieved)
    elapsed = time.time() - t0
    _log(question, len(refs), len(risks), elapsed)
    return {"answer": answer, "references": refs, "risks": risks, "inconsistencies": inconsistencies, "out_of_scope": False}


def _enrich(retrieved: List[Dict], by_id: Dict[str, Dict]) -> List[Dict]:
    """Swap child chunk text with parent text when available for richer LLM context."""
    out = []
    for r in retrieved:
        full = by_id.get(r.get("chunk_id"), {})
        if full.get("chunk_kind") == "child" and full.get("parent_text"):
            out.append({**r, "text": full["parent_text"]})
        else:
            out.append(r)
    return out


def _build_refs(retrieved: List[Dict]) -> List[Dict]:
    seen = set()
    refs = []
    for r in retrieved:
        meta = r.get("metadata", {})
        key = (meta.get("doc_id", ""), meta.get("section_path", ""))
        if key not in seen and key[0]:
            seen.add(key)
            refs.append({"doc_id": key[0], "section_path": key[1], "section_title": meta.get("section_title", "")})
    return refs


def _early_exit(answer: str, t0: float, out_of_scope: bool = False) -> Dict[str, Any]:
    elapsed = time.time() - t0
    _log(answer[:40], 0, 0, elapsed)
    return {"answer": answer, "references": [], "risks": [], "inconsistencies": [], "out_of_scope": out_of_scope}


def _log(preview: str, ref_count: int, risk_count: int, elapsed: float):
    if not LOG_REQUESTS_TO_CONSOLE:
        return
    preview = (preview[:60] + "...") if len(preview) > 60 else preview
    print(f"[req] q={preview} elapsed={elapsed:.1f}s refs={ref_count} risks={risk_count}")
