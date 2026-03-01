"""
Custom evaluation metrics — fast, no LLM calls needed.

These complement Ragas (LLM-as-judge) with deterministic, interpretable checks:
  1. key_phrase_recall  – does the answer mention expected facts?
  2. citation_present   – does the answer cite sources?
  3. retrieval_hit_rate – did retrieval fetch the right documents?
  4. risk_detection     – for risk queries, did the system flag risks?
  5. scope_handling     – for out-of-scope queries, did it correctly refuse?
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


def key_phrase_recall(answer: str, key_phrases: List[str]) -> float:
    """Fraction of expected key phrases found in the answer (case-insensitive)."""
    if not key_phrases:
        return 1.0
    lower = answer.lower()
    hits = sum(1 for kp in key_phrases if kp.lower() in lower)
    return hits / len(key_phrases)


def citation_present(answer: str) -> bool:
    """True if the answer contains at least one citation pattern like [doc_id:section]."""
    return bool(re.search(r"\[.+?:.+?\]", answer))


def retrieval_hit_rate(retrieved_doc_ids: List[str], expected_doc_ids: List[str]) -> float:
    """Fraction of expected documents that appeared in retrieval results."""
    if not expected_doc_ids:
        return 1.0
    found = set(retrieved_doc_ids)
    hits = sum(1 for d in expected_doc_ids if d in found)
    return hits / len(expected_doc_ids)


def risk_detection(risks_returned: List[Dict], expect_risks: bool) -> bool:
    """True if risk presence/absence matches expectation."""
    has_risks = len(risks_returned) > 0
    return has_risks == expect_risks


def scope_handling(answer: str, out_of_scope_flag: bool, expected_intent: str) -> bool:
    """True if the system correctly handled out-of-scope queries."""
    if expected_intent == "out_of_scope":
        return out_of_scope_flag
    return not out_of_scope_flag


def score_one(query_meta: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single query result against its ground truth metadata."""
    answer = result.get("answer", "")
    refs = result.get("references", [])
    risks = result.get("risks", [])
    oos = result.get("out_of_scope", False)
    expected_intent = query_meta.get("expected_intent", "fact")

    retrieved_docs = list({r.get("doc_id", "") for r in refs if r.get("doc_id")})
    expected_docs = query_meta.get("expected_docs", [])
    key_phrases = query_meta.get("key_phrases", [])

    is_oos_query = expected_intent == "out_of_scope"

    return {
        "query_id": query_meta["id"],
        "query": query_meta["query"],
        "key_phrase_recall": key_phrase_recall(answer, key_phrases) if not is_oos_query else None,
        "citation_present": citation_present(answer) if not is_oos_query else None,
        "retrieval_hit_rate": retrieval_hit_rate(retrieved_docs, expected_docs) if not is_oos_query else None,
        "risk_correct": risk_detection(risks, query_meta.get("expect_risks", False)),
        "scope_correct": scope_handling(answer, oos, expected_intent),
    }


def aggregate(scores: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute mean of each metric across scored queries."""
    metrics = ["key_phrase_recall", "retrieval_hit_rate"]
    bool_metrics = ["citation_present", "risk_correct", "scope_correct"]

    agg: Dict[str, float] = {}
    for m in metrics:
        vals = [s[m] for s in scores if s[m] is not None]
        agg[f"avg_{m}"] = sum(vals) / len(vals) if vals else 0.0

    for m in bool_metrics:
        vals = [s[m] for s in scores if s[m] is not None]
        agg[f"rate_{m}"] = sum(1 for v in vals if v) / len(vals) if vals else 0.0

    return agg
