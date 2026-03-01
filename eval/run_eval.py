"""
Evaluation runner.

Usage (from Legal-QA-Bot/):
  python -m eval.run_eval              # custom metrics only (fast)
  python -m eval.run_eval --ragas      # custom + Ragas (slow, needs LLM)

Outputs:
  eval/results/eval_results.json   – raw pipeline outputs + per-query scores
  eval/results/eval_summary.json   – aggregate metrics
  eval/results/eval_report.csv     – one row per query with all scores
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.config import CHILD_MAX_CHARS, CHUNK_MIN_CHARS, CHUNK_OVERLAP, DATA_DIR
from eval.ground_truth import SAMPLE_QUERIES
from eval.metrics import aggregate, score_one
from ingestion.chunking import chunk_all
from ingestion.loader import load_corpus
from retrieval.retriever import build_retriever
from pipeline import process_turn

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _run_pipeline_queries(retriever, chunks, chunk_by_id):
    """Run each sample query through the pipeline and collect results."""
    outputs = []
    total = len(SAMPLE_QUERIES)
    for i, q in enumerate(SAMPLE_QUERIES):
        query = q["query"]
        print(f"  [{i + 1}/{total}] {query[:70]}")
        t0 = time.time()
        result = process_turn(
            query,
            history=None,
            retriever=retriever,
            chunks=chunks,
            chunk_by_id=chunk_by_id,
        )
        elapsed = time.time() - t0
        outputs.append({
            "meta": q,
            "result": result,
            "elapsed_s": round(elapsed, 2),
        })
    return outputs


def _compute_custom_scores(outputs):
    """Score each output with deterministic custom metrics."""
    scores = []
    for o in outputs:
        s = score_one(o["meta"], o["result"])
        s["elapsed_s"] = o["elapsed_s"]
        scores.append(s)
    return scores


def _run_ragas(outputs):
    """Run Ragas evaluation (Faithfulness, AnswerRelevancy, ContextPrecision)."""
    try:
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas import evaluate
        from datasets import Dataset
    except ImportError as e:
        print(f"  Ragas dependencies missing: {e}")
        print("  Install: pip install ragas datasets langchain-ollama")
        return None

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        LangchainEmbeddingsWrapper = None

    try:
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
    except ImportError:
        try:
            from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision
        except ImportError:
            print("  Could not import Ragas metrics.")
            return None

    from utils.config import LLM_MODEL, EMBED_MODEL

    questions = [o["meta"]["query"] for o in outputs]
    answers = [o["result"].get("answer", "") for o in outputs]
    references = [o["meta"]["reference"] for o in outputs]

    contexts = []
    for o in outputs:
        refs = o["result"].get("references", [])
        ctx = [f"{r.get('doc_id', '')}:{r.get('section_path', '')}: {r.get('section_title', '')}" for r in refs]
        if not ctx:
            ctx = [o["result"].get("answer", "")[:200]]
        contexts.append(ctx)

    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": references,
    })

    llm = LangchainLLMWrapper(ChatOllama(model=LLM_MODEL, temperature=0.0))
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    if LangchainEmbeddingsWrapper:
        emb = LangchainEmbeddingsWrapper(emb)

    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision()]

    print("  Running Ragas evaluation (this may take several minutes) ...")
    result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=emb, show_progress=True)
    return result


def _save_results(outputs, custom_scores, agg, ragas_result):
    """Save all results to eval/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_data = []
    for o, s in zip(outputs, custom_scores):
        results_data.append({
            "query_id": o["meta"]["id"],
            "query": o["meta"]["query"],
            "reference": o["meta"]["reference"],
            "answer": o["result"].get("answer", ""),
            "references": o["result"].get("references", []),
            "risks": o["result"].get("risks", []),
            "inconsistencies": o["result"].get("inconsistencies", []),
            "out_of_scope": o["result"].get("out_of_scope", False),
            "scores": s,
        })
    with open(RESULTS_DIR / "eval_results.json", "w") as f:
        json.dump({"results": results_data}, f, indent=2, default=str)
    print(f"  Saved: {RESULTS_DIR / 'eval_results.json'}")

    summary = {"custom_metrics": agg}
    if ragas_result is not None:
        try:
            if isinstance(ragas_result, dict):
                summary["ragas_metrics"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in ragas_result.items()}
            elif hasattr(ragas_result, "to_pandas"):
                import pandas as pd
                df = ragas_result.to_pandas()
                numeric = df.select_dtypes(include="number")
                summary["ragas_metrics"] = {col: round(numeric[col].mean(), 4) for col in numeric.columns}
        except Exception:
            summary["ragas_metrics"] = str(ragas_result)
    with open(RESULTS_DIR / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {RESULTS_DIR / 'eval_summary.json'}")

    try:
        import pandas as pd
        rows = []
        for o, s in zip(outputs, custom_scores):
            row = {
                "query_id": o["meta"]["id"],
                "query": o["meta"]["query"],
                "answer_preview": o["result"].get("answer", "")[:200],
                "elapsed_s": o["elapsed_s"],
                **{k: v for k, v in s.items() if k not in ("query_id", "query")},
            }
            rows.append(row)

        if ragas_result is not None and hasattr(ragas_result, "to_pandas"):
            rdf = ragas_result.to_pandas()
            ragas_cols = [c for c in rdf.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
            for i, row in enumerate(rows):
                if i < len(rdf):
                    for col in ragas_cols:
                        row[f"ragas_{col}"] = rdf.iloc[i][col]

        df = pd.DataFrame(rows)
        df.to_csv(RESULTS_DIR / "eval_report.csv", index=False)
        print(f"  Saved: {RESULTS_DIR / 'eval_report.csv'}")
    except ImportError:
        print("  pandas not available — skipping CSV report.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Legal QA pipeline")
    parser.add_argument("--ragas", action="store_true", help="Also run Ragas LLM-as-judge metrics")
    args = parser.parse_args()

    print("Loading corpus and building retriever ...")
    corpus = load_corpus(DATA_DIR)
    chunks = chunk_all(corpus, child_max_chars=CHILD_MAX_CHARS, overlap_chars=CHUNK_OVERLAP, min_chars=CHUNK_MIN_CHARS)
    retriever = build_retriever(chunks)
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    print(f"\nRunning {len(SAMPLE_QUERIES)} sample queries ...")
    outputs = _run_pipeline_queries(retriever, chunks, chunk_by_id)

    print("\nComputing custom metrics ...")
    custom_scores = _compute_custom_scores(outputs)
    agg = aggregate(custom_scores)

    print("\n--- Custom Metric Summary ---")
    for k, v in agg.items():
        print(f"  {k}: {v:.3f}")

    ragas_result = None
    if args.ragas:
        print("\nRunning Ragas evaluation ...")
        ragas_result = _run_ragas(outputs)
        if ragas_result is not None:
            print("\n--- Ragas Summary ---")
            try:
                if isinstance(ragas_result, dict):
                    for k, v in ragas_result.items():
                        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
                elif hasattr(ragas_result, "to_pandas"):
                    df = ragas_result.to_pandas()
                    numeric = df.select_dtypes(include="number")
                    for col in numeric.columns:
                        print(f"  avg_{col}: {numeric[col].mean():.4f}")
            except Exception as e:
                print(f"  (could not format Ragas result: {e})")

    print("\nSaving results ...")
    _save_results(outputs, custom_scores, agg, ragas_result)
    print("\nDone.")


if __name__ == "__main__":
    main()
