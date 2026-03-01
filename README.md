# Legal Contract Q&A — Multi-Agent RAG System

A console-based RAG system that lets you ask natural-language questions about legal contracts and get grounded, cited answers. It flags legal risks, catches contradictions across agreements, and refuses to answer things it shouldn't.

Built with llama3.1 and nomic-embed-text running locally via Ollama — nothing leaves your machine.

## Table of Contents

- [Problem](#problem)
- [Architecture](#architecture)
- [Design Choices](#design-choices)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Production Roadmap](#production-roadmap)

## Problem

Legal professionals need to pull specific facts out of contracts — notice periods, liability caps, governing law, breach timelines — without reading every page. This system takes a small corpus (4 contracts: NDA, vendor agreement, SLA, DPA) and answers questions about them.

It needs to cite its sources, flag financial/legal risks, detect contradictions across documents (e.g., two contracts specifying different governing laws), carry context across conversation turns, and refuse out-of-scope requests like drafting or legal strategy.

## Architecture

Every query goes through this pipeline:

```
User Query + Conversation History
          |
          v
  1. Routing Agent
     classifies intent, picks target docs,
     generates BM25 query + HyDE hypothesis
          |
          v
  2. Hybrid Retrieval
     vector (Chroma) + BM25 keyword search
     → cross-encoder re-ranking
          |
          v
  3. Answer Agent
     grounded answer with [doc:section] citations
          |
          v
  4. Grounding Verifier (CRAG)
     checks each claim against source text
     → if unsupported claims found, retry retrieval once
          |
          v
  5. Risk + Consistency Agents
     (only for risk/cross-doc/summary queries)
          |
          v
  Final output: answer, citations, risks, contradictions
```

The routing agent classifies intent into one of: `fact`, `risk`, `cross_document`, `summary`, `greeting`, or `out_of_scope`. Risk and consistency agents only run on the intents that need them — this prevents the system from tacking on generic risk warnings to simple factual questions.

### Project Structure

```
Legal-QA-Bot/
├── main.py                 # entrypoint
├── build_index.py          # offline index builder
├── pipeline.py             # orchestration (plan → retrieve → answer → verify → risks)
├── repl.py                 # interactive console with session history and caching
├── requirements.txt
│
├── utils/
│   ├── config.py           # all settings (env vars with defaults)
│   ├── llm.py              # Ollama LLM client wrapper
│   └── prompts.py          # all system prompts in one place
│
├── ingestion/
│   ├── loader.py           # loads .txt files from data/
│   └── chunking.py         # section-aware hierarchical chunking
│
├── retrieval/
│   ├── chroma_store.py     # ChromaDB vector store wrapper
│   ├── bm25_index.py       # BM25 keyword index
│   └── retriever.py        # hybrid retrieval + cross-encoder re-ranking
│
├── agents/
│   ├── routing.py          # intent classification + query planning
│   ├── answering.py        # answer generation
│   ├── grounding.py        # claim verification (CRAG)
│   ├── risks.py            # risk extraction
│   └── consistency.py      # cross-document conflict detection
│
├── eval/
│   ├── ground_truth.py     # 17 sample queries with expected answers
│   ├── metrics.py          # custom deterministic metrics
│   ├── run_eval.py         # evaluation runner (custom + Ragas)
│   └── results/            # saved evaluation outputs
│
└── data/                   # contract .txt files
```

## Design Choices

### Chunking: parent–child splits

Most RAG systems chunk documents with a fixed sliding window — take 500 characters, step forward 250, repeat. That works okay for unstructured text, but contracts aren't unstructured. They have numbered sections, sub-clauses, enumerated obligations. A sliding window doesn't care about any of that — it'll happily cut a clause in half or lump two unrelated sections into one chunk.

So instead of a sliding window, we detect section boundaries with regex (`1. Title`, `Section N`, `Article N`) and split along those. Each section becomes a "parent" — the full text of that clause. Then each parent gets broken into smaller "children" (≤400 chars) that respect sentence boundaries and keep enumerated lists like (a), (b), (c) together.

Why bother with two levels? It's a precision vs. context trade-off. Small chunks are better for retrieval — when you embed a 200-char chunk that says "30 days written notice," it'll rank high for a query about notice periods. A 2000-char chunk containing that same sentence plus four other clauses would rank lower because the embedding gets diluted by irrelevant text. But small chunks are bad for answering — if you only show the LLM that 200-char snippet, it's missing the surrounding context: what conditions apply, what definitions are referenced, what exceptions exist.

The parent–child split lets you have both. Search against the small children for precise matches, then at answer time swap in the full parent section so the LLM gets the complete picture. That swap happens in `_enrich` in `pipeline.py`.

Children overlap by 100 chars so sentences near a split boundary still show up in both chunks. Min chunk size is 20 chars — set that low on purpose because some of the most important clauses are very short ("99.5% monthly uptime") and dropping them would hurt retrieval.

### Hybrid retrieval

Pure vector search misses exact terms. Ask about "Section 4.2" or "72 hours" and the embedding might not surface the right chunk because those are specific tokens, not semantic concepts. Pure keyword search (BM25) has the opposite problem — it can't match "breach notification timeline" to a clause that says "notify within 72 hours of discovery."

So we run both and combine their scores: 70% vector, 30% BM25 (configurable via `LEGAL_QA_HYBRID_ALPHA`). Both score sets are min-max normalized before fusion.

### Cross-encoder re-ranking

The hybrid search gives us ~20 candidate chunks. Since we only keep the top 10, ranking quality matters — a relevant chunk at position 15 gets cut and never reaches the LLM.

Vector search uses a bi-encoder: query and chunk are embedded independently and compared by distance. It's fast (chunk vectors are pre-computed at index time) but approximate, because each text gets compressed into a single vector and word-level interactions between query and chunk are lost. A cross-encoder is more accurate — it takes the query and chunk as a single combined input, so the model sees both sides at once and can directly match phrases like "notice period" against "30 days written notice." The cost is that it needs a full forward pass per (query, chunk) pair, which is too slow for the whole corpus but fine for 20 pre-filtered candidates.

We use `ms-marco-MiniLM-L-6-v2` — a small 22M-parameter model trained for passage re-ranking on MS MARCO (a large dataset of real search queries). It's fast enough for CPU and is the standard choice for this pattern.

### HyDE (Hypothetical Document Embeddings)

Questions and documents live in different parts of the embedding space. "What is the notice period?" doesn't embed anywhere near "Either party may terminate this Agreement by providing 30 days' written notice" — they use different words and different sentence structure, even though they're about the same thing.

HyDE bridges this gap. The routing agent generates a hypothetical answer — what the ideal contract clause might say — and uses that as the vector search query instead of the raw question. The hypothesis uses contract-like language, so it lands closer to the actual clause in embedding space.

This happens inside the routing step alongside intent classification and BM25 query generation, all in a single LLM call, so it doesn't add a round trip.

### CRAG (self-correction)

After the answer agent responds, the grounding verifier breaks the answer into claims and checks each one against the source text. If anything isn't supported, it suggests a refined retrieval query and the pipeline retries once.

One retry only. If the information isn't found in two attempts, more tries won't help — they'll just add latency (each retry = 2 LLM calls + 1 retrieval pass).

The obvious weakness: the verifier is also an LLM. If it hallucinates during verification and marks a fabricated claim as "supported," the loop fails silently.

### Intent gating for risk/consistency

Without gating, the risk agent flags generic risks on every query. Ask "what is the notice period?" and you'd get the answer plus a bunch of irrelevant warnings about liability caps. Gating risk and consistency agents to only run on `risk`, `cross_document`, and `summary` intents eliminated this noise and improved `risk_correct` from 58% to 76%.

### Other choices

- **nomic-embed-text** for embeddings — runs locally, 768-dim, 8192-token window. No API keys, no data leaving the machine.
- **llama3.1** for all agents — temperature 0, seed 42 for deterministic output. One model keeps things simple. The trade-off is that routing doesn't need an 8B model and answering could benefit from a bigger one.
- **All prompts in `utils/prompts.py`** — one file to review, edit, and diff. No hunting through five agent files to tune a prompt.
- **Session history** — last 5 turns (10 messages), 5-minute response cache, last 4 messages passed to agents for conversational context.

## Setup

Prerequisites: Python 3.10+ and [Ollama](https://ollama.com) running locally.

```bash
cd Legal-QA-Bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.1
ollama pull nomic-embed-text
```

Build the index (one-time, re-run if you change documents):

```bash
python build_index.py
```

All settings live in `utils/config.py` and can be overridden with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LEGAL_QA_DATA_DIR` | `./data` | Path to contract `.txt` files |
| `LEGAL_QA_LLM_MODEL` | `llama3.1` | Ollama LLM model |
| `LEGAL_QA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `LEGAL_QA_TOP_K` | `10` | Final chunks returned to LLM |
| `LEGAL_QA_HYBRID_ALPHA` | `0.7` | Vector weight (1.0 = vector only, 0.0 = BM25 only) |
| `LEGAL_QA_RERANK_TOP_K` | `20` | Candidates before cross-encoder re-ranking |
| `LEGAL_QA_CHILD_MAX_CHARS` | `400` | Max child chunk size |
| `LEGAL_QA_CHUNK_MIN_CHARS` | `20` | Min chunk size (below this, chunk is dropped) |
| `LEGAL_QA_MAX_HISTORY_TURNS` | `5` | Conversation turns retained |
| `LEGAL_QA_CACHE_TTL_SEC` | `300` | Response cache TTL |
| `LEGAL_QA_LOG_REQUESTS_TO_CONSOLE` | `0` | Set to `1` to enable per-request logging |

## Usage

```bash
cd Legal-QA-Bot
source .venv/bin/activate
python main.py
```

Type questions in plain English. The system returns an answer with citations, and optionally risks and contradictions depending on the query type.

```
Legal Contract Q&A
Ask questions about the contracts. Type 'exit' or 'quit' to end.

Query: What is the notice period for terminating the NDA?

--- Answer ---
Either party may terminate the NDA by providing 30 days' written notice
to the other party [nda_acme_vendor:7].

References:
  [nda_acme_vendor:7]: Termination

Query: Are there any conflicting governing laws across agreements?

--- Answer ---
Yes. The NDA is governed by the laws of the State of California
[nda_acme_vendor:10], while the Vendor Services Agreement is governed
by the laws of the State of New York [vendor_services_agreement:12].

References:
  [nda_acme_vendor:10]: Governing Law
  [vendor_services_agreement:12]: Governing Law

Risks:
  ⚠ Conflicting governing laws (nda_acme_vendor:10 vs vendor_services_agreement:12)

Inconsistencies:
  [Governing Law] nda_acme_vendor vs vendor_services_agreement:
    A: State of California
    B: State of New York
```

## Evaluation

We evaluate with two complementary approaches: fast deterministic metrics (no LLM calls) and Ragas LLM-as-judge metrics.

### Custom metrics

These are cheap, reproducible, and run in a few minutes. Each one targets a specific failure mode:

- **key_phrase_recall** — what fraction of expected key phrases appear in the answer? This catches the case where the system returns a vaguely relevant response but misses the actual facts (e.g., says "there is a notice period" but doesn't mention "30 days").

- **citation_present** — does the answer contain at least one `[doc:section]` citation? A RAG system that doesn't cite its sources is just a chatbot. This is a binary check: either the answer points back to a document or it doesn't.

- **retrieval_hit_rate** — what fraction of the expected source documents showed up in retrieval results? If this is low, the problem is in retrieval (chunking, embeddings, search), not generation. If this is high but answers are still wrong, the problem is in the LLM's use of context. This metric tells you where to look.

- **risk_correct** — for queries where risks should (or shouldn't) be flagged, did the system get it right? This catches both false positives (flagging risks on a simple factual question) and false negatives (missing obvious risks on a risk-focused query). Before intent gating, this was at 58%.

- **scope_correct** — did the system refuse out-of-scope queries (drafting requests, legal strategy) and *not* refuse in-scope ones? A system that answers everything is unsafe; a system that refuses everything is useless. This measures the boundary.

### Ragas metrics

These use llama3.1 as a judge and are slower (~30 min), but they catch things the deterministic metrics can't:

- **faithfulness** — is every claim in the answer actually supported by the retrieved chunks? This is the hallucination metric. A high key_phrase_recall with low faithfulness means the system gets the right facts but also invents extras.

- **answer_relevancy** — is the answer actually about what the user asked? Retrieval might pull the right documents, but if the LLM latches onto an irrelevant part of the context, this catches it.

- **context_precision** — are the most relevant chunks ranked at the top of the retrieval results? Even if the right document appears somewhere in the results, it matters whether it's ranked #1 or #10 — the LLM pays more attention to earlier context.

### Results

Evaluated on 17 queries covering fact extraction, risk identification, cross-document comparison, summarization, and out-of-scope handling:

| Metric | Score |
|--------|-------|
| Key Phrase Recall | 0.71 |
| Retrieval Hit Rate | 0.98 |
| Citation Present | 80% |
| Risk Correct | 76% |
| Scope Correct | 94% |
| Answer Relevancy (Ragas) | 0.86 |
| Context Precision (Ragas) | 0.78 |
| Faithfulness (Ragas) | 0.47* |

*Faithfulness is unreliable here. Ragas faithfulness decomposes the answer into atomic claims, then asks the judge LLM to verify each one — that's multiple chained LLM calls per query. llama3.1 frequently times out or produces inconsistent decompositions in this setting. This is a known issue with running Ragas on local models; the metric would be meaningful with GPT-4 as judge but isn't trustworthy here.

### Running it

```bash
python -m eval.run_eval              # custom metrics only (~5 min)
python -m eval.run_eval --ragas      # custom + Ragas (~30 min)
```

Results go to `eval/results/` — JSON with raw outputs, a summary file, and a CSV with per-query scores.

### Evaluation caveats

The query set is small (17 queries) and the ground truths are hand-crafted, not reviewed by legal experts. Multi-turn quality isn't measured — each eval query runs independently. Take the numbers as directional, not definitive.

## Limitations

**Retrieval:** Chunking relies on regex section detection — contracts with unusual formatting (tables, nested appendices, inline headers) may chunk poorly. The BM25 index rebuilds in-memory on every startup since only the Chroma vector index is persisted.

**Grounding:** No confidence score is surfaced to the user. The grounding verifier helps internally, but users can't tell whether the system is confident or guessing. And since the verifier is the same LLM that generated the answer, it can fail silently.

**LLM:** JSON output is fragile — llama3.1 sometimes wraps it in markdown fences or adds trailing text. The parser handles most cases with `find("{")`/`rfind("}")` extraction, but edge cases slip through. Citation consistency hovers around 80% due to prompt compliance limitations at this model size.

**System:** Queries make up to 5 sequential LLM calls, so latency is 30-60s on CPU. Risk and consistency agents could run in parallel but don't yet. Only `.txt` files are supported — no PDF or DOCX. No prompt injection detection, no access control, no incremental indexing.

## Production Roadmap

What we'd build first if this were going to production:

**Immediate:** PDF/DOCX support with layout-aware parsing — real legal corpora are never plain text. Incremental indexing so adding a document doesn't require a full rebuild. Confidence scoring so users know when to trust vs. verify. Prompt injection detection before the routing agent.

**Next:** Parallel execution of risk + consistency agents (they're independent, easy win for latency). Streaming responses so users see tokens while analysis runs in the background. Per-request observability with input/output logging, latency, and retrieval scores. Document versioning for audit trails.

**Eventually:** Fine-tuned legal embeddings, human-annotated eval set, user feedback loop, multi-tenant access control, and prompt version tracking tied to eval metrics.
