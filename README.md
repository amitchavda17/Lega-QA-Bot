# Legal Contract Q&A — Multi-Agent RAG System

A console-based multi-agent RAG system for querying and analyzing legal contracts. Users ask natural language questions and receive grounded, cited answers with risk indicators and cross-document conflict detection.

---

## Table of Contents

1. [Problem Overview](#1-problem-overview)
2. [Architecture](#2-architecture)
3. [Design Choices and Trade-offs](#3-design-choices-and-trade-offs)
4. [Setup Instructions](#4-setup-instructions)
5. [How to Run](#5-how-to-run)
6. [Evaluation](#6-evaluation)
7. [Known Limitations](#7-known-limitations)
8. [Production Roadmap](#8-production-roadmap)

---

## 1. Problem Overview

Legal professionals need to extract specific information from contracts — notice periods, liability caps, governing law, breach notification timelines — without reading every page. This system takes a small corpus of legal documents (NDAs, vendor agreements, SLAs, DPAs) and lets users ask questions in plain English.

The system must:

- Return answers **grounded only in the provided documents**, with inline citations
- Surface **legal and financial risks** (liability exposure, indemnification gaps, strict deadlines)
- Detect **contradictions** across agreements (e.g., conflicting governing laws)
- Support **multi-turn conversation** with context carry-over
- Correctly **refuse** out-of-scope requests (drafting, legal strategy)

The corpus contains four contracts:

| Document | Key Topics |
|----------|-----------|
| NDA (Acme Corp ↔ Vendor XYZ) | Confidentiality, termination, notice periods |
| Vendor Services Agreement | Liability, indemnification, governing law |
| Service Level Agreement (SLA) | Uptime commitments, remedies, service credits |
| Data Processing Agreement (DPA) | Data breach notification, subprocessors, compliance |

---

## 2. Architecture

### Pipeline Flow

Every user query follows this path:

```
User Query + Conversation History
          |
          v
+----------------------------+
| 1. Routing Agent           |
|    intent, doc_scope,      |
|    refined_query, HyDE     |
+----------------------------+
          |
          v
+----------------------------+
| 2. Hybrid Retrieval        |
|    Vector (Chroma) + BM25  |
|    Cross-encoder re-rank   |
+----------------------------+
          |
          v
+----------------------------+
| 3. Answer Agent            |
|    Cited, grounded answer  |
+----------------------------+
          |
          v
+----------------------------+
| 4. Grounding Verifier      |-----> if unsupported claims:
|    (CRAG self-correction)  |       retry retrieval (1x)
+----------------------------+
          |
          v
+----------------------------+
| 5. Risk + Consistency      |  (only for risk/cross-doc/summary intents)
|    Agents                  |
+----------------------------+
          |
          v
  Answer | Citations | Risks | Contradictions
```

### Agents

| Agent | Responsibility | Output |
|-------|---------------|--------|
| **Routing** | Classifies intent (fact, risk, cross_document, summary, greeting, out_of_scope), selects target document(s), generates a BM25-optimized query and a HyDE hypothesis | JSON: intent, doc_scope, refined_query, hyde_query |
| **Answer** | Produces a grounded answer using only retrieved excerpts, with mandatory inline citations `[doc_id §section]` | Plain text with citations |
| **Grounding Verifier** | Breaks the answer into claims and checks each against the excerpts. If unsupported claims exist, suggests a refined retrieval query | JSON: all_supported, unsupported_claims, suggested_query |
| **Risk** | Identifies material legal/financial risks to Acme Corp from the excerpts | JSON: list of {label, reference} |
| **Consistency** | Compares clauses across documents and flags true conflicts (mutually incompatible obligations) | JSON: list of {doc_a, doc_b, claim_a, claim_b, topic} |

Risk and consistency agents only run when the intent is `risk`, `cross_document`, or `summary` — this prevents spurious risk flagging on simple factual questions.

### Project Structure

```
legal_qa/
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

---

## 3. Design Choices and Trade-offs

### Chunking: Section-Aware Hierarchical

Legal contracts are structured documents — clauses, sub-clauses, numbered lists. Naive fixed-size chunking splits these semantic units arbitrarily.

**Approach:**
- Parse document structure using regex patterns for numbered sections (`1. Title`), `Section N`, `Article N`, and ALL-CAPS headings
- Each section becomes a **parent chunk** (full context for the LLM)
- Parent sections are split into **child chunks** (≤400 chars) for precise vector retrieval
- Sentence-aware splitting ensures no sentence is cut mid-way
- List-aware splitting keeps enumerated items (a), (b), (c) together
- Overlap of 100 chars between consecutive children preserves cross-boundary context
- Minimum chunk size of 20 chars (set deliberately low to avoid dropping short but critical clauses like "99.5% monthly uptime")

**Why this matters:** A query about "uptime commitment" should retrieve the specific SLA clause, not a 2000-char blob. Child chunks enable precise retrieval; parent text gives the LLM the full section for accurate answering.

### Embedding: nomic-embed-text (Ollama)

- Runs locally, no API keys needed
- Good performance on legal/technical text
- Consistent with the fully-offline design goal

### LLM: llama3.1 (Ollama)

- All agents use the same model through a single `OllamaChat` client
- Temperature 0, seed 42, top_p 1.0, repeat_penalty 1.05 for deterministic output
- All prompts enforce strict JSON output where structured responses are needed
- Graceful error handling — if Ollama is unavailable, the system returns an error message instead of crashing

### Retrieval: Hybrid Search + Cross-Encoder Re-ranking

Three-stage retrieval:

1. **Vector search** (ChromaDB, nomic-embed-text embeddings) — finds semantically similar chunks
2. **BM25 keyword search** (rank_bm25) — finds exact term matches (critical for "Section 4.2" or "72 hours")
3. **Score fusion** — normalized scores combined: 70% vector + 30% BM25 (configurable)
4. **Cross-encoder re-ranking** (ms-marco-MiniLM-L-6-v2) — re-scores top 20 candidates, returns top 10

**Why hybrid:** Pure vector search misses exact terms. Pure keyword search misses semantic similarity ("termination notice" ≈ "cancellation period"). The combination covers both.

**Why cross-encoder:** Bi-encoder embeddings are fast but approximate. The cross-encoder processes (query, chunk) pairs jointly for significantly better relevance scoring, at the cost of only running on pre-filtered candidates.

### HyDE (Hypothetical Document Embeddings)

The routing agent generates a `hyde_query` — a 1-2 sentence description of what an ideal contract clause answering the question would say. This hypothetical text is used as the vector search query instead of the raw question.

**Why:** Questions and answers live in different embedding spaces. "What is the notice period?" is far from "Either party may terminate this Agreement by providing 30 days written notice." HyDE bridges this gap.

### Self-Correction (CRAG)

After the answer agent generates a response, the grounding verifier checks every claim against the retrieved excerpts. If unsupported claims are found, it suggests a refined query and the pipeline retries retrieval + answering once.

**Why max 1 retry:** Diminishing returns — if the right information isn't found in two attempts, more retries won't help and will slow the system down.

### Intent-Gated Risk Extraction

Risk and consistency agents only activate for `risk`, `cross_document`, and `summary` intents. Simple factual questions ("What is the notice period?") skip risk analysis entirely.

**Why:** Without this gating, the risk agent would flag generic risks on every query, producing noisy output. This design choice improved `risk_correct` from 58% to 76% in evaluation.

### Session Management

- Up to 5 conversation turns (10 messages) retained in memory
- Response cache with 5-minute TTL for repeated queries
- Agents receive the last 4 messages for conversational context

### All Prompts in One File

Every system prompt lives in `utils/prompts.py`. This makes prompt tuning easy — one file to review, edit, and version-control. No hunting through agent files.

---

## 4. Setup Instructions

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### Install

```bash
cd legal_qa
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

Pull the required Ollama models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

### Build the Index

This is a one-time step that chunks the documents and stores embeddings in ChromaDB:

```bash
python build_index.py
```

Output shows chunk statistics per document and embedding progress.

To rebuild after changing documents: just run `build_index.py` again — it drops and recreates the collection.

### Configuration

All settings are in `utils/config.py` and can be overridden via environment variables:

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

---

## 5. How to Run

### Interactive Console

```bash
cd legal_qa
source .venv/bin/activate
python main.py
```

The console accepts natural language questions. Type `exit` or `quit` to end.

**Output per query:**

- **Answer** — Grounded response with inline citations (`[doc_id §section_path]`)
- **References** — List of cited document sections (shown only when present)
- **Risks** — Flagged legal/financial risks with clause references (shown only for relevant queries)
- **Inconsistencies** — Cross-document conflicts (shown only when detected)

**Example session:**

```
Legal Contract Q&A
Ask questions about the contracts. Type 'exit' or 'quit' to end.

Query: What is the notice period for terminating the NDA?

--- Answer ---
Either party may terminate the NDA by providing 30 days' written notice
to the other party [nda_acme_vendor §7].

References:
  [nda_acme_vendor §7]: Termination

Query: Are there any conflicting governing laws across agreements?

--- Answer ---
Yes. The NDA is governed by the laws of the State of California
[nda_acme_vendor §10], while the Vendor Services Agreement is governed
by the laws of the State of New York [vendor_services_agreement §12].

References:
  [nda_acme_vendor §10]: Governing Law
  [vendor_services_agreement §12]: Governing Law

Risks:
  ⚠ Conflicting governing laws across agreements (nda_acme_vendor §10 vs vendor_services_agreement §12)

Inconsistencies:
  [Governing Law] nda_acme_vendor vs vendor_services_agreement:
    A: State of California
    B: State of New York
```

---

## 6. Evaluation

### Approach

The evaluation framework combines two complementary methods:

**Custom deterministic metrics** (no LLM calls, fast, reproducible):

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| `key_phrase_recall` | Fraction of expected key phrases found in the answer | Does the answer contain the essential facts? |
| `citation_present` | Whether the answer contains at least one `[doc §section]` citation | Are claims grounded with sources? |
| `retrieval_hit_rate` | Fraction of expected documents that appeared in retrieval results | Is the retriever finding the right documents? |
| `risk_correct` | Whether risk presence/absence matches expectations | Does the system flag risks only when appropriate? |
| `scope_correct` | Whether out-of-scope queries are correctly refused | Does the system avoid answering unanswerable questions? |

**Ragas LLM-as-judge metrics** (uses llama3.1 as judge, slower):

| Metric | What It Measures |
|--------|-----------------|
| `faithfulness` | Is every claim in the answer supported by the retrieved context? |
| `answer_relevancy` | How relevant is the answer to the question? |
| `context_precision` | Are the most relevant chunks ranked highest? |

### Running the Evaluation

```bash
cd legal_qa
source .venv/bin/activate

# Custom metrics only (fast, ~5 min)
python -m eval.run_eval

# Custom + Ragas (slow, ~30 min with local LLM)
python -m eval.run_eval --ragas
```

**Outputs:**

- `eval/results/eval_results.json` — Raw pipeline outputs + per-query scores
- `eval/results/eval_summary.json` — Aggregate metric summary
- `eval/results/eval_report.csv` — One row per query with all scores

### Results

Evaluated on 17 sample queries covering fact extraction, risk identification, cross-document comparison, summarization, and out-of-scope handling:

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

*Faithfulness scores are unreliable with local LLM judges — Ragas faithfulness requires multiple chained LLM calls that frequently time out or produce inconsistent decompositions with llama3.1. This is a known limitation of running Ragas with local models.

### Evaluation Limitations

- **Small query set** — 17 queries cover the main scenarios but cannot test every edge case
- **No human annotation** — Ground truths are hand-crafted by the developer, not validated by legal experts
- **Ragas with local LLM** — Faithfulness and answer relevancy metrics are less reliable than with GPT-4 due to the judge model's own limitations
- **Single-turn only** — Evaluation runs each query independently; multi-turn conversation quality is not measured

---

## 7. Known Limitations

### Retrieval & Accuracy

- **Chunking is regex-based** — Section parsing relies on patterns like `N. Title` and `Section N`. Contracts with non-standard formatting (e.g., tables, nested appendices, inline headers) may chunk poorly, leading to missed or malformed sections.
- **No confidence signal** — The system returns answers without any confidence score. A user cannot tell whether the system is highly certain or guessing from loosely related chunks. The grounding verifier helps internally, but its verdict is not surfaced.
- **Grounding verifier is itself an LLM** — CRAG relies on llama3.1 to judge its own output. If the LLM hallucinates during verification (marking a fabricated claim as "supported"), the self-correction loop fails silently.
- **BM25 index rebuilt on every startup** — The keyword index is constructed in-memory from chunks each time `main.py` runs. With a large corpus this adds noticeable startup latency. Only the vector index (Chroma) is persisted.

### LLM & Prompting

- **JSON output is fragile** — The LLM is prompted to return JSON, but llama3.1 occasionally wraps it in markdown fences or adds trailing text. The parser uses `find("{")` / `rfind("}")` extraction which handles most cases, but deeply nested or malformed JSON can still fail.
- **Citation consistency varies** — Despite strict prompt instructions, the LLM sometimes omits citations on 1-2 sentences or uses slightly different formatting. The 80% citation rate reflects this — it's a prompt compliance issue inherent to the model size.
- **Single model for all roles** — Using llama3.1 for routing, answering, grounding, risk, and consistency means each agent is limited by the same model's capabilities. A production system could use a smaller/faster model for routing and a stronger model for answering.

### System Design

- **Sequential agent calls** — Each query makes up to 5 LLM calls in series (routing → answering → grounding → retry → risk/consistency). With llama3.1 on CPU, total latency per query can reach 30-60 seconds. Risk and consistency agents are independent and could run in parallel.
- **No incremental indexing** — Adding or updating a single document requires a full rebuild (`build_index.py` drops and recreates the entire Chroma collection). There is no diffing or selective re-embedding.
- **Text files only** — The loader reads `.txt` files. PDF, DOCX, and scanned documents are not supported. Real legal corpora are predominantly PDF.
- **Conversation context is naive** — The last 4 messages are passed as raw text to agents. There is no summarization of older turns, so context from earlier in a long conversation is lost entirely.
- **No query guardrails** — The system does not detect prompt injection, jailbreaking, or PII in user queries. It assumes all queries are well-intentioned and contract-related.
- **No access control** — All documents are visible to all users. In a real legal setting, different users/teams would need scoped access to specific contracts.

---

## 8. Production Roadmap

If this system were deployed for real legal teams, here is what I would prioritize:

### P0 — Must Have

| Enhancement | Why |
|------------|-----|
| **Document format support** (PDF with layout parsing, DOCX) | Real legal corpora are never plain text. Layout-aware parsing (e.g., unstructured.io, PyMuPDF) preserves tables, headers, and page structure that regex-based chunking currently misses. |
| **Incremental indexing** | Detect changed/new documents, re-embed only affected chunks. Legal teams update contracts frequently — full rebuilds are not acceptable. |
| **Confidence scoring** | Combine retrieval distance, grounding result, and citation density into a user-facing confidence indicator. Lets users know when to trust the answer vs. verify manually. |
| **Prompt injection detection** | Add a lightweight classifier before the routing agent to flag adversarial inputs. Legal systems handle sensitive data — injection attacks could extract or fabricate clause interpretations. |

### P1 — High Value

| Enhancement | Why |
|------------|-----|
| **Parallel agent execution** | Risk and consistency agents are independent of each other — run them concurrently to cut 30-40% of latency on risk/cross-doc queries. |
| **Streaming responses** | Stream answer tokens to the user while risk/consistency analysis runs in the background. Reduces perceived latency from 30s+ to first-token-in-3s. |
| **Observability & tracing** | Log every agent call with input/output, token count, latency, and retrieval scores. Essential for debugging production failures and identifying prompt regressions. |
| **Document versioning** | Track which version of a contract was used to generate each answer. Legal teams need audit trails — "this answer was based on the NDA signed 2024-01-15, not the amended version." |

### P2 — Long Term

| Enhancement | Why |
|------------|-----|
| **Domain-adapted embeddings** | Fine-tune the embedding model on legal text. General-purpose embeddings treat "material breach" and "significant violation" as loosely related; a legal-tuned model would rank them much closer. |
| **Human-annotated evaluation set** | Replace developer-crafted ground truths with answers validated by legal professionals. Current evaluation measures system behavior, not legal correctness. |
| **User feedback loop** | Thumbs up/down on answers → feed into retrieval tuning and prompt refinement. Closes the loop between deployment and improvement. |
| **Multi-tenant access control** | Per-user or per-team document visibility. Different departments should only see contracts relevant to them. |
| **Prompt version control** | Track prompt changes alongside eval metrics. When a prompt edit improves citation rate but degrades risk detection, you need the data to make the tradeoff. |
