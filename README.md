# Gen-AI Platform for Automated Content Transformation

## Current milestone: Phase 1 + Phase 2 + Phase 3 complete

This repository currently implements the **retrieval/indexing layer (Phase 1)**, **adaptive multimodal ingestion layer (Phase 2)**, and **LangGraph orchestration/session layer (Phase 3)** of the planned API-native content transformation system.

No ML model weights are loaded by this application. Harrier, SigLIP, VLM and Docling inference are accessed through hosted API endpoints. Local computation is deterministic control-plane logic such as file inspection, chunking, BM25, RRF, LangGraph routing, context assembly and metadata handling.

## Implemented architecture

```text
TEXT -------------------------------+
                                     |
PDF/PPT/PPTX -> adaptive router -> Docling API ----+
PDF visual/scanned pages -> VLM API ---------------+--> Unified Source Schema
                                                     |
IMAGE -> SigLIP API -> document-like -> Docling API +
                   \-> general visual -> VLM API ---+
                                                     |
                                                     v
                                         Structure-aware chunking
                                                     |
                                                     v
                                      Microsoft Harrier API embeddings
                                                     |
                                                     v
                                                Chroma Cloud
                                                     |
                                    +----------------+----------------+
                                    |                                 |
                                  BM25                         Harrier dense search
                                    |                                 |
                                    +--------------- RRF -------------+
                                                     |
                                               optional hosted reranker
                                                     |
                                                     v
                                                  Top-K
```

## Phase 1 — Retrieval and indexing

Implemented:

- Microsoft Harrier OSS v1 0.6B **hosted embedding endpoint adapter**.
- Batched document embedding.
- Query-side Harrier retrieval instruction.
- Chroma Cloud only; no local Chroma fallback.
- Non-destructive `upsert` into a shared collection.
- Session/source metadata isolation.
- BM25 sparse retrieval.
- Harrier/Chroma dense retrieval.
- Reciprocal Rank Fusion (RRF), replacing the previous English-only local cross-encoder as the default.
- Optional hosted reranker adapter.

## Phase 2 — Adaptive multimodal ingestion

Implemented inputs:

- `.txt`, `.md`
- `.pdf`
- `.ppt`, `.pptx`
- `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`

### PDF routing

Before model routing, PDFs are inspected page-by-page using lightweight deterministic preflight logic:

- **native-text** page
- **scanned/image-only** page
- **mixed** page
- **visual-heavy** page

The PDF is then sent to the Docling API with OCR settings selected from the preflight result. Visual/scanned pages can additionally be sent to the hosted VLM as a retrieval-oriented fallback, solving the all-image-PDF edge case.

### Image routing

Standalone images are classified by a hosted SigLIP endpoint:

- document page / screenshot -> Docling API
- photograph / diagram / map / chart / other visual -> VLM API

All branches converge into the same `IngestionResult` / `SourceElement` schema before chunking.

## Phase 3 — LangGraph orchestration and session flow

Implemented:

- Typed, checkpoint-compatible `AgentState`.
- Thread/session continuation using LangGraph `thread_id`.
- In-memory development checkpointer by default; production checkpointers can be injected at graph construction.
- Upload/index-only workflow.
- Multi-turn reuse of previously indexed sources without re-ingestion.
- Deterministic Phase 3 intent router with explicit `qa` / `transform` overrides.
- QA branch using Phase 1 hybrid Top-K retrieval.
- Whole-document transformation branch using all indexed chunks with structural ordering.
- Hierarchical grouping by source and section.
- Provenance-rich bounded context construction.
- Error states for missing sources / failed ingestion.
- Explicit `phase4_ready` handoff without prematurely implementing SLM generation.

### Phase 3 graph

```text
START
  |
  v
initialize
  |
  +-- pending files --> ingest_sources --> classify_intent
  |                                      |
  +------------------> classify_intent ---+
                                         |
                         +---------------+----------------+
                         |               |                |
                        QA          TRANSFORM        INDEX ONLY
                         |               |                |
                    hybrid Top-K     full corpus      finalize
                         |          hierarchical          |
                         +-------+-------+                |
                                 |                        |
                                 v                        |
                           build_context                  |
                                 |                        |
                                 +-----------+------------+
                                             |
                                             v
                                            END
```

`build_context` produces both:

1. `context_groups`: the complete structured retrieval result for Phase 4 hierarchical generation, and
2. `prepared_context`: a bounded source/page/section-aware text representation suitable for a direct downstream model call when the corpus fits.

The transformation branch intentionally does **not** summarize the document in Phase 3. SLM generation and CRR construction belong to Phase 4.

## API contracts

### Docling

`DOCLING_API_URL` must expose a docling-serve-compatible API. The implementation calls:

```text
POST {DOCLING_API_URL}/v1/convert/file
```

and requests Markdown + JSON output, OCR and enrichment options as needed.

### Harrier

Deploy `microsoft/harrier-oss-v1-0.6b` behind a hosted endpoint. Two payload styles are supported:

- `HARRIER_API_STYLE=hf`: `{"inputs": ["..."]}`
- `HARRIER_API_STYLE=openai`: `{"input": ["..."]}`

The response may be a raw list of vectors, `{"embeddings": [...]}`, or an OpenAI-compatible `{"data": [{"embedding": [...]}]}` structure.

### SigLIP

Use a custom hosted endpoint accepting:

```json
{
  "image_base64": "...",
  "candidate_labels": ["document page", "screenshot", "chart", "diagram", "map", "photograph", "other visual"]
}
```

and returning label/score pairs.

### VLM

The generic VLM endpoint contract is:

```json
{
  "image_base64": "...",
  "prompt": "..."
}
```

with a response containing `generated_text`, `text`, `description`, or `output`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

Export/load the `.env` values in your deployment environment.

Phase 3 optional settings:

```text
PHASE3_DEFAULT_TOP_K=5
PHASE3_CONTEXT_MAX_CHARS=60000
```

Required services:

1. Chroma Cloud database.
2. Hosted Harrier 0.6B endpoint.
3. Docling Serve / Docling for IBM watsonx-compatible endpoint.
4. Hosted SigLIP routing endpoint.
5. Hosted VLM endpoint.

## High-level Python API

```python
from app.config import Settings
from app.factory import build_phase12

pipeline = build_phase12(Settings.from_env())
result, chunks = pipeline.ingest_and_index("report.pdf", session_id="session-1")
hits = pipeline.retrieve("What are the main recommendations?", session_id="session-1", source_id=result.source_id)
```

Phase 3 LangGraph API:

```python
from app.config import Settings
from app.factory import build_phase3

agent = build_phase3(Settings.from_env())

# First turn: ingest/index only.
agent.invoke(
    session_id="session-1",
    source_paths=["report.pdf"],
)

# Later turn on the same LangGraph thread: no re-ingestion.
state = agent.invoke(
    session_id="session-1",
    query="What mitigations are recommended?",
)
print(state["prepared_context"])

# Whole-document route.
state = agent.invoke(
    session_id="session-1",
    query="Summarize the complete report into an executive briefing",
    request_mode="transform",
)
print(state["context_groups"])
```

## CLI smoke test

```bash
python -m scripts.index_file ./sample.pdf --session-id demo-session --query "What are the main recommendations?"
```

This performs:

1. adaptive ingestion,
2. normalized structured extraction,
3. chunking,
4. Harrier API embedding,
5. Chroma Cloud indexing,
6. BM25 + dense retrieval,
7. RRF,
8. Top-K output.

## Phase 3 CLI

```bash
python -m scripts.run_phase3 \
  --session-id demo-session \
  --file ./sample.pdf \
  --query "Summarize the complete report" \
  --mode transform
```

The CLI prints routing, ingestion, retrieval and Phase-4-handoff statistics; it does not generate final content yet.

## Tests

Unit tests use mock providers and require no API keys:

```bash
pytest -q
```

## Not yet implemented

The following intentionally belong to later phases:

- **Phase 4:** general-purpose SLM generation and Canonical Response Representation (CRR).
- **Phase 5:** factuality verification and repair loop.
- **Phase 6:** text/PDF/PPTX/image output decoders and serializers.
- **Phase 7:** production frontend/API/authentication.
- **Phase 8:** deployment hardening, evaluation and observability.

## Security

Never expose provider keys in the frontend or commit them to Git. Keep them in deployment secrets/environment variables.


## User sessions and multi-tenant isolation

Phase 3 now treats `user_id + session_id` as the isolation boundary. The same session ID may safely exist for different users. LangGraph thread IDs are generated as a stable SHA-256-derived key so raw user identifiers are not stored in the checkpoint key. Chroma metadata and all retrieval filters include both `user_id` and `session_id`.

Session operations are available through `Phase3Orchestrator`:

```python
session = agent.create_session("user-123", title="Incident report review")
state = agent.invoke(
    user_id="user-123",
    session_id=session["session_id"],
    source_paths=["report.pdf"],
)
all_sessions = agent.list_sessions("user-123")
agent.rename_session("user-123", session["session_id"], "Updated title")
agent.delete_session("user-123", session["session_id"])
```

The session registry is pluggable:

- `SESSION_STORE_BACKEND=memory` — tests/single-process development only.
- `SESSION_STORE_BACKEND=postgres` — recommended for deployed multi-instance systems.

For production LangGraph state persistence, inject a PostgreSQL checkpointer rather than relying on the development `InMemorySaver`. `app/checkpointing.py` provides a helper using `langgraph-checkpoint-postgres`. Run its `.setup()` migration once during deployment.

### Generalization / configuration

There are no embedded provider URLs, API keys, local model paths, CUDA assumptions, user IDs, or session IDs. Provider endpoints and credentials are environment-driven. Retrieval limits, chunk sizes, PDF routing thresholds, SigLIP document labels, the Harrier query instruction, the PDF visual prompt, Chroma collection, HTTP retry/timeout behavior, and session persistence backend are configurable.

Some defaults remain intentionally opinionated but are not deployment assumptions: supported file extensions, default routing thresholds, default `top_k`, context-size limit, and fallback prompt text. They can be changed without modifying the pipeline architecture; the most deployment-sensitive ones are exposed in `.env.example`.

### Session CLI

```bash
python -m scripts.manage_sessions --user-id user-123 create --title "Threat report"
python -m scripts.manage_sessions --user-id user-123 list
python -m scripts.manage_sessions --user-id user-123 rename SESSION_ID "New title"
python -m scripts.manage_sessions --user-id user-123 delete SESSION_ID
```
