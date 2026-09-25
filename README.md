# Gen-AI Platform for Automated Content Transformation

## Current milestone: Phase 1 + Phase 2 complete

This repository currently implements the **retrieval/indexing layer (Phase 1)** and the **adaptive multimodal ingestion layer (Phase 2)** of the planned API-native content transformation system.

No ML model weights are loaded by this application. Harrier, SigLIP, VLM and Docling inference are accessed through hosted API endpoints. The only local computation is deterministic orchestration such as file inspection, chunking, BM25, RRF and metadata handling.

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

## Tests

Unit tests use mock providers and require no API keys:

```bash
pytest -q
```

## Not yet implemented

The following intentionally belong to later phases:

- LangGraph orchestration/state graph.
- Query-vs-whole-document transformation router.
- General-purpose SLM generation.
- Canonical Response Representation (CRR).
- Factuality verification/repair loop.
- Text/PDF/PPTX/image output decoders and serializers.
- Production frontend/authentication.

## Security

Never expose provider keys in the frontend or commit them to Git. Keep them in deployment secrets/environment variables.
