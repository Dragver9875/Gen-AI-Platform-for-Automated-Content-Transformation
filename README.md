# Gen-AI Platform for Automated Content Transformation

API-native multimodal content-transformation platform that ingests text, PDFs, PPT/PPTX files, and images; retrieves grounded context with multilingual hybrid search; generates a canonical response; verifies factual claims; and produces text, PDF, PPTX, SVG/infographic, or creative-image artifacts.

## Current milestone

Implemented:

- **Phase 1** — Retrieval and indexing
- **Phase 2** — Adaptive multimodal ingestion
- **Phase 3** — LangGraph orchestration and user sessions
- **Phase 4** — Hosted open-source LLM generation and Canonical Response Representation (CRR)
- **Phase 5** — Factuality verification and bounded repair
- **Phase 6** — Generic artifact generation
- **Phase 9** — Evaluation, quality metrics, and runtime telemetry

Still pending before public deployment:

- **Phase 7** — service/API + operator dashboard
- **Phase 8** — deployment hardening, authentication, persistent production infrastructure, queues, rate limiting, and operational observability

Phase 9 has intentionally been implemented before the deployment layer so retrieval, grounding, artifact quality, latency, and cost baselines exist before production infrastructure is introduced.

## Design principles

- **Zero model weights in the application.** ML inference is performed through hosted APIs/endpoints.
- **Provider-agnostic control plane.** Providers are resolved by capability rather than being embedded in LangGraph nodes.
- **Source truth is preserved.** Raw extracted content and provenance are retained alongside normalized content.
- **Multi-tenant isolation.** `user_id + session_id` scope LangGraph state and Chroma retrieval.
- **One stable LangGraph topology.** The prototype keeps the current QA-vs-transform flow; artifact formats are resolved through registries rather than separate graph branches.
- **Content IR → Artifact IR.** Semantic content is separated from PDF/PPTX/SVG serialization.
- **Verification precedes rendering.** Unresolved factual issues do not proceed to final artifact generation.

## End-to-end architecture

```text
USER / CLIENT
     |
     v
TEXT / PDF / PPT / PPTX / IMAGE
     |
     v
Adaptive ingestion router
     |
     +--> Docling API -------- document/OCR/layout
     +--> SigLIP API -------- visual routing
     +--> VLM API ----------- visual understanding
     |
     v
Unified Source Representation
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
     +--> BM25 lexical retrieval
     +--> Harrier dense retrieval
                |
                v
               RRF
                |
         optional reranker
                |
                v
          LangGraph router
          /             \
        QA            Transform
        |                |
      Top-K       hierarchical corpus
        \                /
         \              /
          Context Builder
                |
                v
          Hosted open-source LLM API
                |
                v
               CRR
                |
                v
        Claim verification
                |
        repair when needed
                |
                v
            Content IR
                |
                v
         Artifact Registry
      /      |      |      \
    text    PDF    PPTX    image
                |
                v
             output
```

---

# Phase 1 — Retrieval and indexing

Implemented:

- hosted `microsoft/harrier-oss-v1-0.6b` embedding adapter
- Hugging Face serverless Harrier defaults using a single `HF_TOKEN`
- batched document embeddings
- Harrier `web_search_query` prompt on query embeddings only
- L2-normalized feature-extraction embeddings
- Chroma Cloud only
- non-destructive `upsert`
- user/session/source metadata isolation
- BM25 sparse retrieval
- Harrier/Chroma dense retrieval
- Reciprocal Rank Fusion (RRF)
- optional hosted reranker
- configurable `hybrid`, `dense`, or `lexical` retrieval

# Phase 2 — Adaptive multimodal ingestion

Supported inputs:

- `.txt`, `.md`
- `.pdf`
- `.ppt`, `.pptx`
- `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`

PDF pages are preflight-classified as native-text, scanned/image-only, mixed, or visual-heavy. Image-only PDFs therefore follow OCR/visual-understanding routes instead of failing as text PDFs.

Standalone images are routed through SigLIP; document-like images go to Docling and general visuals go to the VLM endpoint.

# Phase 3 — LangGraph orchestration and user sessions

The current topology is intentionally retained:

```text
START
  |
initialize
  |
ingest if needed
  |
classify intent
  |
  +--> QA ---------> Top-K retrieval -------+
  |                                         |
  +--> Transform --> whole-document context +--> Context Builder
  |                                         |
  +--> Index only --------------------------+
```

Session characteristics:

- isolation boundary: `user_id + session_id`
- hashed LangGraph `thread_id`
- session reuse without re-ingestion
- memory store for development
- PostgreSQL session store/checkpointer hooks for deployment

# Phase 4 — Hosted open-source LLM generation + CRR

The hosted open-source LLM generates a format-independent Canonical Response Representation containing:

- artifact type
- title / summary
- sections and bullets
- key points
- factual claims
- chunk-level evidence references
- tone, audience, language and rendering hints
- explicit insufficiencies

Long-document transformations use hierarchical section digests before final CRR synthesis.

# Phase 5 — Factuality verification and repair

Each factual claim is checked only against its cited evidence chunks.

Statuses:

- `supported`
- `partially_supported`
- `unsupported`
- `insufficient_evidence`

Deterministic checks additionally validate critical numbers, percentages, dates and identifiers. Repair is bounded by `PHASE5_MAX_REPAIR_ATTEMPTS`.

The application computes faithfulness itself; it does not trust a model-supplied aggregate score.

# Phase 6 — Generic artifact generation

Phase 6 keeps a single LangGraph artifact node and resolves formats through an `ArtifactRegistry`.

Supported artifact families:

- text
- PDF via model-generated Typst + thin compiler
- editable PPTX via model-generated Presentation IR + deterministic serializer
- factual SVG/infographic
- creative image via hosted image-generation API

The repository also includes:

- capability-based `ProviderRegistry`
- registrable ingestion processors
- configurable retrieval strategies
- named verification profiles
- service-level `TaskSpec`
- `CRR -> ContentIR -> ArtifactIR`

---

# Phase 9 — Evaluation and metrics

Phase 9 provides offline benchmark metrics and runtime telemetry without changing the LangGraph topology.

## Retrieval metrics

For benchmark cases with ground-truth `relevant_chunk_ids`:

- `Precision@K`
- `Recall@K`
- `HitRate@K`
- `MRR`
- `nDCG@K`

## Reference-text metrics

For cases with a human/reference output:

- normalized exact match
- token precision
- token recall
- token F1
- ROUGE-L precision / recall / F1
- generated/reference length ratio

The implementation is dependency-light and does not download evaluation-model weights.

## Grounding metrics

Derived directly from the Phase 5 verification report:

- faithfulness score
- claim support rate
- partial-support rate
- unsupported-claim rate
- insufficient-evidence rate
- mean verifier confidence
- repair attempts

## Artifact metrics

- artifact generation success rate
- artifact failure rate
- source-only rate
- non-empty output-file rate
- requested-format coverage

## Runtime/provider metrics

`core.telemetry` instruments hosted provider calls and Chroma operations during eval runs.

Reported metrics include:

- end-to-end latency
- provider request count
- provider error count
- retry count
- API p50 latency
- API p95 latency
- input/output token usage when returned by the provider
- optional estimated cost
- per-provider request, error, latency and token totals

Pricing is external configuration and is never hardcoded because provider rates change.

## Evaluation dataset format

JSON:

```json
{
  "name": "threat-advisory-eval",
  "version": "1.0",
  "cases": [
    {
      "case_id": "qa-001",
      "user_id": "eval-user",
      "session_id": "eval-session-001",
      "source_paths": ["./data/report.pdf"],
      "query": "What mitigations are recommended?",
      "request_mode": "qa",
      "top_k": 5,
      "relevant_chunk_ids": ["chunk-10", "chunk-11"],
      "reference_text": "...",
      "expected_formats": ["text"],
      "transformation_config": {
        "artifact_type": "answer",
        "output_formats": ["text"],
        "verification_profile": "strict"
      }
    }
  ]
}
```

`relevant_chunk_ids`, `reference_text`, and `expected_formats` are optional. Metric families are calculated only when the corresponding ground truth is provided.

JSONL is also supported: one `EvalCase` object per line.

## Run Phase 9

```bash
python -m scripts.run_evals \
  --dataset scripts/eval_data/sample.json \
  --output eval_reports/latest.json
```

The command writes:

```text
eval_reports/latest.json
eval_reports/latest.md
```

### Optional pricing file

```json
{
  "llm": {
    "input_per_million": 0.0,
    "output_per_million": 0.0,
    "per_request": 0.0
  },
  "image_generation": {
    "per_request": 0.0
  }
}
```

Run with:

```bash
python -m scripts.run_evals \
  --dataset eval.json \
  --pricing-json scripts/eval_data/pricing.example.json \
  --output eval_reports/latest.json
```

The aggregate report contains the mean of every numeric metric available across benchmark cases plus an overall pipeline pass rate.

See [`PHASE9.md`](PHASE9.md) for the focused Phase 9 description.

---


## Hugging Face token setup

A single Hugging Face token can now be reused by Hugging Face-hosted providers. The application only forwards `HF_TOKEN` to URLs owned by Hugging Face (`*.huggingface.co` or `*.huggingface.cloud`), so an accidentally configured third-party URL does not receive the shared token.

```env
HF_TOKEN=hf_your_token_here
```

With only this token set, the defaults are:

```text
Harrier -> HF serverless feature-extraction endpoint
Qwen    -> https://router.huggingface.co/v1/chat/completions
```

Provider-specific keys remain supported and take precedence. For example, `HARRIER_API_KEY` overrides `HF_TOKEN` for Harrier. SigLIP, VLM, and creative-image providers also reuse `HF_TOKEN` automatically **only when their configured URL is a Hugging Face-owned endpoint**. Docling and Chroma continue to use their own credentials.

> Hugging Face serverless/provider availability and free credits are account/model dependent. For production, the same code can point at a dedicated HF Inference Endpoint by setting the provider URL explicitly.

# Provider/API contracts

## Harrier

The default Harrier configuration uses Hugging Face serverless feature extraction:

```text
model: microsoft/harrier-oss-v1-0.6b
endpoint: https://router.huggingface.co/hf-inference/models/microsoft/harrier-oss-v1-0.6b
auth: HF_TOKEN
query prompt: web_search_query
normalize: true
```

You therefore do **not** need a separate Microsoft or Harrier API key. Set one Hugging Face token:

```env
HF_TOKEN=hf_...
```

`HARRIER_API_URL` and `HARRIER_API_KEY` are optional overrides for a dedicated endpoint. The adapter also supports OpenAI-compatible embedding endpoints when `HARRIER_API_STYLE=openai`.

## Docling

`DOCLING_API_URL` must expose a Docling Serve-compatible `/v1/convert/file` endpoint.

## SigLIP

Expected custom endpoint payload:

```json
{
  "image_base64": "...",
  "candidate_labels": ["document page", "screenshot", "chart", "diagram", "map", "photograph", "other visual"]
}
```

## VLM

```json
{
  "image_base64": "...",
  "prompt": "..."
}
```

## Hosted open-source LLM

Default model: `Qwen/Qwen3-30B-A3B-Instruct-2507` (Apache-2.0). It is consumed only through a hosted API endpoint; no Qwen weights are bundled with or loaded by this repository. `LLM_MODEL` remains configurable.

Supported modes:

- OpenAI-compatible chat-completions API
- Hugging Face/dedicated text-generation API

## Creative image API

Hosted FLUX-style/HF or OpenAI-style image endpoint through `HostedImageGenerationProvider`.

---

# Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

Configure `HF_TOKEN`, Chroma, Docling, and any non-default vision endpoints in the deployment environment. See `.env.example`.

The application never requires local Harrier, Qwen, SigLIP, VLM or image-generation weights.

# Main CLIs

```bash
# Phase 1/2 index + retrieve smoke test
python -m scripts.index_file ./sample.pdf --session-id demo --query "What are the recommendations?"

# Phase 3 orchestration
python -m scripts.run_phase3 --user-id user-1 --file ./sample.pdf --query "Summarize it" --mode transform

# Phase 4 generation
python -m scripts.run_phase4 --user-id user-1 --file ./sample.pdf --query "Create an executive briefing" --mode transform

# Phase 5 verified CRR
python -m scripts.run_phase5 --user-id user-1 --file ./sample.pdf --query "Create an advisory" --mode transform

# Phase 6 artifacts
python -m scripts.run_phase6 --user-id user-1 --file ./sample.pdf --query "Create an advisory" --mode transform --format text --format pdf

# Phase 9 evaluation
python -m scripts.run_evals --dataset scripts/eval_data/sample.json --output eval_reports/latest.json
```

# Testing

Unit tests use mock providers and require no provider API keys:

```bash
pytest -q
```

Current regression suite: **40 tests**.

# Security

- Never expose provider credentials in the frontend.
- Never commit `.env`, Streamlit secrets, service-account files or generated artifacts.
- Model-generated Typst and SVG are validated before serialization.
- User/session IDs are sanitized before artifact paths are created.
- Chroma reads/writes are scoped by user/session metadata.
- Raw user identifiers are not used directly as LangGraph checkpoint thread IDs.

# Remaining work before public deployment

## Phase 7

- FastAPI service endpoints
- job/task lifecycle API
- dynamic capability endpoint
- artifact download API
- operator dashboard
- authentication boundary

## Phase 8

- production PostgreSQL checkpointing/session infrastructure
- object storage for uploads/artifacts
- async/background jobs
- rate limiting and quotas
- deployment secret management
- provider fallback policies
- health/readiness endpoints
- operational logging/monitoring using the telemetry events introduced in Phase 9

The Phase 9 baseline should be re-run after Phase 7/8 so deployment overhead and provider behavior can be compared against the pre-deployment baseline.
