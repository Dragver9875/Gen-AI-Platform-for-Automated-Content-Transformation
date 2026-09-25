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
     +--> Granite Docling HF -- document/OCR/layout
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
- optional Hugging Face TEI multilingual reranker
- configurable `hybrid`, `dense`, or `lexical` retrieval

# Phase 2 — Adaptive multimodal ingestion

Supported inputs:

- `.txt`, `.md`
- `.pdf`
- `.pptx` (legacy `.ppt` must be converted to `.pptx` before upload)
- `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`

PDF pages are preflight-classified as native-text, scanned/image-only, mixed, or visual-heavy. Image-only PDFs therefore follow OCR/visual-understanding routes instead of failing as text PDFs.

Standalone images are routed through SigLIP; document-like images go to the Hugging Face Granite Docling adapter and general visuals go to the VLM endpoint. PDF pages are rendered locally and sent as images to `ibm-granite/granite-docling-258M` through a Hugging Face-compatible multimodal endpoint. PPTX text/tables are extracted deterministically with `python-pptx`; legacy `.ppt` should be converted to `.pptx` before upload.

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

The ML layer now uses one shared Hugging Face credential. `HF_TOKEN` is only forwarded to Hugging Face-owned URLs (`*.huggingface.co` and `*.huggingface.cloud`).

```env
HF_TOKEN=hf_your_token_here
```

The same token can authenticate Harrier, Granite Docling, Qwen generation/verification, Hugging Face SigLIP/VLM endpoints, the optional HF reranker endpoint, and FLUX/image-generation providers.

The only independent service credentials required by the prototype are:

```env
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...
SESSION_DATABASE_URL=postgresql://user:password@host:5432/database
```

`CHROMA_TENANT` and `CHROMA_DATABASE` are identifiers rather than secrets, but are required by Chroma Cloud. Provider-specific model-key environment variables remain accepted only for backwards compatibility and custom non-HF deployments; they are not required for the HF-only profile.

> Hugging Face provider/model availability and included credits are account dependent. A model that is unavailable through serverless Inference Providers can be deployed as a Hugging Face Inference Endpoint and still uses the same `HF_TOKEN`.

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

`HARRIER_API_URL` is an optional endpoint override. The adapter also supports OpenAI-compatible embedding endpoints when `HARRIER_API_STYLE=openai`.

## Hugging Face reranker

The default hybrid retriever remains Harrier + BM25 + RRF. An optional multilingual second-stage reranker can now be served as a Hugging Face Text Embeddings Inference endpoint using `BAAI/bge-reranker-v2-m3`, an Apache-2.0 multilingual reranker.

```env
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_API_URL=https://your-reranker.endpoints.huggingface.cloud
RERANKER_API_STYLE=hf_tei
RETRIEVAL_USE_RERANKER=true
```

No reranker key is needed; the endpoint uses `HF_TOKEN`. If `RERANKER_API_URL` is omitted, the service falls back to RRF without failing.

## Granite Docling on Hugging Face

Document understanding no longer requires a separate Docling Serve credential. The default model is `ibm-granite/granite-docling-258M` (Apache-2.0), consumed through a Hugging Face-compatible multimodal/chat endpoint with `HF_TOKEN`. The model is designed for document-page conversion and its official instruction is `Convert this page to docling.`

The HF-only ingestion adapter works as follows:

```text
PDF  -> PyMuPDF renders pages -> Granite Docling HF endpoint
Image -> Granite Docling HF endpoint
PPTX -> deterministic python-pptx text/table extraction
PPT   -> convert to PPTX before upload
```

Default configuration:

```env
DOCLING_MODEL=ibm-granite/granite-docling-258M
DOCLING_API_URL=
DOCLING_RENDER_DPI=144
```

If `DOCLING_API_URL` is blank, the application uses the Hugging Face OpenAI-compatible router. If that model is not available from a serverless provider in your account/region, deploy the same model as a Hugging Face Inference Endpoint and set `DOCLING_API_URL` to that endpoint; authentication still uses the same `HF_TOKEN`. Hugging Face supports image-text-to-text through its unified inference API and chat-compatible multimodal requests.

## SigLIP

SigLIP no longer requires a dedicated endpoint URL. By default the service uses
`huggingface_hub.InferenceClient.zero_shot_image_classification()` with:

```text
model: google/siglip-so400m-patch14-384
auth: HF_TOKEN
```

Optional override:

```env
SIGLIP_MODEL=google/siglip-so400m-patch14-384
SIGLIP_API_URL=
```

If `SIGLIP_API_URL` is supplied, the original custom endpoint contract is still
supported. If serverless SigLIP inference is unavailable, Phase 2 automatically
falls back to VLM-based visual routing instead of aborting ingestion.

## VLM

The default VLM uses Hugging Face's OpenAI-compatible multimodal router, so no
endpoint URL needs to be configured manually:

```env
VLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
VLM_API_URL=
VLM_API_STYLE=openai
```

An explicit `VLM_API_URL` remains available as an override for a dedicated HF
Inference Endpoint or other compatible deployment. Hugging Face authentication
still comes from `HF_TOKEN` for HF-owned URLs.

## Hosted open-source LLM

Default model: `openai/gpt-oss-20b:fastest` (Apache-2.0). It is consumed only through a hosted API endpoint; no model weights are bundled with or loaded by this repository. `LLM_MODEL` remains configurable.

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

Configure `HF_TOKEN`, Chroma Cloud, and (for persistent sessions) the session database. SigLIP, the VLM, Harrier, Granite Docling, Qwen, and verification all have Hugging Face defaults; dedicated endpoint URLs are optional overrides. See `.env.example`.

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

## Windows PowerShell launcher notes

If Windows marks the downloaded scripts as coming from the Internet, unblock them once after you trust the repository:

```powershell
Unblock-File .\setup.ps1
Unblock-File .\run_ps.ps1
```

`run_ps.ps1` transports queries to Python as UTF-8 Base64, so spaces, embedded quotes, Unicode text, and newlines are preserved correctly even in Windows PowerShell 5.1.

Prompt-only transformations are supported. When no file and no existing `-SessionId` are supplied, the entered query is temporarily materialized as a text source and passed through the normal ingestion pipeline. For clearer separation between source content and instruction, use `-SourceText` explicitly:

```powershell
.\run_ps.ps1 `
  -SourceText 'I finished a project today' `
  -Query 'Rewrite this as a formal LinkedIn post' `
  -Mode transform `
  -Format text
```


### Hugging Face chat router 400 errors

The original prototype briefly defaulted to `Qwen/Qwen3-30B-A3B-Instruct-2507`. That checkpoint is not currently router-served by Hugging Face Inference Providers, so old `.env` files can produce an HTTP 400 during Phase 4. The current default is `openai/gpt-oss-20b:fastest`, which Hugging Face documents as supported by Inference Providers.

Recommended configuration:

```env
LLM_API_URL=
LLM_API_KEY=
LLM_API_STYLE=openai
LLM_MODEL=openai/gpt-oss-20b:fastest
LLM_RESPONSE_MODE=json_schema
```

When the endpoint does not support the requested structured-output mode, the adapter automatically degrades from JSON Schema to JSON object and finally to prompt-constrained JSON, with Pydantic validation still enforced afterwards.


### Phase 6 text artifact behavior

Plain-text artifacts are generated through the hosted LLM's normal text mode. They are **not** forced through JSON parsing. Structured output remains in use for PDF/Typst, PPTX, SVG, and creative-image planning where an intermediate schema is required.
