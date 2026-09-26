# OmniTransform — Gen-AI Platform for Automated Content Transformation

API-native multimodal content transformation pipeline built around one shared visual/document encoder and one shared semantic decoder.

## Current model stack

| Role | Model / component |
|---|---|
| Multimodal understanding for images, PDF pages and PPTX slides | `Qwen/Qwen2.5-VL-3B-Instruct` via Hugging Face |
| Retrieval embeddings | `microsoft/harrier-oss-v1-0.6b` via Hugging Face |
| Semantic generation, verification, repair and artifact planning | `openai/gpt-oss-20b:fastest` via Hugging Face |
| Creative image decoding | `black-forest-labs/FLUX.1-schnell` via Hugging Face Inference Providers |
| Vector store | Chroma Cloud |
| Session persistence | Memory locally; PostgreSQL optionally |

No local model weights, CUDA runtime, SigLIP, Granite Docling, MinerU, OCR-specific model, or separate verifier model is required.

## Simplified architecture

```text
TXT / MD ------------------------------> direct text

IMAGE ---------------------> Qwen2.5-VL -----+
PDF -> render each page ----> Qwen2.5-VL -----+--> Unified Source IR
PPTX -> render each slide --> Qwen2.5-VL -----+
                                               |
                                               v
                                      structure-aware chunks
                                               |
                                               v
                                           Harrier
                                               |
                                               v
                                        Chroma Cloud
                                               |
                                               v
                                          LangGraph
                                               |
                                               v
                                        gpt-oss-20b
                                  generation + verification
                                      + repair + planning
                                               |
                    +--------------------------+-------------------------+
                    |                          |                         |
                  text                    PDF / PPTX                 SVG
                    |                          |                         |
              deterministic            Typst / python-pptx       serializer
              serialization

Creative image:
gpt-oss-20b prompt plan -> FLUX.1-schnell -> PNG
```

## Multimodal ingestion behavior

The runtime deliberately does not classify inputs into “document image” vs “normal image.” Qwen2.5-VL receives every supported image directly with a retrieval-oriented prompt. If the image is a photo/scan of a document page, the prompt tells Qwen to transcribe it rather than merely describe it.

PDF files are rasterized page-by-page with PyMuPDF, then every rendered page is encoded by Qwen2.5-VL. PPTX files are rendered locally to slide canvases using `python-pptx` + Pillow: text, tables, chart data and embedded pictures are placed onto an approximate slide image, then every slide is encoded by the same Qwen2.5-VL provider. This avoids requiring Microsoft PowerPoint or LibreOffice locally.

`MULTIMODAL_MAX_PDF_PAGES` and `MULTIMODAL_MAX_PPTX_SLIDES` guard against accidental huge interactive jobs. Set either to `0` for unlimited processing.

## Credentials

Minimal local setup:

```env
HF_TOKEN=hf_...
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...
SESSION_STORE_BACKEND=memory
```

For persistent deployment add:

```env
SESSION_STORE_BACKEND=postgres
SESSION_DATABASE_URL=postgresql://...
```

All model calls can reuse `HF_TOKEN`. See `.env.example` for tuning options.

## Local GUI

```powershell
Unblock-File .\deploy_locally.ps1
.\deploy_locally.ps1 -RunSmokeTests
```

The launcher creates `.venv_local`, installs GUI/runtime dependencies, validates `.env`, compiles the project, optionally runs focused tests, starts Streamlit, polls its health endpoint and only then opens the browser.

Default URL:

```text
http://127.0.0.1:8501
```

Use another port with:

```powershell
.\deploy_locally.ps1 -Port 8502
```

The GUI invokes the real `build_phase6()` pipeline. It shows ingestion strategy, provider telemetry, wall-clock time, retrieval mode, verification metrics, warnings/errors, and generated artifacts.

## PowerShell CLI

```powershell
.\run_ps.ps1 `
  -UserId user-1 `
  -File '.\Test data\test_pdf.pdf' `
  -Query 'Explain its contents' `
  -Mode auto `
  -Format text
```

For prompt-only transformations:

```powershell
.\run_ps.ps1 `
  -SourceText 'I finished a project today.' `
  -Query 'Rewrite this as a professional LinkedIn post.' `
  -Mode transform `
  -ArtifactType linkedin_post `
  -Format text
```

## Artifact decoders

- **Text**: `gpt-oss-20b` generates final prose directly; deterministic UTF-8 serializer writes it.
- **PDF**: `gpt-oss-20b` plans Typst source; Typst compiles the final PDF. If Typst is absent, the `.typ` source is retained with `source_only` status.
- **PPTX**: `gpt-oss-20b` produces `PresentationIR`; `python-pptx` serializes it into an editable deck.
- **Factual infographic**: `gpt-oss-20b` produces safe SVG; deterministic serializer writes it.
- **Creative image**: `gpt-oss-20b` plans a visual prompt; FLUX.1-schnell generates the image via Hugging Face Inference Providers.

## Verification

Phase 5 uses the same `gpt-oss-20b` provider as Phase 4. It performs semantic evidence verification plus deterministic checks for literals such as numbers/dates and bounded repair. There is no separate verifier endpoint/model to configure. All model-facing citations use short `E1`/`E2` aliases; real chunk IDs remain internal. Repair-provider failures are non-fatal and finish as `complete_with_issues` while retaining the current CRR.

If structured provider output is rejected or malformed, the LLM adapter degrades through:

```text
json_schema -> json_object -> prompt-only JSON
```

and validates the recovered object locally. A verifier-format failure is surfaced as degraded verification rather than crashing artifact generation.

## Retrieval

Default retrieval combines:

```text
BM25 + Harrier dense retrieval -> Reciprocal Rank Fusion
```

An HF-compatible reranker can still be enabled with `RERANKER_API_URL`, but it is optional and disabled by default.

## Important runtime directories

- Python source artifact package: `artifacts/`
- Generated files: `runtime_artifacts/`
- Local GUI temporary uploads: `.local_gui_uploads/`
- Local GUI logs: `.runtime_logs/`

Do not change `PHASE6_OUTPUT_DIR` back to `artifacts`; that directory contains Python source code.

## Tests

Focused multimodal/runtime checks:

```powershell
python -m pytest -q `
  tests/test_phase2_ingestion.py `
  tests/test_hf_token_config.py `
  tests/test_hf_vlm_reranker.py `
  tests/test_phase5_verification.py `
  tests/test_phase6_artifacts.py
```

## Supported inputs

- `.txt`
- `.md`
- `.pdf`
- `.pptx`
- `.png`
- `.jpg` / `.jpeg`
- `.webp`
- `.bmp`
- `.tif` / `.tiff`

Legacy binary `.ppt` is intentionally unsupported; convert it to `.pptx` first.
