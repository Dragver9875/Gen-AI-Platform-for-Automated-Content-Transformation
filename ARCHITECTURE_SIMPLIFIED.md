# Simplified Runtime Architecture

## Model graph

```text
TXT/MD ------------------------------- direct text
IMAGE -------------------------------- Qwen2.5-VL
PDF -> PyMuPDF page rasterization ---- Qwen2.5-VL
PPTX -> python-pptx/Pillow raster ----- Qwen2.5-VL
                                         |
                                         v
                                  Unified Source IR
                                         |
                                      chunking
                                         |
                                      Harrier
                                         |
                                   Chroma Cloud
                                         |
                                     LangGraph
                                         |
                                   gpt-oss-20b
                         generation + verification + repair
                                         |
                text / Typst-PDF / PresentationIR / SVG
                                         |
                    deterministic file serializers

Creative image: gpt-oss-20b prompt -> FLUX.1-schnell
```

## Deliberately removed

- SigLIP visual routing
- document/photo classification branch
- Granite Docling
- MinerU/PaddleOCR dependency
- PDF native/scanned/mixed preflight routing
- dedicated verifier model/endpoint
- VLM fallback-model chain
- local ML model weights

## Why

The target is a hackathon/demo system where deployment reliability matters more than minimizing every provider call. Using one multimodal understanding model removes routing disagreement and endpoint fragmentation. TXT/MD remain direct because sending already-clean text through vision adds no information.

## Known trade-off

Every PDF page and every PPTX slide now incurs a Qwen2.5-VL call. This is intentionally simpler but slower/more expensive than selective visual fallback. `MULTIMODAL_MAX_PDF_PAGES` and `MULTIMODAL_MAX_PPTX_SLIDES` provide explicit caps for interactive testing.

## Evidence aliasing

Model-facing prompts never expose storage/vector chunk IDs. Retrieved chunks are assigned deterministic short aliases (`E1`, `E2`, ...). Phase 4 generation, Phase 5 verification, and repair use only those aliases; the application resolves them back to the real chunk IDs before persistence and deterministic verification. Unknown aliases are discarded rather than being treated as valid citations.

## Non-fatal repair failures

Phase 5 repair is best-effort. If the shared inference provider is unavailable, rate-limited, or out of quota during a repair call, the current CRR and verification report are retained and the run terminates as `phase5_complete_with_issues` rather than `error`. No unsupported claim is silently promoted to supported.
