Query transport synchronization hotfix
=====================================

Copy/extract these files into the repository root, preserving folders and replacing existing files:

  run_ps.ps1
  scripts/run_phase6.py
  scripts/query_transport.py
  ingestion/preflight.py
  providers/docling.py

Why:
- run_ps.ps1 sends --query-b64 so quoted/Unicode Windows queries are safe.
- scripts/run_phase6.py must accept --query-b64 and decode it.
- scripts/query_transport.py performs UTF-8 Base64 transport.
- ingestion/preflight.py and providers/docling.py use `import pymupdf` instead of deprecated `import fitz`.

After copying, from repository root run:

  Unblock-File .\run_ps.ps1
  .\.venv\Scripts\python.exe -m scripts.run_phase6 --help
  .\run_ps.ps1

The help output must contain:
  --query-b64 QUERY_B64
