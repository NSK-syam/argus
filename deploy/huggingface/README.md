---
title: Argus Backend
emoji: 🛰️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.27.0
python_version: "3.11"
app_file: hf_space.py
pinned: false
license: mit
---

# Argus backend (FastAPI) — Hugging Face Space

Public-demo deployment of the Argus backend, served through a free Gradio
Space (Docker Spaces are a paid tier). `hf_space.py` fetches the pinned,
checksummed FD001 files at startup and serves the same `app.main:app` as
every other deployment, with a small landing page at `/`.

Readiness: `GET /ready` (DB, FD001 files, demo bundle, seeded demo run,
loadable passing model). Liveness: `GET /health`.

Set `ARGUS_CORS_ORIGINS` in the Space's **Settings → Variables** to the
deployed frontend origin once it exists, then restart the Space.
