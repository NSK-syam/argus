---
title: Argus Backend
emoji: 🛰️
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Argus backend (FastAPI) — Hugging Face Space

Public-demo deployment of the Argus backend. Built from the exact same
`backend/Dockerfile` as the Render blueprint; this Space just adds the
public-demo safety defaults as environment variables (see `Dockerfile`).

Readiness: `GET /ready` (DB, FD001 files, demo bundle, seeded demo run,
loadable passing model). Liveness: `GET /health`.

Set `ARGUS_CORS_ORIGINS` in the Space's **Settings → Variables** to the
deployed frontend origin once it exists.
