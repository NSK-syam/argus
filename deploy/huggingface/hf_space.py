"""Hugging Face *Gradio* Space launcher for the Argus backend.

Docker Spaces are a paid tier on Hugging Face; Gradio Spaces are free and
simply run this file with Python. So: do at startup what the Dockerfile
does at build time (fetch the pinned, checksummed FD001 files), then serve
the real FastAPI app on the port HF expects (7860) with a small Gradio
landing page mounted at / so the Space shows something human-readable.

Every API route is unchanged: /ready, /health, /api/v1/... are served by
the same app.main:app as Render/docker-compose. The public-demo safety
defaults mirror render.yaml and can be overridden in the Space's
Settings -> Variables (e.g. ARGUS_CORS_ORIGINS once the frontend exists).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "cmapss"

# Safety defaults for a public URL (same as render.yaml). setdefault so the
# Space's own Variables win when set.
os.environ.setdefault("ARGUS_ENABLE_LIVE_RUNS", "false")
os.environ.setdefault("ARGUS_ENABLE_UPLOADS", "false")
os.environ.setdefault("ARGUS_DATA_DIR", str(DATA_DIR))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{ROOT / 'argus.db'}")
os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
os.environ.setdefault("ARGUS_MODEL_ARTIFACT_DIR", str(ROOT / "artifacts" / "models"))

required = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]
if not all((DATA_DIR / f).exists() for f in required):
    print("fetching NASA C-MAPSS FD001 (pinned commit, SHA-256 verified)...", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fetch_cmapss_data.py"), str(DATA_DIR)],
        check=True,
    )

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402

from app.main import app as fastapi_app  # noqa: E402  (the real backend)

LANDING = """
# Argus backend

This Space hosts the **FastAPI backend** for Argus, an agentic
predictive-maintenance ML studio (ABB Accelerator 2026 submission).
The user-facing app is the separate Next.js frontend; this page just
confirms the API is up.

- Readiness: [`/ready`](/ready) — DB, FD001 data, demo bundle, seeded demo
  run, loadable passing model. Returns 503 if anything is missing.
- Liveness: [`/health`](/health)
- Preloaded demo run: [`/api/v1/runs/demo-seed-run`](/api/v1/runs/demo-seed-run)
- OpenAPI docs: [`/docs`](/docs)

Live training runs and generic uploads are **disabled** on this public
deployment (`ARGUS_ENABLE_LIVE_RUNS` / `ARGUS_ENABLE_UPLOADS`); the
preloaded demo (promote → predict → replay) works fully.

Source: https://github.com/NSK-syam/argus
"""

with gr.Blocks(title="Argus backend") as demo:
    gr.Markdown(LANDING)

# Mounted at "/" AFTER the API routes are registered, so /ready, /health,
# /api/v1/* and /docs still win; everything else shows the landing page.
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
