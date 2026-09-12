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

# The committed demo bundle (app/demo_bundle/) holds models pickled with a
# specific scikit-learn. Free HF Spaces run Python 3.10, which forces an
# older scikit-learn than the repo pins (see requirements-py310.txt), and
# unpickling across versions is not something to trust for a judged demo.
# So: if the running version differs, rebuild the bundle in place before
# the app seeds from it (~1 min on the Space's CPU; the numbers it
# produces are the same deterministic reflection loop the docs describe).
BUNDLE_SKLEARN = "1.8.0"  # keep in sync with backend/requirements.txt
import sklearn  # noqa: E402

BUNDLE_DIR = ROOT / "app" / "demo_bundle"
if sklearn.__version__ != BUNDLE_SKLEARN:
    print(
        f"scikit-learn {sklearn.__version__} != bundle's {BUNDLE_SKLEARN}; "
        "rebuilding the demo bundle before seeding...",
        flush=True,
    )
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_demo_artifact.py")], check=True)

import gradio as gr  # noqa: E402

from app.db.session import init_db  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402  (the real backend)
from app.main import health, ready  # noqa: E402
from app.services.run_service import (  # noqa: E402
    cleanup_expired_uploads,
    recover_interrupted_runs,
    seed_demo_run,
)
from app.db.session import SessionLocal  # noqa: E402

# On the free (ZeroGPU) tier the HF runner launches the Gradio Blocks
# itself and refuses to start unless some Gradio event is wired to a
# @spaces.GPU function. Argus never touches a GPU, so the placeholder
# below is a no-op on a hidden button; outside HF the `spaces` package is
# absent and it's a plain function.
try:  # pragma: no cover - HF-only
    import spaces  # type: ignore

    _gpu = spaces.GPU
except ImportError:
    def _gpu(fn):  # type: ignore
        return fn


@_gpu
def _zerogpu_placeholder() -> str:
    return "Argus runs on CPU; this exists only to satisfy ZeroGPU startup."


LANDING = """
# Argus backend

This Space hosts the **FastAPI backend** for Argus, an agentic
predictive-maintenance ML studio (ABB Accelerator 2026 submission).
The user-facing app is the separate Next.js frontend; this page just
confirms the API is up.

The API is mounted under **`/backend`** on this host:

- Readiness: [`/backend/ready`](/backend/ready) — DB, FD001 data, demo
  bundle, seeded demo run, loadable passing model. 503 if anything is missing.
- Liveness: [`/backend/health`](/backend/health)
- Preloaded demo run: [`/backend/api/v1/runs/demo-seed-run`](/backend/api/v1/runs/demo-seed-run)
- OpenAPI docs: [`/backend/docs`](/backend/docs)

Live training runs and generic uploads are **disabled** on this public
deployment (`ARGUS_ENABLE_LIVE_RUNS` / `ARGUS_ENABLE_UPLOADS`); the
preloaded demo (promote → predict → replay) works fully.

Source: https://github.com/NSK-syam/argus
"""

with gr.Blocks(title="Argus backend") as demo:
    gr.Markdown(LANDING)
    _btn = gr.Button("zerogpu placeholder", visible=False)
    _out = gr.Textbox(visible=False)
    _btn.click(_zerogpu_placeholder, outputs=_out)


def _run_backend_startup() -> None:
    """What app.main's lifespan does. A FastAPI app mounted inside Gradio's
    server doesn't get its own lifespan events, so run them here."""
    init_db()
    db = SessionLocal()
    try:
        recover_interrupted_runs(db)
        seed_demo_run(db)
        cleanup_expired_uploads(db)
    finally:
        db.close()


# On the free HF tier the served Gradio app is not necessarily the one a
# __main__ block would get to touch after launch() (the ZeroGPU wrapper
# builds its own), so nothing done post-launch is reliable there. Instead,
# hook Gradio's own app factory: whichever process builds the
# Gradio FastAPI app gets the backend attached -- the whole API under
# /backend (no collisions with Gradio's /api/* routes) plus /ready and
# /health at the root.
_run_backend_startup()

from gradio import routes as _gr_routes  # noqa: E402

_orig_create_app = _gr_routes.App.create_app


def _create_app_with_backend(*args, **kwargs):
    # Gradio's own CORS middleware stands down when it sees a "parent" app
    # that configures CORS; pointing it at the backend makes
    # ARGUS_CORS_ORIGINS the single source of truth instead of stacking
    # two sets of Access-Control headers.
    kwargs.setdefault("parent_app", fastapi_app)
    server = _orig_create_app(*args, **kwargs)
    server.add_api_route("/ready", ready, methods=["GET"])
    server.add_api_route("/health", health, methods=["GET"])
    server.mount("/backend", fastapi_app)
    return server


_gr_routes.App.create_app = staticmethod(_create_app_with_backend)


if __name__ == "__main__":
    # Plain Gradio launch, exactly what the HF runner expects. Server name
    # and port come from GRADIO_SERVER_NAME / GRADIO_SERVER_PORT (HF sets
    # them; locally they default to 127.0.0.1:7860). The factory hook above
    # attaches the backend to whichever app this creates. ssr_mode=False:
    # on Spaces, Gradio otherwise puts a Node SSR server in front of the
    # Python app and that front answers unknown paths (like /backend/*)
    # with the SPA shell instead of forwarding them.
    demo.launch(ssr_mode=False)
