"""Central settings, read from environment variables with sane local
defaults so the app runs without docker-compose (SQLite, a local MLflow
file store) for development, tests, and this sandbox — and picks up
Postgres/hosted MLflow automatically when those env vars are set in
docker-compose / the real deployment.
"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings:
    database_url: str = os.environ.get("DATABASE_URL", f"sqlite:///{BACKEND_DIR / 'argus.db'}")
    # MLflow 3.x deprecated the plain filesystem store; sqlite is the
    # lightest backend that still works fully locally with no server.
    mlflow_tracking_uri: str = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{BACKEND_DIR / 'mlflow.db'}"
    )
    data_dir: Path = Path(os.environ.get("ARGUS_DATA_DIR", REPO_ROOT / "data" / "cmapss"))
    demo_bundle_dir: Path = Path(
        os.environ.get("ARGUS_DEMO_BUNDLE_DIR", BACKEND_DIR / "app" / "demo_bundle")
    )
    model_artifact_dir: Path = Path(
        os.environ.get("ARGUS_MODEL_ARTIFACT_DIR", BACKEND_DIR / "artifacts" / "models")
    )
    max_upload_mb: int = int(os.environ.get("ARGUS_MAX_UPLOAD_MB", "10"))
    max_upload_rows: int = int(os.environ.get("ARGUS_MAX_UPLOAD_ROWS", "50000"))
    anthropic_api_key: str | None = os.environ.get("ANTHROPIC_API_KEY")
    max_concurrent_runs: int = int(os.environ.get("ARGUS_MAX_CONCURRENT_RUNS", "1"))
    # Bounds how many PipelineRuns can be pending+running at once (across all
    # datasets), independent of max_concurrent_runs which only gates active
    # training. Without this, POST /api/v1/runs spawns an unbounded
    # background thread per request -- a real DoS surface on a public demo,
    # found in external code review. New requests are refused with 429 once
    # this many runs are queued/in-flight.
    max_queued_runs: int = int(os.environ.get("ARGUS_MAX_QUEUED_RUNS", "5"))
    # Comma-separated list of allowed CORS origins, e.g.
    # "https://argus-demo.vercel.app,https://argus.example.com". Defaults to
    # "*" to keep local dev/tests/docker-compose frictionless; a public
    # deployment (render.yaml) sets this to the real deployed frontend
    # origin(s) instead. Found in external code review: a public backend
    # with allow_origins=["*"] has no origin restriction at all.
    cors_allow_origins: list[str] = [
        o.strip() for o in os.environ.get("ARGUS_CORS_ORIGINS", "*").split(",") if o.strip()
    ]
    # Public-demo safety switch: the generic-upload endpoint accepts and
    # stores arbitrary CSVs with a 24h expiry that nothing previously
    # enforced (found in external code review). Uploads are never wired
    # into the training loop regardless of this flag (only the bundled
    # FD001 dataset can start a run) -- this only controls whether the
    # upload endpoint itself is reachable at all. Defaults on so existing
    # tests/local dev are unaffected; the real public deployment sets this
    # to false until expiry is enforced by a real cleanup job.
    enable_uploads: bool = os.environ.get("ARGUS_ENABLE_UPLOADS", "true").lower() not in (
        "false",
        "0",
        "",
    )
    # Public-demo safety switch for live training. The queue bound caps
    # concurrency, not cost: anyone could keep submitting expensive jobs
    # (and burn Claude credits if ANTHROPIC_API_KEY is set). Found in
    # external code review. Off in render.yaml; the preloaded demo run
    # (promote / predict / replay) works regardless. Defaults on for local
    # dev/tests.
    enable_live_runs: bool = os.environ.get("ARGUS_ENABLE_LIVE_RUNS", "true").lower() not in (
        "false",
        "0",
        "",
    )


settings = Settings()
settings.model_artifact_dir.mkdir(parents=True, exist_ok=True)
