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


settings = Settings()
settings.model_artifact_dir.mkdir(parents=True, exist_ok=True)
