"""Runs before any test module is imported: points the app at an isolated
temp SQLite DB / MLflow store / artifact dir instead of the real
backend/argus.db, so running the test suite never touches (or is affected
by) local dev state."""

import os
import tempfile
from pathlib import Path

_test_dir = Path(tempfile.mkdtemp(prefix="argus_test_"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_test_dir / 'test_argus.db'}")
os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{_test_dir / 'test_mlflow.db'}")
os.environ.setdefault("ARGUS_MODEL_ARTIFACT_DIR", str(_test_dir / "artifacts"))
