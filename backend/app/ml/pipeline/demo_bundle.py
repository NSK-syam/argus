"""The precomputed demo artifact: a trained model + everything needed to
serve predictions/explanations from it, saved to disk so the live
demo/API can fall back to a known-good result if training, the API, or
hosting fails on the day it matters.

This is deliberately dumb (joblib + JSON, no database) so it has the
fewest possible moving parts between "works on my machine" and "works
during the judge's 90 seconds."
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from .orchestrator import PipelineRunResult


def save_bundle(
    bundle_dir: Path,
    model,
    winning_attempt_number: int,
    feature_columns: list[str],
    background_sample: list[dict],
    conformal_q: float,
    model_family: str,
    run_result: PipelineRunResult,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, bundle_dir / "model.joblib")

    metadata = {
        "winning_attempt_number": winning_attempt_number,
        "model_family": model_family,
        "feature_columns": feature_columns,
        "background_sample": background_sample,
        "conformal_q": conformal_q,
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (bundle_dir / "run_result.json").write_text(json.dumps(run_result.to_dict(), indent=2))


def bundle_exists(bundle_dir: Path) -> bool:
    return (bundle_dir / "model.joblib").exists() and (bundle_dir / "metadata.json").exists()


def load_bundle(bundle_dir: Path) -> dict:
    if not bundle_exists(bundle_dir):
        raise FileNotFoundError(f"no demo bundle at {bundle_dir} — run scripts/build_demo_artifact.py first")
    model = joblib.load(bundle_dir / "model.joblib")
    metadata = json.loads((bundle_dir / "metadata.json").read_text())
    run_result_path = bundle_dir / "run_result.json"
    run_result = json.loads(run_result_path.read_text()) if run_result_path.exists() else None
    return {"model": model, "metadata": metadata, "run_result": run_result}
