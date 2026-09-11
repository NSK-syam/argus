"""The precomputed demo bundle: every attempt's trained model + everything
needed to serve predictions/explanations from it, saved to disk so the
API can seed a complete, already-finished run at startup -- so a judge
gets the full retry-evidence -> promote -> replay flow without waiting
for live training (the build plan's own success criterion: "a judge can
run the preloaded demonstration in under 90 seconds ... without waiting
for training").

Deliberately dumb (joblib + JSON, no database) so it has the fewest
possible moving parts between "works on my machine" and "works during the
judge's 90 seconds." app/services/run_service.py's seed_demo_run() is what
turns this into real DB rows at startup.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from .orchestrator import LoggedAttempt, PipelineRunResult


def save_bundle(
    bundle_dir: Path,
    attempts: list[tuple[LoggedAttempt, object]],
    run_result: PipelineRunResult,
) -> None:
    """``attempts`` is every (LoggedAttempt, fitted_model) pair from one
    real reflection-loop run, in order -- not just the winner, so the
    seeded demo run can show the same honest failed-then-passed retry a
    live run shows."""
    attempts_dir = bundle_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for logged, model in attempts:
        r = logged.result
        n = r.attempt_number
        joblib.dump(model, attempts_dir / f"attempt-{n}.joblib")
        manifest.append(
            {
                "attempt_number": n,
                "model_family": r.model_family,
                "feature_spec": r.feature_spec,
                "hyperparams": r.hyperparams,
                "feature_columns": r.feature_columns,
                "background_sample": r.background_sample,
                "conformal_q": r.conformal_q,
                "conformal_coverage": r.conformal_coverage,
                "validation_metrics": r.validation_metrics,
                "test_metrics": r.test_metrics,
                "gate": logged.gate.as_evidence(),
                "revision_action": logged.revision_action,
                "revision_rationale": logged.revision_rationale,
                "plan_source": logged.plan_source,
                "fallback_reason": logged.fallback_reason,
            }
        )

    (bundle_dir / "attempts_manifest.json").write_text(json.dumps(manifest, indent=2))
    (bundle_dir / "run_result.json").write_text(json.dumps(run_result.to_dict(), indent=2))


def bundle_exists(bundle_dir: Path) -> bool:
    return (bundle_dir / "attempts_manifest.json").exists() and any(
        (bundle_dir / "attempts").glob("attempt-*.joblib")
    )


def load_bundle(bundle_dir: Path) -> dict:
    """Returns {"attempts": [{"model": ..., **manifest_entry}], "run_result": {...}}."""
    if not bundle_exists(bundle_dir):
        raise FileNotFoundError(f"no demo bundle at {bundle_dir} — run scripts/build_demo_artifact.py first")
    manifest = json.loads((bundle_dir / "attempts_manifest.json").read_text())
    attempts = []
    for entry in manifest:
        model = joblib.load(bundle_dir / "attempts" / f"attempt-{entry['attempt_number']}.joblib")
        attempts.append({"model": model, **entry})
    run_result_path = bundle_dir / "run_result.json"
    run_result = json.loads(run_result_path.read_text()) if run_result_path.exists() else None
    return {"attempts": attempts, "run_result": run_result}
