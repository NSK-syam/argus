"""Ties the pure ML core (orchestrator.py) to persistence (SQLAlchemy),
tracking (MLflow), and model artifacts -- none of which the ML core knows
anything about. A run executes in a background thread so
``POST /api/v1/runs`` can return immediately with a run id, and
``GET /api/v1/runs/{id}/events`` can stream progress by polling the
database (see ``app/api/runs.py``).

Job state is persisted BEFORE execution starts (status="pending" at
creation, "running" once the concurrency slot is acquired) — see
``app/main.py``'s startup handler for how a restart recovers from a run
stuck mid-flight.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy.orm import Session

from ..core.config import settings
from ..db import models
from ..db.session import SessionLocal
from ..ml.data import cmapss
from ..ml.pipeline import explain
from ..ml.pipeline import demo_bundle
from ..ml.pipeline.mlflow_logging import log_attempt
from ..ml.pipeline.orchestrator import LoggedAttempt, run_reflection_loop

logger = logging.getLogger(__name__)

# The build plan requires the public backend to run one training job at a
# time; settings.max_concurrent_runs defaults to 1.
_run_slot = threading.Semaphore(settings.max_concurrent_runs)

# Fixed IDs (not random uuids) for the seeded preloaded-demo dataset/run --
# lets the frontend link straight to /runs/{DEMO_RUN_ID} and lets
# seed_demo_run() check "does this already exist" with a plain db.get(),
# idempotent across restarts without a separate "is this the demo" flag.
DEMO_DATASET_ID = "demo-seed-dataset"
DEMO_RUN_ID = "demo-seed-run"


def create_dataset_from_bundled_fd001(db: Session) -> models.Dataset:
    train_df = cmapss.load_fd001_train(settings.data_dir)
    profile = cmapss.profile_dataset(train_df)
    dataset = models.Dataset(
        name="NASA C-MAPSS FD001 (bundled)",
        source="bundled_fd001",
        schema_json={"columns": cmapss.ALL_COLS},
        profile_json={
            "n_engines": profile.n_engines,
            "n_rows": profile.n_rows,
            "constant_sensors": profile.constant_sensors,
            "near_constant_sensors": profile.near_constant_sensors,
            "engine_life_stats": profile.engine_life_stats,
            "leakage_risk_notes": profile.leakage_risk_notes,
        },
        storage_path=str(settings.data_dir),
        n_rows=profile.n_rows,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def seed_demo_run(db: Session) -> bool:
    """Called once at startup (app/main.py's lifespan). Turns the
    precomputed demo bundle (app/demo_bundle/, built offline by
    scripts/build_demo_artifact.py) into a real, already-succeeded
    PipelineRun with real Attempt/ModelVersion rows and real model
    artifacts on disk -- so a judge who opens the app immediately sees the
    honest attempt-1-fails/attempt-2-passes retry, can promote the passing
    model, and can replay a real held-out engine, all without waiting for
    a live ~45-90s training run. This is exactly the build plan's own
    success criterion ("a judge can run the preloaded demonstration in
    under 90 seconds ... without waiting for training").

    Idempotent: fixed IDs mean a second call (e.g. a restart) is a no-op
    once the demo run exists. Never raises -- a missing/corrupt bundle
    just means no preloaded demo is offered; live runs are unaffected.
    """
    if db.get(models.PipelineRun, DEMO_RUN_ID) is not None:
        return True  # already seeded

    if not demo_bundle.bundle_exists(settings.demo_bundle_dir):
        logger.info("no demo bundle at %s -- skipping preloaded-demo seed", settings.demo_bundle_dir)
        return False

    try:
        bundle = demo_bundle.load_bundle(settings.demo_bundle_dir)
        train_df = cmapss.load_fd001_train(settings.data_dir)
        profile = cmapss.profile_dataset(train_df)

        dataset = models.Dataset(
            id=DEMO_DATASET_ID,
            name="NASA C-MAPSS FD001 (preloaded demo)",
            source="bundled_fd001",
            schema_json={"columns": cmapss.ALL_COLS},
            profile_json={
                "n_engines": profile.n_engines,
                "n_rows": profile.n_rows,
                "constant_sensors": profile.constant_sensors,
                "near_constant_sensors": profile.near_constant_sensors,
                "engine_life_stats": profile.engine_life_stats,
                "leakage_risk_notes": profile.leakage_risk_notes,
            },
            storage_path=str(settings.data_dir),
            n_rows=profile.n_rows,
        )
        db.add(dataset)

        run_result = bundle["run_result"] or {}
        run = models.PipelineRun(
            id=DEMO_RUN_ID,
            dataset_id=DEMO_DATASET_ID,
            goal="Predict remaining useful life (RUL) for FD001 turbofan engines.",
            status="succeeded",
            winning_attempt_number=run_result.get("promoted_attempt"),
            stopped_reason=run_result.get("stopped_reason"),
        )
        db.add(run)

        artifact_dir = settings.model_artifact_dir / DEMO_RUN_ID
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for entry in bundle["attempts"]:
            n = entry["attempt_number"]
            attempt = models.Attempt(
                run_id=DEMO_RUN_ID,
                attempt_number=n,
                plan_json={"feature_spec": entry["feature_spec"], "model_family": entry["model_family"]},
                model_family=entry["model_family"],
                hyperparams_json=entry["hyperparams"],
                validation_metrics_json=entry["validation_metrics"],
                test_metrics_json=entry["test_metrics"],
                conformal_coverage=entry["conformal_coverage"],
                gate_json=entry["gate"],
                gate_passed=entry["gate"]["passed"],
                revision_action=entry["revision_action"],
                revision_rationale=entry["revision_rationale"],
                plan_source=entry["plan_source"],
                fallback_reason=entry["fallback_reason"],
            )
            db.add(attempt)

            artifact_path = artifact_dir / f"attempt-{n}.joblib"
            joblib.dump(entry["model"], artifact_path)

            shap_summary: dict = {}
            try:
                if entry["background_sample"]:
                    bg = pd.DataFrame(entry["background_sample"])[entry["feature_columns"]]
                    explainer = explain.build_explainer(entry["model"], bg, entry["model_family"])
                    shap_summary = {"global_importance": explain.global_importance(explainer, bg)}
            except Exception as exc:  # noqa: BLE001 - a SHAP failure must not block seeding
                logger.warning("SHAP summary failed while seeding demo attempt %d: %s", n, exc)

            db.add(
                models.ModelVersion(
                    run_id=DEMO_RUN_ID,
                    attempt_number=n,
                    model_family=entry["model_family"],
                    artifact_path=str(artifact_path),
                    feature_columns_json=entry["feature_columns"],
                    conformal_q=entry["conformal_q"],
                    shap_summary_json=shap_summary,
                    trust_gate_passed=entry["gate"]["passed"],
                    stage="shadow",
                )
            )

        db.commit()
        logger.info("seeded preloaded demo run %s (%d attempts)", DEMO_RUN_ID, len(bundle["attempts"]))
        return True
    except Exception:
        db.rollback()
        logger.exception("failed to seed preloaded demo run -- continuing without it")
        return False


class RunQueueFullError(Exception):
    """Raised by start_run when too many PipelineRuns are already
    pending+running. Found in external code review: _run_slot only ever
    gated *active training* (the `with _run_slot:` block inside
    _execute_run below) -- nothing bounded how many background threads
    and DB rows POST /api/v1/runs could create while requests pile up
    waiting on that same slot, which is a real, unbounded-resource-growth
    DoS surface on a public demo. app/api/runs.py turns this into HTTP
    429."""


def start_run(db: Session, dataset: models.Dataset, goal: str) -> models.PipelineRun:
    in_flight = (
        db.query(models.PipelineRun)
        .filter(models.PipelineRun.status.in_(("pending", "running")))
        .count()
    )
    if in_flight >= settings.max_queued_runs:
        raise RunQueueFullError(
            f"{in_flight} run(s) already pending/running (limit {settings.max_queued_runs}); "
            "try again shortly"
        )

    run = models.PipelineRun(dataset_id=dataset.id, goal=goal, status="pending")
    db.add(run)
    db.commit()
    db.refresh(run)

    thread = threading.Thread(target=_execute_run, args=(run.id,), daemon=True)
    thread.start()
    return run


def _execute_run(run_id: str) -> None:
    db = SessionLocal()
    try:
        with _run_slot:
            run = db.get(models.PipelineRun, run_id)
            if run is None:
                return
            run.status = "running"
            db.commit()

            def on_attempt(logged: LoggedAttempt, model) -> None:
                _persist_attempt(db, run_id, logged, model)

            try:
                result = run_reflection_loop(settings.data_dir, on_attempt=on_attempt)
            except Exception as exc:  # noqa: BLE001 - a training crash must be recorded, not lost
                run = db.get(models.PipelineRun, run_id)
                run.status = "failed"
                run.error = f"{type(exc).__name__}: {exc}"
                db.commit()
                logger.exception("run %s failed", run_id)
                return

            run = db.get(models.PipelineRun, run_id)
            run.status = "succeeded"
            run.winning_attempt_number = result.promoted_attempt
            run.stopped_reason = result.stopped_reason
            db.commit()
    finally:
        db.close()


def _persist_attempt(db: Session, run_id: str, logged: LoggedAttempt, model) -> None:
    r = logged.result
    attempt = models.Attempt(
        run_id=run_id,
        attempt_number=r.attempt_number,
        plan_json={"feature_spec": r.feature_spec, "model_family": r.model_family},
        model_family=r.model_family,
        hyperparams_json=r.hyperparams,
        validation_metrics_json=r.validation_metrics,
        test_metrics_json=r.test_metrics,
        conformal_coverage=r.conformal_coverage,
        gate_json=logged.gate.as_evidence(),
        gate_passed=logged.gate.passed,
        revision_action=logged.revision_action,
        revision_rationale=logged.revision_rationale,
        plan_source=logged.plan_source,
        fallback_reason=logged.fallback_reason,
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)

    mlflow_run_id = log_attempt(settings.mlflow_tracking_uri, run_id, logged)
    if mlflow_run_id:
        attempt.mlflow_run_id = mlflow_run_id
        db.commit()

    artifact_dir = settings.model_artifact_dir / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"attempt-{r.attempt_number}.joblib"
    joblib.dump(model, artifact_path)

    shap_summary: dict = {}
    try:
        if r.background_sample:
            bg = pd.DataFrame(r.background_sample)[r.feature_columns]
            explainer = explain.build_explainer(model, bg, r.model_family)
            shap_summary = {"global_importance": explain.global_importance(explainer, bg)}
    except Exception as exc:  # noqa: BLE001 - explanation failures must not lose the model/run
        logger.warning("SHAP summary failed for run %s attempt %d: %s", run_id, r.attempt_number, exc)

    model_version = models.ModelVersion(
        run_id=run_id,
        attempt_number=r.attempt_number,
        model_family=r.model_family,
        artifact_path=str(artifact_path),
        feature_columns_json=r.feature_columns,
        conformal_q=r.conformal_q,
        shap_summary_json=shap_summary,
        trust_gate_passed=logged.gate.passed,
        stage="shadow",
    )
    db.add(model_version)
    db.commit()


def recover_interrupted_runs(db: Session) -> int:
    """Called on startup: a run stuck at status='running' means the
    process died mid-flight (crash, redeploy). Mark it failed with a
    clear reason rather than leaving it silently stuck forever -- the
    caller can retry by starting a new run against the same dataset."""
    stuck = db.query(models.PipelineRun).filter(models.PipelineRun.status == "running").all()
    for run in stuck:
        run.status = "failed"
        run.error = "interrupted_by_restart"
    if stuck:
        db.commit()
    return len(stuck)


def cleanup_expired_uploads(db: Session) -> int:
    """Called on startup (alongside recover_interrupted_runs): sweeps any
    uploaded dataset past its 24h expires_at, deleting both its stored
    file and DB row. Found in external code review that nothing enforced
    the 24h expiry promised in the upload response's `warning` field --
    app/api/datasets.py also enforces this lazily on read/list, but a
    dataset nobody ever requests again would otherwise sit on disk
    forever, so this closes that gap on every restart too."""
    now = datetime.now(timezone.utc)
    candidates = (
        db.query(models.Dataset)
        .filter(models.Dataset.source == "upload", models.Dataset.expires_at.isnot(None))
        .all()
    )
    # Filtered in Python, not SQL: SQLite doesn't reliably round-trip
    # tz-aware datetimes through SQLAlchemy's DateTime(timezone=True), so a
    # SQL-side "< now" comparison against a naive stored value is fragile.
    # Everything in this app writes expires_at in UTC, so a naive value
    # read back is treated as already UTC.
    expired = []
    for dataset in candidates:
        expires_at = dataset.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at < now:
            expired.append(dataset)
    for dataset in expired:
        with contextlib.suppress(OSError):
            Path(dataset.storage_path).unlink(missing_ok=True)
        db.delete(dataset)
    if expired:
        db.commit()
    return len(expired)
