from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import models
from ..db.session import SessionLocal, get_db
from ..services import run_service

router = APIRouter(prefix="/api/v1/runs", tags=["runs"])


class CreateRunRequest(BaseModel):
    dataset_id: str
    goal: str = "Predict remaining useful life (RUL) for FD001 turbofan engines."


@router.post("")
def create_run(req: CreateRunRequest, db: Session = Depends(get_db)):
    dataset = db.get(models.Dataset, req.dataset_id)
    if dataset is None:
        raise HTTPException(404, "dataset not found")
    if dataset.source != "bundled_fd001":
        raise HTTPException(400, "starting a run is currently only supported for the bundled FD001 dataset")
    try:
        run = run_service.start_run(db, dataset, req.goal)
    except run_service.LiveRunsDisabledError as exc:
        raise HTTPException(403, str(exc)) from exc
    except run_service.RunQueueFullError as exc:
        raise HTTPException(429, str(exc)) from exc
    return _run_out(run, db)


@router.post("/{run_id}/retry")
def retry_run(run_id: str, db: Session = Depends(get_db)):
    """Start a fresh run against the same dataset/goal as a failed one --
    the 'retry option' the build plan requires after a restart-interrupted
    run, rather than the original run silently staying stuck."""
    prior = db.get(models.PipelineRun, run_id)
    if prior is None:
        raise HTTPException(404, "run not found")
    if prior.status not in ("failed",):
        raise HTTPException(409, f"only a failed run can be retried (status is '{prior.status}')")
    dataset = db.get(models.Dataset, prior.dataset_id)
    try:
        new_run = run_service.start_run(db, dataset, prior.goal)
    except run_service.LiveRunsDisabledError as exc:
        raise HTTPException(403, str(exc)) from exc
    except run_service.RunQueueFullError as exc:
        raise HTTPException(429, str(exc)) from exc
    return _run_out(new_run, db)


@router.get("/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.get(models.PipelineRun, run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return _run_out(run, db)


@router.get("")
def list_runs(db: Session = Depends(get_db)):
    runs = db.query(models.PipelineRun).order_by(models.PipelineRun.created_at.desc()).limit(50).all()
    return [_run_out(r, db) for r in runs]


@router.get("/{run_id}/events")
async def run_events(run_id: str):
    """Server-Sent Events: poll the DB for new attempts and status changes
    until the run reaches a terminal state. Persisted state is the source
    of truth either way -- a client that reconnects mid-run just re-reads
    GET /runs/{id} and catches up; polling a database is a deliberate
    simplicity trade-off over a true pub/sub for this prototype."""

    async def event_stream():
        seen_attempts = 0
        db = SessionLocal()
        try:
            while True:
                run = db.get(models.PipelineRun, run_id)
                if run is None:
                    yield _sse("error", {"detail": "run not found"})
                    return
                db.refresh(run)
                attempts = (
                    db.query(models.Attempt)
                    .filter(models.Attempt.run_id == run_id)
                    .order_by(models.Attempt.attempt_number)
                    .all()
                )
                for attempt in attempts[seen_attempts:]:
                    yield _sse("attempt", _attempt_out(attempt))
                seen_attempts = len(attempts)

                if run.status in ("succeeded", "failed"):
                    yield _sse("run_status", _run_out(run, db))
                    return

                yield _sse("run_status", {"status": run.status})
                await asyncio.sleep(1.0)
        finally:
            db.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _attempt_out(attempt: models.Attempt) -> dict:
    return {
        "attempt_number": attempt.attempt_number,
        "model_family": attempt.model_family,
        "hyperparams": attempt.hyperparams_json,
        "validation_metrics": attempt.validation_metrics_json,
        "test_metrics": attempt.test_metrics_json,
        "conformal_coverage": attempt.conformal_coverage,
        "gate": attempt.gate_json,
        "gate_passed": attempt.gate_passed,
        "revision_action": attempt.revision_action,
        "revision_rationale": attempt.revision_rationale,
        "plan_source": attempt.plan_source,
        "fallback_reason": attempt.fallback_reason,
        "mlflow_run_id": attempt.mlflow_run_id,
    }


def _run_out(run: models.PipelineRun, db: Session) -> dict:
    attempts = (
        db.query(models.Attempt)
        .filter(models.Attempt.run_id == run.id)
        .order_by(models.Attempt.attempt_number)
        .all()
    )
    model_versions = (
        db.query(models.ModelVersion)
        .filter(models.ModelVersion.run_id == run.id)
        .order_by(models.ModelVersion.attempt_number)
        .all()
    )
    return {
        "id": run.id,
        "dataset_id": run.dataset_id,
        "goal": run.goal,
        "status": run.status,
        "winning_attempt_number": run.winning_attempt_number,
        "stopped_reason": run.stopped_reason,
        "error": run.error,
        "attempts": [_attempt_out(a) for a in attempts],
        "model_versions": [
            {"id": mv.id, "attempt_number": mv.attempt_number, "stage": mv.stage, "trust_gate_passed": mv.trust_gate_passed}
            for mv in model_versions
        ],
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
    }
