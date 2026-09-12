"""Regression test for a real finding from an external code review:
``_run_slot`` (a threading.Semaphore) only ever gated *active training*
inside ``_execute_run`` -- nothing stopped ``POST /api/v1/runs`` from
creating an unbounded number of pending ``PipelineRun`` rows and
background threads while requests piled up waiting on that same slot, a
real unbounded-resource-growth DoS surface on a public demo. Fix: bound
how many runs can be pending+running at once (``ARGUS_MAX_QUEUED_RUNS``)
and refuse new ones with HTTP 429 once that's reached.

Uses direct DB rows (no real training) to make this fast and deterministic.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import settings
from app.db import models as db_models
from app.db.session import SessionLocal
from app.services import run_service


def _make_dataset(db) -> str:
    dataset = db_models.Dataset(
        name="queue-test-dataset",
        source="bundled_fd001",
        storage_path="unused",
        n_rows=0,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset.id


def test_start_run_is_refused_once_the_queue_bound_is_reached(monkeypatch):
    monkeypatch.setattr(settings, "max_queued_runs", 2)

    from app.main import app

    with TestClient(app):  # triggers lifespan's init_db(); no HTTP calls needed
        db = SessionLocal()
        try:
            dataset_id = _make_dataset(db)
            # fabricate two runs already "in flight" without any real training
            db.add(db_models.PipelineRun(dataset_id=dataset_id, status="pending"))
            db.add(db_models.PipelineRun(dataset_id=dataset_id, status="running"))
            db.commit()
            dataset = db.get(db_models.Dataset, dataset_id)

            try:
                run_service.start_run(db, dataset, "test goal")
                assert False, "expected RunQueueFullError once the bound is reached"
            except run_service.RunQueueFullError:
                pass
        finally:
            db.close()


def test_create_run_endpoint_returns_429_when_queue_is_full(monkeypatch):
    monkeypatch.setattr(settings, "max_queued_runs", 1)

    from app.main import app

    with TestClient(app) as client:
        db = SessionLocal()
        try:
            dataset_id = _make_dataset(db)
            db.add(db_models.PipelineRun(dataset_id=dataset_id, status="running"))
            db.commit()
        finally:
            db.close()

        resp = client.post("/api/v1/runs", json={"dataset_id": dataset_id})
        assert resp.status_code == 429
