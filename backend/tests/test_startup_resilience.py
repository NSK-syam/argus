"""Regression tests for findings from a second external code review of the
public repo, all about what happens across a restart/redeploy on a
persistent-DB + ephemeral-disk deployment:

1. seed_demo_run() returned early once the demo run's DB row existed and
   never recreated the model artifacts a redeploy had wiped -- so /ready
   passed while /predict and /replay failed. Fix: rehydrate missing demo
   artifacts from the bundle on every startup, and make /ready require a
   passing demo model whose artifact exists AND loads.

2. recover_interrupted_runs() only failed 'running' rows; a 'pending' row
   whose worker thread died with the old process stayed pending forever
   and permanently consumed queue capacity. Fix: recover both.

3. The queue-bound check was count-then-insert with no lock, so concurrent
   requests could all pass the check and all insert. Fix: a process-wide
   lock around check+insert.

4. Live training was reachable by anyone on a public URL (the queue bound
   caps concurrency, not total cost / Claude credits). Fix:
   ARGUS_ENABLE_LIVE_RUNS, off in render.yaml.
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import settings
from app.db import models as db_models
from app.db.session import SessionLocal
from app.services import run_service


def test_missing_demo_artifacts_are_rehydrated_on_startup():
    from app.main import app

    # first startup seeds the demo run and writes artifacts
    with TestClient(app):
        pass

    db = SessionLocal()
    try:
        versions = (
            db.query(db_models.ModelVersion)
            .filter(db_models.ModelVersion.run_id == run_service.DEMO_RUN_ID)
            .all()
        )
        assert versions, "demo run should have been seeded"
        paths = [Path(mv.artifact_path) for mv in versions]
    finally:
        db.close()

    # simulate an ephemeral-disk redeploy: DB rows survive, files don't
    for p in paths:
        p.unlink()
    assert not any(p.exists() for p in paths)

    with TestClient(app) as client:
        assert all(p.exists() for p in paths), "artifacts must be re-dumped from the bundle"
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["checks"]["demo_model_artifact_loadable"] is True


def test_ready_fails_when_demo_artifacts_are_missing_and_cannot_be_restored(monkeypatch):
    from app.main import app

    with TestClient(app) as client:
        db = SessionLocal()
        try:
            versions = (
                db.query(db_models.ModelVersion)
                .filter(db_models.ModelVersion.run_id == run_service.DEMO_RUN_ID)
                .all()
            )
            paths = [Path(mv.artifact_path) for mv in versions]
        finally:
            db.close()

        # rehydration only runs at startup; deleting mid-life must be
        # caught by /ready rather than reported as healthy
        for p in paths:
            p.unlink()
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["detail"]["checks"]["demo_model_artifact_loadable"] is False

    # restart restores them again so later tests in this process are unaffected
    with TestClient(app):
        assert all(p.exists() for p in paths)


def test_recover_interrupted_runs_fails_stuck_pending_rows_too():
    from app.main import app

    with TestClient(app):
        db = SessionLocal()
        try:
            dataset = db_models.Dataset(name="d", source="bundled_fd001", storage_path="x", n_rows=0)
            db.add(dataset)
            db.commit()
            pending = db_models.PipelineRun(dataset_id=dataset.id, status="pending")
            running = db_models.PipelineRun(dataset_id=dataset.id, status="running")
            db.add_all([pending, running])
            db.commit()
            pending_id, running_id = pending.id, running.id

            recovered = run_service.recover_interrupted_runs(db)
            assert recovered >= 2
            assert db.get(db_models.PipelineRun, pending_id).status == "failed"
            assert db.get(db_models.PipelineRun, running_id).status == "failed"
        finally:
            db.close()


def test_queue_bound_holds_under_concurrent_requests(monkeypatch):
    """Many threads racing start_run must never exceed the bound. Training
    is stubbed out so this stays fast; only the admission logic is under
    test."""
    from app.main import app

    monkeypatch.setattr(settings, "max_queued_runs", 3)
    monkeypatch.setattr(run_service, "_execute_run", lambda run_id: None)

    with TestClient(app):
        db = SessionLocal()
        try:
            # a clean slate: fail anything left over from other tests
            run_service.recover_interrupted_runs(db)
            dataset = db_models.Dataset(name="d", source="bundled_fd001", storage_path="x", n_rows=0)
            db.add(dataset)
            db.commit()
            dataset_id = dataset.id
        finally:
            db.close()

        admitted, refused = [], []
        barrier = threading.Barrier(12)

        def worker():
            db = SessionLocal()
            try:
                ds = db.get(db_models.Dataset, dataset_id)
                barrier.wait()
                try:
                    run_service.start_run(db, ds, "race")
                    admitted.append(1)
                except run_service.RunQueueFullError:
                    refused.append(1)
            finally:
                db.close()

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(admitted) == 3
        assert len(refused) == 9


def test_live_runs_can_be_disabled_for_public_deployment(monkeypatch):
    from app.main import app

    monkeypatch.setattr(settings, "enable_live_runs", False)
    with TestClient(app) as client:
        dataset = client.post("/api/v1/datasets", params={"source": "bundled_fd001"}).json()
        resp = client.post("/api/v1/runs", json={"dataset_id": dataset["id"]})
        assert resp.status_code == 403
        assert "demo-seed-run" in resp.json()["detail"]

        # the preloaded demo is unaffected
        assert client.get("/api/v1/runs/demo-seed-run").json()["status"] == "succeeded"
