"""Regression tests for two real findings from an external code review of
the public repo:

1. ``POST /predict`` never checked a model version's ``stage`` -- a shadow
   model (never promoted, possibly trust-gate-failing) could serve real
   predictions exactly like a production one. Fix: reject anything that
   isn't ``stage == "production"``.

2. ``POST /models/{id}/promote`` was neither idempotent (a repeat click
   created a fresh ``Deployment`` audit row every time) nor scoped
   correctly (nothing demoted a previously-production model when a
   different one was promoted, even though the frontend assumes at most
   one production model at a time via
   ``run.model_versions.find(mv => mv.stage === "production")``). Fix:
   re-promoting an already-production model is a no-op, and promoting a
   new model demotes every other currently-production model version.

Uses the app's demo-bundle-backed startup (no FD001 raw data download
required, unlike the slow end-to-end tests in test_api.py) and inserts
ModelVersion rows directly for a fast, deterministic, non-slow test.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.db import models as db_models
from app.db.session import SessionLocal


def _make_model_version(db, *, run_id: str, trust_gate_passed: bool, stage: str = "shadow"):
    mv = db_models.ModelVersion(
        run_id=run_id,
        attempt_number=1,
        model_family="random_forest",
        artifact_path="/nonexistent/does-not-matter-for-this-test.joblib",
        feature_columns_json=["sensor_2"],
        conformal_q=5.0,
        shap_summary_json={},
        trust_gate_passed=trust_gate_passed,
        stage=stage,
    )
    db.add(mv)
    db.commit()
    db.refresh(mv)
    return mv


def _make_run(db) -> str:
    dataset = db_models.Dataset(
        name="unit-test-dataset",
        source="bundled_fd001",
        storage_path="unused",
        n_rows=0,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    run = db_models.PipelineRun(dataset_id=dataset.id, status="succeeded")
    db.add(run)
    db.commit()
    db.refresh(run)
    return run.id


def test_predict_rejects_a_non_production_model_version():
    from app.main import app

    with TestClient(app) as client:
        db = SessionLocal()
        try:
            run_id = _make_run(db)
            shadow_mv = _make_model_version(db, run_id=run_id, trust_gate_passed=True, stage="shadow")
        finally:
            db.close()

        resp = client.post(
            "/api/v1/predict",
            json={"model_version_id": shadow_mv.id, "features": {"sensor_2": 0.0}},
        )
        assert resp.status_code == 403
        assert "production" in resp.json()["detail"].lower()


def test_promote_is_idempotent_and_demotes_other_production_models():
    from app.main import app

    with TestClient(app) as client:
        db = SessionLocal()
        try:
            run_id = _make_run(db)
            first_id = _make_model_version(db, run_id=run_id, trust_gate_passed=True, stage="shadow").id
            second_id = _make_model_version(db, run_id=run_id, trust_gate_passed=True, stage="shadow").id
        finally:
            db.close()

        # promote the first model
        resp = client.post(f"/api/v1/models/{first_id}/promote")
        assert resp.status_code == 200
        assert resp.json()["stage"] == "production"

        # re-promoting the SAME model must be a no-op, not an error and not
        # a fresh audit event every time
        resp = client.post(f"/api/v1/models/{first_id}/promote")
        assert resp.status_code == 200
        assert resp.json()["stage"] == "production"

        db = SessionLocal()
        try:
            deployments_for_first = (
                db.query(db_models.Deployment)
                .filter(db_models.Deployment.model_version_id == first_id)
                .all()
            )
            # exactly one promotion event, even after two calls
            assert len(deployments_for_first) == 1
        finally:
            db.close()

        # promoting a DIFFERENT model version must demote the first one --
        # the frontend assumes at most one production model at a time
        resp = client.post(f"/api/v1/models/{second_id}/promote")
        assert resp.status_code == 200
        assert resp.json()["stage"] == "production"

        resp = client.get(f"/api/v1/models/{first_id}")
        assert resp.json()["stage"] == "shadow"
