"""Regression test for a real finding from an external code review:
``/health`` returned 200 unconditionally, even if the DB were unreachable
or the demo bundle were missing -- exactly the state where a judge's first
click ("already trained, no waiting") would fail, while Render's own health
check kept reporting the deploy as healthy. Fix: a ``/ready`` endpoint that
actually checks the three things that first click depends on, returning
503 (not 200) if any of them fail.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_ready_reports_ok_when_demo_bundle_and_db_and_seed_are_present():
    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["checks"]["database_reachable"] is True
        assert body["checks"]["demo_bundle_present"] is True
        assert body["checks"]["demo_run_seeded"] is True


def test_health_stays_a_plain_liveness_check():
    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
