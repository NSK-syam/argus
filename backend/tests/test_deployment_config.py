"""Regression test for a UX finding from the live public deployment.

The public demo runs with ARGUS_ENABLE_LIVE_RUNS=false, which is
deliberate -- a free-tier box shouldn't let strangers queue real training
jobs. But the frontend had no way to know that, so its most prominent
button ("Start pipeline run") looked enabled, and clicking it produced a
red `403 "live training runs are disabled on this deployment"` banner. To
a judge that reads as a broken app, not a safety setting.

GET /api/v1/config reports what the deployment allows so a client can
reflect it up front. Kept deliberately trivial: two booleans and the demo
run id, no auth, nothing a public caller couldn't already infer by
receiving a 403.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import settings


def test_config_reports_enabled_by_default():
    from app.main import app

    with TestClient(app) as client:
        body = client.get("/api/v1/config").json()
        assert body["live_runs_enabled"] is True
        assert body["uploads_enabled"] is True
        assert body["demo_run_id"] == "demo-seed-run"


def test_config_reflects_the_public_demo_settings(monkeypatch):
    """What the deployed Space actually reports."""
    from app.main import app

    monkeypatch.setattr(settings, "enable_live_runs", False)
    monkeypatch.setattr(settings, "enable_uploads", False)
    with TestClient(app) as client:
        body = client.get("/api/v1/config").json()
        assert body["live_runs_enabled"] is False
        assert body["uploads_enabled"] is False


def test_config_agrees_with_the_endpoint_it_describes(monkeypatch):
    """The point of the endpoint is that it predicts the 403, so assert the
    two can't drift apart."""
    from app.main import app

    monkeypatch.setattr(settings, "enable_live_runs", False)
    with TestClient(app) as client:
        assert client.get("/api/v1/config").json()["live_runs_enabled"] is False

        dataset = client.post("/api/v1/datasets", params={"source": "bundled_fd001"}).json()
        resp = client.post("/api/v1/runs", json={"dataset_id": dataset["id"]})
        assert resp.status_code == 403
