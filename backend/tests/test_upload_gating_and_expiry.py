"""Regression tests for two real findings from an external code review of
the public repo:

1. The upload endpoint's 24-hour expiry was promised in the response
   `warning` field but nothing ever enforced it -- an uploaded file (and
   its DB row) would sit around indefinitely. Fix: any read of an expired
   upload dataset (single GET or list) deletes it and behaves as
   already-gone; a startup sweep (``cleanup_expired_uploads`` in
   ``run_service.py``) catches ones nobody reads again.

2. Generic uploads have no real production use yet (a run still can't be
   started against one) but were reachable on any deployment, including a
   public one accepting arbitrary files from strangers. Fix: a settings
   flag (``ARGUS_ENABLE_UPLOADS``, on by default so existing tests/local
   dev are unaffected) that the public deployment's render.yaml turns off.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.api.datasets import _expire_if_past_ttl
from app.core.config import settings
from app.db import models as db_models
from app.db.session import SessionLocal


def test_expired_upload_dataset_is_deleted_on_read_not_just_flagged(tmp_path):
    from app.main import app

    with TestClient(app) as client:
        # a real uploaded CSV, already past its expiry
        f = tmp_path / "old_upload.csv"
        f.write_text("a,b,c\n1,2,3\n")

        db = SessionLocal()
        try:
            dataset = db_models.Dataset(
                name="old_upload.csv",
                source="upload",
                storage_path=str(f),
                n_rows=1,
                expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
            )
            db.add(dataset)
            db.commit()
            db.refresh(dataset)
            dataset_id = dataset.id
        finally:
            db.close()

        assert f.exists()

        resp = client.get(f"/api/v1/datasets/{dataset_id}")
        assert resp.status_code == 404

        # the file itself was actually cleaned up, not just hidden from the API
        assert not f.exists()

        # and it no longer shows up in the list either
        listed_ids = {d["id"] for d in client.get("/api/v1/datasets").json()}
        assert dataset_id not in listed_ids


def test_non_expired_upload_dataset_is_left_alone():
    db = SessionLocal()
    try:
        dataset = db_models.Dataset(
            name="fresh.csv",
            source="upload",
            storage_path="/does-not-need-to-exist-for-this-check.csv",
            n_rows=1,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=23),
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        assert _expire_if_past_ttl(dataset, db) is False
    finally:
        db.close()


def test_upload_endpoint_can_be_disabled_via_settings(monkeypatch):
    from app.main import app

    monkeypatch.setattr(settings, "enable_uploads", False)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("small.csv", b"a,b,c\n1,2,3\n", "text/csv")},
        )
        assert resp.status_code == 403
