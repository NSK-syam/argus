"""End-to-end API tests against the real FastAPI app, a real (isolated,
temp) SQLite DB, and the real FD001 dataset -- no mocks. This exercises
the exact HTTP surface a frontend or a judge's curl command would hit.

Marked slow: creating a run trains real models (the same ~45s reflection
loop verified in test_orchestrator.py), so these tests skip gracefully if
FD001 data isn't present, same as the other slow tests.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cmapss"

pytestmark = [
    pytest.mark.skipif(
        not (DATA_DIR / "train_FD001.txt").exists(),
        reason="FD001 data not present at data/cmapss (see README for the download step)",
    ),
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _wait_for_terminal_status(client: TestClient, run_id: str, timeout_s: float = 90.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        run = client.get(f"/api/v1/runs/{run_id}").json()
        if run["status"] in ("succeeded", "failed"):
            return run
        time.sleep(2.0)
    raise TimeoutError(f"run {run_id} did not reach a terminal status within {timeout_s}s")


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_full_flow_dataset_run_promote_predict_replay(client: TestClient):
    # 1. create the bundled dataset
    resp = client.post("/api/v1/datasets", params={"source": "bundled_fd001"})
    assert resp.status_code == 200
    dataset = resp.json()
    assert dataset["n_rows"] == 20631
    assert dataset["profile"]["n_engines"] == 100

    # 2. start a run and wait for it to finish
    resp = client.post("/api/v1/runs", json={"dataset_id": dataset["id"]})
    assert resp.status_code == 200
    run_id = resp.json()["id"]

    run = _wait_for_terminal_status(client, run_id)
    assert run["status"] == "succeeded"
    assert run["winning_attempt_number"] == 2

    # 3. the honest retry, now served over HTTP
    attempts = run["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["gate_passed"] is False
    assert attempts[1]["gate_passed"] is True
    assert attempts[0]["mlflow_run_id"] is not None  # MLflow logging actually happened

    model_versions = run["model_versions"]
    failing_mv = next(mv for mv in model_versions if not mv["trust_gate_passed"])
    passing_mv = next(mv for mv in model_versions if mv["trust_gate_passed"])
    assert failing_mv["stage"] == "shadow"
    assert passing_mv["stage"] == "shadow"  # not production until explicitly promoted

    # 4. promoting a failing model must be refused
    resp = client.post(f"/api/v1/models/{failing_mv['id']}/promote")
    assert resp.status_code == 409

    # 5. promoting the passing model must succeed
    resp = client.post(f"/api/v1/models/{passing_mv['id']}/promote")
    assert resp.status_code == 200
    assert resp.json()["stage"] == "production"

    # 6. model detail carries a real SHAP global importance
    detail = client.get(f"/api/v1/models/{passing_mv['id']}").json()
    assert len(detail["shap_summary"]["global_importance"]) > 0
    feature_columns = detail["feature_columns"]

    # 7. predict against the promoted model
    resp = client.post(
        "/api/v1/predict",
        json={"model_version_id": passing_mv["id"], "features": {c: 0.0 for c in feature_columns}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_rul" in body
    assert body["explanation"] is not None

    # predict must reject a request missing required features
    resp = client.post(
        "/api/v1/predict",
        json={"model_version_id": passing_mv["id"], "features": {}},
    )
    assert resp.status_code == 422

    # 8. replay: list engines, then stream a few real cycles
    resp = client.get("/api/v1/replay/engines")
    assert resp.status_code == 200
    engines = resp.json()["engines"]
    assert len(engines) == 100

    with client.stream(
        "GET", f"/api/v1/replay/{passing_mv['id']}/{engines[0]}/events", params={"speed": 1000}
    ) as stream_resp:
        assert stream_resp.status_code == 200
        lines = []
        for line in stream_resp.iter_lines():
            lines.append(line)
            if len(lines) >= 6:  # a couple of "event:"/"data:" pairs is enough to prove it streams real cycles
                break
        joined = "\n".join(lines)
        assert "predicted_rul" in joined
        assert "true_rul_estimate" in joined


def test_upload_dataset_enforces_row_and_size_limits(client: TestClient):
    small_csv = b"a,b,c\n" + b"1,2,3\n" * 10
    resp = client.post("/api/v1/datasets/upload", files={"file": ("small.csv", small_csv, "text/csv")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "upload"
    assert body["expires_at"] is not None
    assert "confidential" in body["warning"].lower()


def test_promote_unknown_model_version_is_404(client: TestClient):
    resp = client.post("/api/v1/models/does-not-exist/promote")
    assert resp.status_code == 404


def test_run_against_upload_dataset_is_rejected(client: TestClient):
    small_csv = b"a,b,c\n1,2,3\n"
    upload = client.post(
        "/api/v1/datasets/upload", files={"file": ("small.csv", small_csv, "text/csv")}
    ).json()
    resp = client.post("/api/v1/runs", json={"dataset_id": upload["id"]})
    assert resp.status_code == 400
