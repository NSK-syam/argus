"""Regression tests for a finding from deploying to a real public host.

CORS is advisory: the app tells the browser what to allow, so any layer in
front of the app can override it. `*.hf.space` does exactly that -- its
proxy echoes whatever `Origin` it is given, so `ARGUS_CORS_ORIGINS` was
being applied correctly by the app's own CORSMiddleware and then
overridden on the way out. Verified against the live deployment: a request
claiming `Origin: https://evil.example.com` came back with
`access-control-allow-origin: https://evil.example.com` and an
`access-control-expose-headers: *` this app never sets.

The fix is enforcement rather than advice: `enforce_allowed_origin` in
app/main.py refuses the request outright (403) before it reaches a route,
which no downstream proxy can undo. It inspects only `Origin` -- which
browsers attach to cross-origin requests and omit for same-origin ones --
so curl and server-to-server callers are unaffected.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import settings

ALLOWED = "https://argus-five-weld.vercel.app"


def test_no_enforcement_by_default_so_local_dev_is_unaffected(monkeypatch):
    """Default (and docker-compose/test) config is "*" -- any origin, and
    in particular no 403, so nothing about local dev changes."""
    from app.main import app

    monkeypatch.setattr(settings, "cors_allow_origins", ["*"])
    with TestClient(app) as client:
        resp = client.get("/health", headers={"Origin": "https://anything.example.com"})
        assert resp.status_code == 200


def test_disallowed_origin_is_refused_not_merely_unheadered(monkeypatch):
    from app.main import app

    monkeypatch.setattr(settings, "cors_allow_origins", [ALLOWED])
    with TestClient(app) as client:
        resp = client.get("/health", headers={"Origin": "https://evil.example.com"})
        assert resp.status_code == 403
        assert "evil.example.com" in resp.json()["detail"]

        # the API surface itself, not just the liveness probe
        resp = client.get(
            "/api/v1/runs/demo-seed-run", headers={"Origin": "https://evil.example.com"}
        )
        assert resp.status_code == 403


def test_allowed_origin_passes_through(monkeypatch):
    """Only pass-through is asserted, not the CORS response header:
    CORSMiddleware is constructed at import time from
    settings.cors_allow_origins, so monkeypatching the setting afterwards
    cannot change the header it emits -- whereas enforce_allowed_origin
    reads the setting per request and does see the patch. (The header
    itself is verified against the real deployment, where the env var is
    set before import: it returns access-control-allow-origin for the
    allowed origin only.)"""
    from app.main import app

    monkeypatch.setattr(settings, "cors_allow_origins", [ALLOWED])
    with TestClient(app) as client:
        resp = client.get("/health", headers={"Origin": ALLOWED})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_requests_without_an_origin_header_still_work(monkeypatch):
    """curl, server-to-server callers and same-origin browser requests send
    no Origin; they must not be caught by the allowlist."""
    from app.main import app

    monkeypatch.setattr(settings, "cors_allow_origins", [ALLOWED])
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
