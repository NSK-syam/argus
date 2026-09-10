"""FastAPI application entrypoint.

Placeholder for the Sep 14-17 build step. Only a health check is wired up
today; the real endpoints (/api/v1/datasets, /runs, /runs/{id}/events,
/models/{id}/promote, /predict, /replay/{engine_id}) land alongside the
SQLAlchemy models in app/db and the LangGraph-wired orchestrator.

The ML core these endpoints will call is already implemented and tested —
see app/ml/pipeline/orchestrator.py and backend/scripts/run_pipeline_demo.py
for a working, server-less demonstration in the meantime.
"""

from fastapi import FastAPI

app = FastAPI(title="Argus", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "argus-backend"}
