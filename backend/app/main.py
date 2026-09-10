"""FastAPI application entrypoint.

Wires together the DB, the ML core (via app/services/run_service.py), and
the API routers described in the build plan: datasets, runs (+SSE
events), model promotion, predict, and held-out engine replay (+SSE).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import datasets, models as models_api, predict, replay, runs
from .db.session import SessionLocal, init_db
from .services.run_service import recover_interrupted_runs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        recovered = recover_interrupted_runs(db)
        if recovered:
            logger.warning(
                "recovered %d run(s) interrupted by a restart -- marked failed, retry via POST /api/v1/runs/{id}/retry",
                recovered,
            )
    finally:
        db.close()
    yield


app = FastAPI(title="Argus", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # narrow this to the deployed frontend origin before shipping
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "argus-backend"}


app.include_router(datasets.router)
app.include_router(runs.router)
app.include_router(models_api.router)
app.include_router(predict.router)
app.include_router(replay.router)
