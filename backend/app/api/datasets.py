from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..core.config import settings
from ..db import models
from ..db.session import get_db
from ..services import run_service

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


@router.post("")
def create_dataset(source: str = "bundled_fd001", db: Session = Depends(get_db)):
    if source != "bundled_fd001":
        raise HTTPException(
            400,
            "only source='bundled_fd001' is supported for starting a run today; "
            "use POST /api/v1/datasets/upload to store a CSV (profiling only, no run yet)",
        )
    dataset = run_service.create_dataset_from_bundled_fd001(db)
    return _dataset_out(dataset)


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Accepts and stores a CSV within the size/row caps from the build
    plan. Deliberately does NOT support starting a run against it yet --
    the reflection loop's feature engineering and trust gate are
    calibrated specifically for FD001's schema. Per the plan's own
    assumptions, generic uploads are a stretch goal kept only if the core
    FD001 demo is stable; this endpoint exists so the Dataset/upload
    entity and its limits are real and testable ahead of that work."""
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(413, f"file too large ({size_mb:.1f} MB > {settings.max_upload_mb} MB limit)")
    n_rows = max(0, contents.count(b"\n") - 1)  # rough estimate, minus header
    if n_rows > settings.max_upload_rows:
        raise HTTPException(413, f"too many rows (~{n_rows} > {settings.max_upload_rows} limit)")

    upload_dir = settings.model_artifact_dir.parent / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = str(uuid.uuid4())
    path = upload_dir / f"{dataset_id}.csv"
    path.write_bytes(contents)

    dataset = models.Dataset(
        id=dataset_id,
        name=file.filename or "uploaded.csv",
        source="upload",
        schema_json={},
        profile_json={
            "note": "Generic schema profiling and run execution against uploads is not yet "
            "implemented -- only the bundled FD001 dataset can start a run today."
        },
        storage_path=str(path),
        n_rows=n_rows,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return _dataset_out(dataset)


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(404, "dataset not found")
    return _dataset_out(dataset)


@router.get("")
def list_datasets(db: Session = Depends(get_db)):
    datasets = db.query(models.Dataset).order_by(models.Dataset.created_at.desc()).limit(50).all()
    return [_dataset_out(d) for d in datasets]


def _dataset_out(dataset: models.Dataset) -> dict:
    return {
        "id": dataset.id,
        "name": dataset.name,
        "source": dataset.source,
        "n_rows": dataset.n_rows,
        "profile": dataset.profile_json,
        "created_at": dataset.created_at.isoformat(),
        "expires_at": dataset.expires_at.isoformat() if dataset.expires_at else None,
        "warning": (
            "Do not upload confidential plant data. Uploaded datasets expire after 24 hours "
            "and cannot yet be used to start a run."
        )
        if dataset.source == "upload"
        else None,
    }
