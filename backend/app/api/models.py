from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import models
from ..db.session import get_db

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("/{model_version_id}")
def get_model_version(model_version_id: str, db: Session = Depends(get_db)):
    mv = db.get(models.ModelVersion, model_version_id)
    if mv is None:
        raise HTTPException(404, "model version not found")
    return _model_out(mv)


@router.post("/{model_version_id}/promote")
def promote_model(model_version_id: str, db: Session = Depends(get_db)):
    """One-click human-confirmed promotion. This is the ONLY way a model
    reaches 'production' -- passing the trust gate makes it eligible, it
    never auto-promotes itself. A failing model is refused outright."""
    mv = db.get(models.ModelVersion, model_version_id)
    if mv is None:
        raise HTTPException(404, "model version not found")
    if not mv.trust_gate_passed:
        raise HTTPException(
            409,
            "this model failed the trust gate and is not promotion_eligible; "
            "production exposure always requires a promotion_eligible model",
        )

    mv.stage = "production"
    deployment = models.Deployment(
        model_version_id=mv.id,
        stage="production",
        promoted_by="human_confirmed",
        audit_json={
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "trust_gate_passed": mv.trust_gate_passed,
            "conformal_q": mv.conformal_q,
        },
    )
    db.add(deployment)
    db.commit()
    db.refresh(mv)
    return _model_out(mv)


def _model_out(mv: models.ModelVersion) -> dict:
    return {
        "id": mv.id,
        "run_id": mv.run_id,
        "attempt_number": mv.attempt_number,
        "model_family": mv.model_family,
        "feature_columns": mv.feature_columns_json,
        "conformal_q": mv.conformal_q,
        "shap_summary": mv.shap_summary_json,
        "trust_gate_passed": mv.trust_gate_passed,
        "stage": mv.stage,
        "created_at": mv.created_at.isoformat(),
    }
